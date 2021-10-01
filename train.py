import wandb
from tqdm import tqdm
import torchgeometry as tgm

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda import amp
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from train_datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator, MultiscaleDiscriminator, GANLoss
from utils import cleanup, gen_noise, seed_everything, load_checkpoint, synchronize, AverageMeter
from train_options import get_args


class TrainModel:
    def __init__(self, args):
        self.device = torch.device('cuda', args.local_rank)
        self.segG = SegGenerator(args, input_nc=args.semantic_nc + 8, output_nc=args.semantic_nc).to(self.device).train()
        self.segD = MultiscaleDiscriminator(args, args.semantic_nc + args.semantic_nc + 8, use_sigmoid=args.no_lsgan).to(self.device).train()
        # self.gmm = GMM(args, inputA_nc=7, inputB_nc=3).to(self.device).train()
        # args.semantic_nc = 7
        # alias = ALIASGenerator(args, input_nc=9).to(self.device).train()
        # args.semantic_nc = 13

        # load_checkpoint(self.segG, os.path.join(args.checkpoint_dir, args.seg_checkpoint))
        # load_checkpoint(self.gmm, os.path.join(args.checkpoint_dir, args.gmm_checkpoint))
        # load_checkpoint(alias, os.path.join(args.checkpoint_dir, args.alias_checkpoint))

        # self.gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(self.device)
        # self.up = nn.Upsample(size=(args.load_height, args.load_width), mode='bilinear')

        self.train_dataset = VITONDataset(args)
        self.train_loader = VITONDataLoader(args, self.train_dataset)

        self.criterion_gan = GANLoss(use_lsgan=not args.no_lsgan)
        self.ce_loss = nn.CrossEntropyLoss()

        self.optimizer_seg = optim.Adam(list(self.segG.parameters()) + list(self.segD.parameters()),
                                        lr=0.0004, betas=(0.5,0.999))
        self.optimizer_seg.zero_grad(set_to_none=True)

        self.scaler = amp.GradScaler(enabled=args.use_amp)

        self.parse_labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],   # contains face and lower body.
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
        if args.num_gpus > 1:
            if args.sync_bn:
                self.segG = nn.SyncBatchNorm.convert_sync_batchnorm(self.segG)
                self.segD = nn.SyncBatchNorm.convert_sync_batchnorm(self.segD)
                # self.gmm = nn.SyncBatchNorm.convert_sync_batchnorm(self.gmm)
                # self.alias = nn.SyncBatchNorm.convert_sync_batchnorm(self.alias)
            self.segG = DDP(self.segG, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
            self.segD = DDP(self.segD, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
            # self.gmm = DDP(self.gmm, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
            # self.alias = DDP(self.alias, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
        if args.local_rank==0 and args.use_wandb:
            wandb.watch([self.segG, self.segD], log=None)
    
    def train_epoch(self, args, epoch):
        if args.num_gpus > 1: self.train_loader.train_sampler.set_epoch(epoch)
        segG_losses = AverageMeter()
        segD_losses = AverageMeter()
        with tqdm(enumerate(self.train_loader.data_loader), desc=f"Epoch {epoch:>2}") as pbar:
            for step, batch in pbar:
                # img = batch['img'].to(self.device, non_blocking=True)
                # img_agnostic = batch['img_agnostic'].to(self.device, non_blocking=True)
                parse_target_down = batch['parse_target_down'].to(self.device, non_blocking=True)
                parse_agnostic = batch['parse_agnostic'].to(self.device, non_blocking=True)
                pose = batch['pose'].to(self.device, non_blocking=True)
                c = batch['cloth'].to(self.device, non_blocking=True)
                cm = batch['cloth_mask'].to(self.device, non_blocking=True)

                seg_lossG, seg_lossD = self.segmentation_train_step(args, parse_target_down, parse_agnostic, pose, c, cm)

                segG_losses.update(seg_lossG.detach_(), len(batch))
                segD_losses.update(seg_lossD.detach_(), len(batch))
                if args.local_rank == 0:
                    if not step % args.log_interval:
                        info = {'SegG Loss': float(segG_losses.avg), 'SegD Loss': float(segD_losses.avg)}
                        if args.use_wandb: wandb.log(info)
                        pbar.set_postfix(info)
                self.scaler.update()
    
    def segmentation_train_step(self, args, parse_target_down, parse_agnostic, pose, c, cm):
        with amp.autocast(enabled=args.use_amp):
            # Part 1. Segmentation generation
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size(), device=self.device)), dim=1)

            parse_pred_down = self.segG(seg_input)

            lambda_ce = 10
            seg_lossG = lambda_ce * self.ce_loss(parse_pred_down, parse_target_down.argmax(dim=1))
            
            fake_out = self.segD(torch.cat((seg_input, parse_pred_down.detach()), dim=1))
            real_out = self.segD(torch.cat((seg_input, parse_target_down.detach()), dim=1))
            seg_lossG += self.criterion_gan(fake_out, True)                          # Treat fake images as real to train the Generator.
            seg_lossD = (self.criterion_gan(real_out, True)     # Treat real as real
                        + self.criterion_gan(fake_out, False))   # and fake as fake to train Discriminator.

        self.segD.requires_grad_(False)
        self.scaler.scale(seg_lossG).backward(retain_graph=True)
        self.segD.requires_grad_(True)
        self.segG.requires_grad_(False)
        self.scaler.scale(seg_lossD).backward()
        self.segG.requires_grad_(True)

        self.scaler.step(self.optimizer_seg)
        self.optimizer_seg.zero_grad(set_to_none=True)
        return seg_lossG.detach_(), seg_lossG.detach_()


    def gmm_train_step(self, args, img, img_agnostic, parse_target_down, pose, c, cm):
        # convert 13 channel body parse to 7 channel parse.
        parse_target = self.gauss(self.up(parse_target_down))
        parse_target = parse_target.argmax(dim=1, keepdim=True)
        parse_old = torch.zeros(parse_target.size(0), 13, args.load_height, args.load_width, dtype=torch.float32, device=self.device)
        parse_old.scatter_(1, parse_target, 1.0)

        parse = torch.zeros(parse_target.size(0), 7, args.load_height, args.load_width, dtype=torch.float32, device=self.device)
        for j in range(len(self.parse_labels)):
            for lbl in self.parse_labels[j][1]:
                parse[:, j] += parse_old[:, lbl]

        # Part 2. Clothes Deformation
        agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
        parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
        pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
        c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
        gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

        _, warped_grid = self.gmm(gmm_input, c_gmm)
        warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
        warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')
        return


    def alias_train_step(self, args, img, img_agnostic, parse, pose, warped_c, warped_cm):
        # Part 3. Try-on synthesis
        misalign_mask = parse[:, 2:3] - warped_cm
        misalign_mask[misalign_mask < 0.0] = 0.0
        parse_div = torch.cat((parse, misalign_mask), dim=1)
        parse_div[:, 2:3] -= misalign_mask

        output = self.alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)
        return


def main():
    args = get_args()
    print(args)
    seed_everything(args.seed)
    try:
        tm = TrainModel(args)
        tm.train()
    except KeyboardInterrupt:
        print("[!] Keyboard Interrupt! Cleaning up and shutting down.")
    finally:
        cleanup(args.num_gpus > 1)


if __name__ == '__main__':
    main()
