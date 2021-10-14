import os
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
        self.memory_format = torch.channels_last if args.memory_format == "channels_last" else torch.contiguous_format
        self.segG = SegGenerator(args, input_nc=args.semantic_nc + 8, output_nc=args.semantic_nc).train().to(self.device, memory_format=self.memory_format)
        self.segD = MultiscaleDiscriminator(args, args.semantic_nc + args.semantic_nc + 8, use_sigmoid=args.no_lsgan).train().to(self.device, memory_format=self.memory_format)
        # self.gmm = GMM(args, inputA_nc=7, inputB_nc=3).train().to(self.device, memory_format=self.memory_format)
        # args.semantic_nc = 7
        # alias = ALIASGenerator(args, input_nc=9).train().to(self.device, memory_format=self.memory_format)
        # args.semantic_nc = 13

        # load_checkpoint(self.segG, os.path.join(args.checkpoint_dir, args.seg_checkpoint))
        # load_checkpoint(self.gmm, os.path.join(args.checkpoint_dir, args.gmm_checkpoint))
        # load_checkpoint(alias, os.path.join(args.checkpoint_dir, args.alias_checkpoint))

        self.gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(self.device)
        self.up = nn.Upsample(size=(args.load_height, args.load_width), mode='bilinear')

        self.criterion_gan = GANLoss(use_lsgan=not args.no_lsgan)
        self.ce_loss = nn.CrossEntropyLoss()

        self.optimizer_seg = optim.Adam(list(self.segG.parameters()) + list(self.segD.parameters()),
                                        lr=0.0004, betas=(0.5,0.999))
        self.optimizer_seg.zero_grad(set_to_none=True)

        self.scaler = amp.GradScaler(enabled=args.use_amp)

        self.parse_labels = {
                # 0:  ['background',  [0]],
                # 1:  ['hair',        [1]],
                # 3:  ['upper',       [3]],
                # 5:  ['left_arm',    [5]],
                # 6:  ['right_arm',   [6]],
                4:  ['noise',       [12]],
                2:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],   # contains face and lower body.
            }
        if args.distributed:
            dist.init_process_group(backend="nccl")
            if args.sync_bn:
                self.segG = nn.SyncBatchNorm.convert_sync_batchnorm(self.segG)
                self.segD = nn.SyncBatchNorm.convert_sync_batchnorm(self.segD)
                # self.gmm = nn.SyncBatchNorm.convert_sync_batchnorm(self.gmm)
                # self.alias = nn.SyncBatchNorm.convert_sync_batchnorm(self.alias)
            self.segG = DDP(self.segG, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
            self.segD = DDP(self.segD, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
            # self.gmm = DDP(self.gmm, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)
            # self.alias = DDP(self.alias, device_ids=[args.local_rank], output_device=args.local_rank, broadcast_buffers=False)

        self.train_dataset = VITONDataset(args)
        self.train_loader = VITONDataLoader(args, self.train_dataset)

        if args.local_rank==0 and args.use_wandb:
            wandb.watch([self.segG, self.segD], log=None)
    
    def segmentation_train_step(self, args, parse_target_down, parse_agnostic, pose, cloth, cloth_mask, get_img_log=False):
        with amp.autocast(enabled=args.use_amp):
            # Part 1. Segmentation generation
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(cloth * cloth_mask, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cloth_mask, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size(), device=self.device)), dim=1)

            parse_pred_down = self.segG(seg_input)
            parse_target_mx = parse_target_down.argmax(dim=1)

            lambda_ce = 10
            seg_lossG = lambda_ce * self.ce_loss(parse_pred_down, parse_target_mx)

            parse_pred_down = F.softmax(parse_pred_down, dim=1)
            fake_out = self.segD(torch.cat((seg_input, parse_pred_down), dim=1))
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
    
        im_log = {}
        if get_img_log:
            parse_pred_mx = parse_pred_down.argmax(dim=1)
            im_log['seg_real'] = (parse_target_mx.detach_()*(255/args.semantic_nc)).cpu().numpy()
            im_log['seg_pred'] = (parse_pred_mx.detach_()*(255/args.semantic_nc)).cpu().numpy()
        return seg_lossG.detach_(), seg_lossD.detach_(), im_log


    def gmm_train_step(self, args, img, img_agnostic, parse, pose, cloth, cloth_mask, get_img_log=False):
        # Part 2. Clothes Deformation
        agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='bilinear')
        parse_cloth_gmm = F.interpolate(parse[:, 3:4], size=(256, 192), mode='nearest')
        pose_gmm = F.interpolate(pose, size=(256, 192), mode='bilinear')
        c_gmm = F.interpolate(cloth, size=(256, 192), mode='bilinear')
        gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

        _, warped_grid = self.gmm(gmm_input, c_gmm)
        warped_c = F.grid_sample(cloth, warped_grid, padding_mode='border')
        warped_cm = F.grid_sample(cloth_mask, warped_grid, padding_mode='border')
        return


    def alias_train_step(self, args, img, img_agnostic, parse, pose, warped_c, warped_cm):
        # Part 3. Try-on synthesis
        misalign_mask = parse[:, 3:4] - warped_cm
        misalign_mask[misalign_mask < 0.0] = 0.0
        parse_div = torch.cat((parse, misalign_mask), dim=1)
        parse_div[:, 3:4] -= misalign_mask

        output = self.alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)
        return
    
    def train_epoch(self, args, epoch):
        if args.distributed: self.train_loader.train_sampler.set_epoch(epoch)
        segG_losses = AverageMeter()
        segD_losses = AverageMeter()
        tsteps = len(self.train_loader.data_loader)
        with tqdm(self.train_loader.data_loader, desc=f"Epoch {epoch:>2}") as pbar:
            for step, batch in enumerate(pbar):
                batch = self.train_loader.device_augment(batch, self.device, self.memory_format)
                img = batch['img']
                img_agnostic = batch['img_agnostic']
                parse_target_down = batch['parse_target_down']
                parse_agnostic = batch['parse_agnostic']
                pose = batch['pose']
                cloth = batch['cloth']
                cloth_mask = batch['cloth_mask']

                seg_lossG, seg_lossD, seg_im_log = self.segmentation_train_step(args, parse_target_down,
                                                            parse_agnostic, pose, cloth, cloth_mask,
                                                            get_img_log=step==(tsteps-1))

                segG_losses.update(seg_lossG.detach_(), parse_target_down.size(0))
                segD_losses.update(seg_lossD.detach_(), parse_target_down.size(0))

                # convert 13 channel body parse to 7 channel parse.
                parse_target = self.gauss(self.up(parse_target_down))
                parse_target = parse_target.argmax(dim=1, keepdim=True)
                parse_orig = parse_target.copy()
                for k,v in self.parse_labels.items():
                    for l in v[1]:
                        if l!=k:
                            parse_target[parse_orig==l] = k

                parse = torch.zeros(parse_target.size(0), 7, args.load_height, args.load_width, dtype=torch.float32, device=self.device)
                parse.scatter_(1, parse_target, 1.0)

                gmm_lossG, gmm_lossD, gmm_im_log = self.gmm_train_step(self, args, img, img_agnostic, parse,
                                                            pose, cloth, cloth_mask,
                                                            get_img_log=step==(tsteps-1))


                if args.local_rank == 0:
                    if not step % args.log_interval:
                        info = {'SegG Loss': float(segG_losses.avg), 'SegD Loss': float(segD_losses.avg)}
                        if args.use_wandb: wandb.log(info)
                        pbar.set_postfix(info)
                self.scaler.update()
        return seg_im_log
    
    def train_loop(self, args, init_epoch=0):
        for epoch in range(init_epoch, args.epochs):
            seg_im_log = self.train_epoch(args, epoch)
            if args.local_rank == 0:
                if args.use_wandb:
                    wandb.log({'epoch': epoch})
                    im_dict = {}
                    for k in seg_im_log:
                        im_dict[k] = []
                        for img in seg_im_log[k]:
                            im_dict[k].append(wandb.Image(img))
                    wandb.log(im_dict)
                if not epoch%10:
                    self.save_models(args)
    
    def save_models(self, args):
        synchronize()
        if args.local_rank == 0:
            torch.save(self.segG.state_dict(), os.path.join(args.checkpoint_dir, "segG.pth"))
            torch.save(self.segD.state_dict(), os.path.join(args.checkpoint_dir, "segD.pth"))
            torch.save(self.optimizer_seg.state_dict(), os.path.join(args.checkpoint_dir, "optimizer_seg.pth"))
    
    def load_models(self, args):
        synchronize()
        # configure map_location properly
        map_location = {'cuda:0': f'cuda:{args.local_rank}'}
        self.segG.load_state_dict(os.path.join(args.checkpoint_dir, "segG.pth"), map_location=map_location)
        self.segD.load_state_dict(os.path.join(args.checkpoint_dir, "segD.pth"), map_location=map_location)
        self.optimizer_seg.load_state_dict(os.path.join(args.checkpoint_dir, "optimizer_seg.pth"), map_location=map_location)


def main():
    args = get_args()
    print(args)
    seed_everything(args.seed)
    try:
        tm = TrainModel(args)
        tm.train_loop(args)
    except KeyboardInterrupt:
        print("[!] Keyboard Interrupt! Cleaning up and shutting down.")
    finally:
        cleanup(args.distributed)


if __name__ == '__main__':
    main()
