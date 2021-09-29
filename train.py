import os
import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda import amp
from torch import autograd
import torchgeometry as tgm

from train_datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator, MultiscaleDiscriminator, GANLoss
from utils import gen_noise, set_grads, load_checkpoint
from train_options import get_opt


class TrainModel:
    def __init__(self, opt):
        self.segG = self.SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc).cuda().train()
        self.segD = MultiscaleDiscriminator(opt, opt.semantic_nc + opt.semantic_nc + 8, use_sigmoid=opt.no_lsgan).cuda().train()
        # self.gmm = GMM(opt, inputA_nc=7, inputB_nc=3).cuda().train()
        # opt.semantic_nc = 7
        # alias = ALIASGenerator(opt, input_nc=9).cuda().train()
        # opt.semantic_nc = 13

        # load_checkpoint(self.segG, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
        # load_checkpoint(self.gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
        # load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

        # self.gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).cuda()
        # self.up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')

        self.scaler = amp.GradScaler(enabled=opt.use_amp)

        train_dataset = VITONDataset(opt)
        train_loader = VITONDataLoader(opt, train_dataset)

        self.criterion_gan = GANLoss(use_lsgan=not opt.no_lsgan)
        self.ce_loss = nn.CrossEntropyLoss()

        self.optimizer_seg = torch.optim.Adam(list(self.segG.parameters()) + list(self.segD.parameters()),
                                        lr=0.0004, betas=(0.5,0.999))

        self.parse_labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],   # contains face and lower body.
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }

    def segmentation_train_step(self, opt, inputs):
        parse_target_down = inputs['parse_target_down'].cuda(non_blocking=True)
        parse_agnostic = inputs['parse_agnostic'].cuda(non_blocking=True)
        pose = inputs['pose'].cuda(non_blocking=True)
        c = inputs['cloth'].cuda(non_blocking=True)
        cm = inputs['cloth_mask'].cuda(non_blocking=True)

        with amp.autocast(enabled=opt.use_amp):
            # Part 1. Segmentation generation
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size(), device='cuda')), dim=1)

            parse_pred_down = self.segG(seg_input)

            lambda_ce = 10
            seg_loss = lambda_ce * self.ce_loss(parse_pred_down, parse_target_down.argmax(dim=1))
            
            fake_out = self.segD(torch.cat((seg_input, parse_pred_down.detach()), dim=1))
            real_out = self.segD(torch.cat((seg_input, parse_target_down.detach()), dim=1))
            seg_loss += self.criterion_gan(fake_out, True)                          # Treat fake images as real to train the Generator.
            seg_lossD = (self.criterion_gan(real_out, True)     # Treat real as real
                        + self.criterion_gan(fake_out, False))   # and fake as fake to train Discriminator.

        scaled_gradsG = autograd.grad(self.scaler.scale(seg_loss), self.segG.parameters(), retain_graph=True)
        scaled_gradsD = autograd.grad(self.scaler.scale(seg_lossD), self.segD.parameters())

        set_grads(scaled_gradsG, self.segG.parameters())
        set_grads(scaled_gradsD, self.segD.parameters())

        self.scaler.step(self.optimizer_seg)
        self.optimizer_seg.zero_grad(set_to_none=True)
        self.scaler.update()


    def gmm_train_step(self, opt, inputs):
        img = inputs['img'].cuda(non_blocking=True)
        img_agnostic = inputs['img_agnostic'].cuda(non_blocking=True)

        # convert 13 channel body parse to 7 channel parse.
        # parse_pred = gauss(up(parse_pred_down))
        # parse_pred = parse_pred.argmax(dim=1, keepdim=True)
        # parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float32, device='cuda')
        # parse_old.scatter_(1, parse_pred, 1.0)

        # VERIFY MEMORY FORMAT AND CHANNEL LAST HERE

        # parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float32, device='cuda')
        # for j in range(len(self.parse_labels)):
        #     for lbl in self.parse_labels[j][1]:
        #         parse[:, j] += parse_old[:, lbl]

        # # Part 2. Clothes Deformation
        # agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
        # parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
        # pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
        # c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
        # gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

        # _, warped_grid = gmm(gmm_input, c_gmm)
        # warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
        # warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')


    def alias_train_step(self, opt, inputs):
        img = inputs['img'].cuda(non_blocking=True)
        img_agnostic = inputs['img_agnostic'].cuda(non_blocking=True)
        # Part 3. Try-on synthesis
        # misalign_mask = parse[:, 2:3] - warped_cm
        # misalign_mask[misalign_mask < 0.0] = 0.0
        # parse_div = torch.cat((parse, misalign_mask), dim=1)
        # parse_div[:, 2:3] -= misalign_mask

        # output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)


def main():
    opt = get_opt()
    print(opt)
    TrainModel(opt)

if __name__ == '__main__':
    main()
