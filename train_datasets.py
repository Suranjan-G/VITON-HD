import json
import os
from os import path as osp
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


class VITONDataset(data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.load_height = args.load_height
        self.load_width = args.load_width
        self.semantic_nc = args.semantic_nc
        self.data_path = osp.join(args.dataset_dir, args.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        self.img_names = os.listdir(osp.join(self.data_path, 'image'))

        self.labels = {
                # 0: ['background', [0, 10]],
                # 1: ['hair', [1, 2]],
                0: ['background', [0]],
                1: ['hair', [2]],
                2: ['face', [4, 13]],
                3: ['upper', [5, 6, 7]],
                4: ['bottom', [9, 12]],
                5: ['left_arm', [14]],
                6: ['right_arm', [15]],
                7: ['left_leg', [16]],
                8: ['right_leg', [17]],
                9: ['left_shoe', [18]],
                10: ['right_shoe', [19]],
                11: ['socks', [8]],
                12: ['noise', [3, 11]],
                13: ['neck', [10]],
            }

    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.array(parse)
        parse_upper = (parse_array == 3).astype(np.uint8) * 255
        parse_neck = (parse_array == 13).astype(np.uint8) * 255

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(5, [2, 5, 6, 7]), (6, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.load_width, self.load_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = np.array(mask_arm) * (parse_array == parse_id)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(parse_upper, 'L'))
        agnostic.paste(0, None, Image.fromarray(parse_neck, 'L'))

        return agnostic

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.array(parse)
        parse_head = (parse_array == 2).astype(np.uint8) * 255
        parse_lower = ((parse_array == 4).astype(np.uint8) +
                       (parse_array == 7).astype(np.uint8) +
                       (parse_array == 8).astype(np.uint8) +
                       (parse_array == 9).astype(np.uint8) +
                       (parse_array == 10).astype(np.uint8)) * 255

        r = 20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(parse_head, 'L'))
        agnostic.paste(img, None, Image.fromarray(parse_lower, 'L'))

        return agnostic

    def __getitem__(self, index):
        img_name = self.img_names[index]
        cloth = Image.open(osp.join(self.data_path, 'cloth', img_name)).convert('RGB')
        cloth = TF.resize(cloth, (self.load_height, self.load_width), interpolation=InterpolationMode.BILINEAR)
        cloth = self.transform(cloth)  # [-1,1]

        cloth_mask = Image.open(osp.join(self.data_path, 'cloth-mask', img_name)).convert('L')
        cloth_mask = TF.resize(cloth_mask, (self.load_height, self.load_width), interpolation=InterpolationMode.NEAREST)
        cloth_mask = np.array(cloth_mask)
        cloth_mask = (cloth_mask >= 128).astype(np.float32)
        cloth_mask = torch.from_numpy(cloth_mask)  # [0,1]
        cloth_mask.unsqueeze_(0)

        # load pose image
        ext = img_name.split('.')[-1]
        pose_name = img_name.replace(f'.{ext}', '_rendered.png')
        pose_rgb = Image.open(osp.join(self.data_path, 'openpose-img', pose_name))
        pose_rgb = TF.resize(pose_rgb, (self.load_height, self.load_width), interpolation=InterpolationMode.BILINEAR)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        pose_name = img_name.replace('.{ext}', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose-json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

        # load parsing image
        parse_name = img_name.replace('.{ext}', '.png')
        parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        parse = np.array(parse)
        parse_orig = parse.copy()
        for k,v in self.labels.items():
            for l in v[1]:
                parse[parse_orig==l] = k
        del parse_orig
        parse = Image.fromarray(parse)
        parse_down = TF.resize(parse, (256, 192), interpolation=InterpolationMode.NEAREST)
        parse_down = torch.from_numpy(np.array(parse_down)[None]).long()
        parse_down[parse_down==13] = 0
        parse_down_map = torch.zeros(self.semantic_nc, 256, 192, dtype=torch.float)
        parse_down_map.scatter_(0, parse_down, 1.0)

        parse = TF.resize(parse, (self.load_height, self.load_width), interpolation=InterpolationMode.NEAREST)
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()
        parse_agnostic[parse_agnostic==13] = 0
        parse_agnostic_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)

        # load person image
        # img = Image.open(osp.join(self.data_path, 'image', img_name))
        # img = transforms.Resize((self.load_height, self.load_width), interpolation=InterpolationMode.BILINEAR)(img)
        # img_agnostic = self.get_img_agnostic(img, parse, pose_data)
        # img = self.transform(img)
        # img_agnostic = self.transform(img_agnostic)  # [-1,1]

        result = {
            # 'img': img,
            # 'img_agnostic': img_agnostic,
            'parse_target_down': parse_down_map,
            'parse_agnostic': parse_agnostic_map,
            'pose': pose_rgb,
            'cloth': cloth,
            'cloth_mask': cloth_mask,
        }
        return result
    
    def __len__(self):
        return len(self.img_names)
    
class VITONDataLoader:
    def __init__(self, args, dataset):
        super().__init__()

        self.train_sampler = None
        if args.distributed:
            self.train_sampler = data.distributed.DistributedSampler(dataset, shuffle=args.shuffle)
        elif args.shuffle:
            self.train_sampler = data.sampler.RandomSampler(dataset)

        self.data_loader = data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=(self.train_sampler is None),
                num_workers=args.workers, pin_memory=True, drop_last=True, sampler=self.train_sampler,
                collate_fn=lambda x: x
        )
        self.dataset = dataset

    def device_augment(self, data_batch, device, memory_format=torch.contiguous_format, angle1_keys=['cloth', 'cloth_mask']):
        batch = {}
        flip = torch.randn([]) > 0.5
        rot = torch.randn([]) > 0.5
        angle1, angle2 = torch.randint(-15, 15, (2,))
        angle = 0
        for key in data_batch[0].keys():
            batch[key] = torch.stack([inpd[key].to(device, non_blocking=True)
                                     for inpd in data_batch]).to(memory_format=memory_format)
            if flip: batch[key] = TF.hflip(batch[key])
            if rot: angle = angle1 if key in angle1_keys else angle2
            batch[key] = TF.rotate(batch[key], float(angle))
        return batch