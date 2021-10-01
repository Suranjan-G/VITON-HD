import json
import os
from os import path as osp
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch import nn
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
        self.memory_format = torch.channels_last if args.memory_format == "channels_last" else torch.contiguous_format
        self.data_path = osp.join(args.dataset_dir, args.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        self.img_names = os.listdir(osp.join(self.data_path, 'image'))

    def get_parse_agnostic(self, parse, pose_data):
        parse_array = np.asarray(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
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
            parse_arm = (np.asarray(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def get_img_agnostic(self, img, parse, pose_data):
        parse_array = np.asarray(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

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
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    def __getitem__(self, index):
        img_name = self.img_names[index]
        c = Image.open(osp.join(self.data_path, 'cloth', img_name)).convert('RGB')
        c = transforms.Resize((self.load_height, self.load_width), interpolation=InterpolationMode.BILINEAR)(c)
        c = self.transform(c)  # [-1,1]

        cm = Image.open(osp.join(self.data_path, 'cloth-mask', img_name))
        cm = transforms.Resize((self.load_height, self.load_width), interpolation=InterpolationMode.NEAREST)(cm)
        cm = np.asarray(cm)
        cm = (cm >= 128).astype(np.float32)
        cm = torch.from_numpy(cm)  # [0,1]
        cm.unsqueeze_(0)

        # load pose image
        pose_name = img_name.replace('.jpg', '_rendered.png')
        pose_rgb = Image.open(osp.join(self.data_path, 'openpose-img', pose_name))
        pose_rgb = transforms.Resize((self.load_height, self.load_width), interpolation=InterpolationMode.BILINEAR)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        pose_name = img_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose-json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.asarray(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

        # load parsing image
        parse_name = img_name.replace('.jpg', '.png')
        parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        parse_down = transforms.Resize((256, 192), interpolation=InterpolationMode.NEAREST)(parse)
        parse_down = torch.from_numpy(np.array(parse_down)[None]).long()
        parse_down_map = torch.zeros(20, 256, 192, dtype=torch.float)
        parse_down_map.scatter_(0, parse_down, 1.0)

        parse = transforms.Resize((self.load_height, self.load_width), interpolation=InterpolationMode.NEAREST)(parse)
        parse_agnostic = self.get_parse_agnostic(parse, pose_data)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()
        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)

        labels = {
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
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
            12: ['noise', [3, 11]]
        }
        new_parse_agnostic_map = torch.empty(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float).fill_(0.0)
        new_parse_down_map = torch.empty(self.semantic_nc, 256, 192, dtype=torch.float).fill_(0.0)
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
                new_parse_down_map[i] += parse_down_map[label]

        # load person image
        # img = Image.open(osp.join(self.data_path, 'image', img_name))
        # img = transforms.Resize((self.load_height, self.load_width), interpolation=InterpolationMode.BILINEAR)(img)
        # img_agnostic = self.get_img_agnostic(img, parse, pose_data)
        # img = self.transform(img)
        # img_agnostic = self.transform(img_agnostic)  # [-1,1]

        result = {
            # 'img': img,
            # 'img_agnostic': img_agnostic,
            'parse_target_down': new_parse_down_map,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_rgb,
            'cloth': c,
            'cloth_mask': cm,
        }
        return result
    
    def __len__(self):
        return len(self.img_names)
    
    def collate_fn(self, data_batch, angle_keys=['cloth', 'cloth_mask']):
        angle1, angle2 = torch.randint(-15, 15, (2,))
        flip = torch.randn([]) > 0.5
        result = {}
        for key in data_batch[0].keys():
            result[key] = torch.stack([inpd[key] for inpd in data_batch])
            if flip: result[key] = TF.hflip(result[key])
            angle = angle1 if key in angle_keys else angle2
            result[key] = TF.rotate(result[key], angle)
            result[key] = result[key].to(memory_format=self.memory_format)
        return result


class VITONDataLoader:
    def __init__(self, args, dataset):
        super().__init__()

        self.train_sampler = None
        if args.num_gpus > 1:
            self.train_sampler = data.distributed.DistributedSampler(dataset, shuffle=args.shuffle)
        elif args.shuffle:
            self.train_sampler = data.sampler.RandomSampler(dataset)

        self.data_loader = data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=(self.train_sampler is None),
                num_workers=args.workers, pin_memory=True, drop_last=True, sampler=self.train_sampler,
                collate_fn=dataset.collate_fn
        )
        self.dataset = dataset
