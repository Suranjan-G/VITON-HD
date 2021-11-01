import os
from os import path as osp
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

np.seterr(divide='ignore', invalid='ignore')

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
                0: ['background', [0]],
                1: ['hair', [1, 2]],
                2: ['left_shoe', [18]],
                3: ['noise', [3, 11]],
                4: ['face', [4, 13]],
                5: ['left_arm', [14]],
                6: ['right_arm', [15]],
                7: ['upper', [5, 6, 7]],
                8: ['socks', [8]],
                9: ['bottom', [9, 12]],
                10: ['right_shoe', [19]],
                11: ['left_leg', [16]],
                12: ['right_leg', [17]],
                13: ['neck', [10]],
            }

    def __getitem__(self, index):
        img_name = self.img_names[index]
        cloth = Image.open(osp.join(self.data_path, 'cloth', img_name)).convert('RGB')
        cloth = TF.resize(cloth, (self.load_height, self.load_width), interpolation=InterpolationMode.BILINEAR)
        cloth = self.transform(cloth)  # [-1,1]

        ext = img_name.split('.')[-1]
        cmask_name = img_name.replace(f'.{ext}', '.png')
        cloth_mask = Image.open(osp.join(self.data_path, 'cloth-mask', cmask_name)).convert('L')
        cloth_mask = TF.resize(cloth_mask, (self.load_height, self.load_width), interpolation=InterpolationMode.NEAREST)
        cloth_mask = np.array(cloth_mask, dtype=np.float32)/255.
        cloth_mask = torch.from_numpy(cloth_mask)  # [0,1]
        cloth_mask.unsqueeze_(0)

        # load pose image
        pose_name = img_name.replace(f'.{ext}', '_rendered.png')
        pose_rgb = Image.open(osp.join(self.data_path, 'openpose-img', pose_name))
        pose_rgb = TF.resize(pose_rgb, (self.load_height, self.load_width), interpolation=InterpolationMode.BILINEAR)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        # load parsing image
        parse_name = img_name.replace(f'.{ext}', '.png')
        parse_down = Image.open(osp.join(self.data_path, 'parse-down', parse_name))

        parse_agnostic = Image.open(osp.join(self.data_path, 'parse-agnostic', parse_name))

        # load person image
        img = Image.open(osp.join(self.data_path, 'image', img_name))
        img = transforms.Resize((self.load_height, self.load_width), interpolation=InterpolationMode.BILINEAR)(img)
        img_agnostic = Image.open(osp.join(self.data_path, 'image-agnostic', img_name))
        img = self.transform(img)
        img_agnostic = self.transform(img_agnostic)  # [-1,1]

        result = {
            'img': img,
            'img_agnostic': img_agnostic,
            'parse_target_down': parse_down,
            'parse_agnostic': parse_agnostic,
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
                num_workers=args.workers, pin_memory=True, drop_last=False, sampler=self.train_sampler,
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