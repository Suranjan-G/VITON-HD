import json
import os
from os import path as osp
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import argparse

np.seterr(divide='ignore', invalid='ignore')

class AgnosticGen:
    def __init__(self, args):
        super().__init__()
        self.load_height = args.load_height
        self.load_width = args.load_width
        self.data_path = osp.join(args.dataset_dir, args.dataset_mode)

        parse_down_dir = osp.join(self.data_path, 'parse-down')
        if not osp.exists(parse_down_dir):
            os.makedirs(parse_down_dir)
        
        parse_agn_dir = osp.join(self.data_path, 'parse-agnostic')
        if not osp.exists(parse_agn_dir):
            os.makedirs(parse_agn_dir)
        
        image_agn_dir = osp.join(self.data_path, 'image-agnostic')
        if not osp.exists(image_agn_dir):
            os.makedirs(image_agn_dir)

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


    def get_parse_agnostic(self, parse_array, pose_data):
        parse_upper = (parse_array == 7).astype(np.uint8) * 255
        parse_neck = (parse_array == 13).astype(np.uint8) * 255

        r = 10
        agnostic = Image.fromarray(parse_array.copy())

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

    def get_img_agnostic(self, img, parse_array, pose_data):
        parse_head = (parse_array == 4).astype(np.uint8) * 255
        parse_lower = ((parse_array == 9).astype(np.uint8) +
                       (parse_array == 11).astype(np.uint8) +
                       (parse_array == 12).astype(np.uint8) +
                       (parse_array == 2).astype(np.uint8) +
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

        ext = img_name.split('.')[-1]
        pose_name = img_name.replace(f'.{ext}', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose-json', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

        # load parsing image
        parse_name = img_name.replace(f'.{ext}', '.png')
        parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        parse = np.array(parse, dtype=np.uint8)
        parse_orig = parse.copy()
        for k,v in self.labels.items():
            for l in v[1]:
                if l!=k:
                    parse[parse_orig==l] = k
        del parse_orig

        parse_down_file = osp.join(self.data_path, 'parse-down', parse_name)
        if not osp.exists(parse_down_file):
            parse_down = cv2.resize(parse, (192, 256), interpolation=cv2.INTER_NEAREST)
            parse_down[parse_down==13] = 0
            cv2.imwrite(parse_down_file, parse_down)

        parse = cv2.resize(parse, (self.load_width, self.load_height), interpolation=cv2.INTER_NEAREST)
        parse_agnostic_file = osp.join(self.data_path, 'parse-agnostic', parse_name)
        if not osp.exists(parse_agnostic_file):
            parse_agnostic = self.get_parse_agnostic(parse, pose_data)
            parse_agnostic = np.array(parse_agnostic, dtype=np.uint8)
            parse_agnostic[parse_agnostic==13] = 0
            cv2.imwrite(parse_agnostic_file, parse_agnostic)

        # load person image
        img_agnostic_file = osp.join(self.data_path, 'image-agnostic', img_name)
        if not osp.exists(img_agnostic_file):
            img = Image.open(osp.join(self.data_path, 'image', img_name))
            img = img.resize((self.load_width, self.load_height), Image.BILINEAR)
            img_agnostic = self.get_img_agnostic(img, parse, pose_data)
            img_agnostic.save(img_agnostic_file)

    def __len__(self):
        return len(self.img_names)
    
    def generate(self):
        for i in tqdm(self):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_height', type=int, default=1024, help="Height of input image.")
    parser.add_argument('--load_width', type=int, default=768, help="Width of input image.")
    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--dataset_mode', type=str, default='train', help="train or test.")
    args = parser.parse_args()

    agn = AgnosticGen(args)
    agn.generate()