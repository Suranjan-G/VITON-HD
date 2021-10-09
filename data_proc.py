from glob import glob
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import os

ims = glob("../data/cloth_swap_final/top/cloth/*")[::-2]

sz = (768, 1024)
for i in tqdm(ims):
    tgt = i.replace('cloth_swap_final/top', 'docker_data/train').replace('.webp','.png')
    if not os.path.exists(tgt):
        img = Image.open(i)
        img.thumbnail(sz, Image.LANCZOS, reducing_gap=3.0)
        w,h = img.size
        img = np.array(img)
        dffw, dffh = sz[0] - w, sz[1] - h
        mean = img[:10,-10:].mean(axis=(0,1))
        img = cv2.copyMakeBorder(img, dffh//2, dffh - dffh//2, dffw//2, dffw - dffw//2, borderType=cv2.BORDER_CONSTANT, value=mean)
        img = Image.fromarray(img)
        if img.size != sz:
            print(i)
        img.save(tgt, quality=100, subsampling=0)
