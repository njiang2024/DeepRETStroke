# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# define the pretrained dataset
class GetData(Dataset):
    def __init__(self, root_dir, pre_train=False, progression=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pre_train = pre_train
        self.progression = progression

        file_path = os.path.join(self.root_dir, "ground_truth.csv")
        self.file = pd.read_csv(file_path)
        self.path = os.path.join(self.root_dir, "fundus")
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, k):
        img_name = self.file.iloc[k,0]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        if self.pre_train:
            label = torch.tensor([self.file.iloc[k,p] for p in range(3)], dtype=torch.long)
        elif self.progression:
            label = torch.tensor([self.file.iloc[k,p] for p in range(5)], dtype=torch.long)
        else:
            label = self.file.iloc[k,1]

        if self.transform:
            img = self.transform(img)

        return img, img_name, label

    def __len__(self):
        return len(self.img_path_list)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    if args.pre_train:
        dataset = GetData(root, pre_train=True, transform=transform)
    elif args.progression:
        dataset = GetData(root, progression=True, transform=transform)
    else:
        #dataset = GetData(root, transform=transform)
        dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    t.append(transforms.RandomRotation(30)) # add image rotation
    return transforms.Compose(t)
