import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class GetData_s1e(Dataset):
	def __init__(self, path, dft, transform):
		self.path = path
		self.transform = transform
		self.file = dft

	def __getitem__(self, k):
		pid = self.file.iloc[k,0]
		label = torch.tensor([self.file.iloc[k, p] for p in range(3, self.file.shape[1])], dtype=torch.long)
		fts = []
		for pt in range(1,3):
			img_name = self.file.iloc[k, pt]
			img_item_path = os.path.join(self.path, img_name)
			img = Image.open(img_item_path)
			img = self.transform(img)
			fts.append(img)

		return pid, fts, label

	def __len__(self):
		return self.file.shape[0]

class GetData_fts(Dataset):
	def __init__(self, dft, meta_nums):
		self.meta_nums = meta_nums
		self.file = dft

	def __getitem__(self, k):
		pid = self.file.iloc[k,0]
		label = torch.tensor([self.file.iloc[k, p] for p in range(2 * self.meta_nums + 1, self.file.shape[1])],
							 dtype=torch.long)
		fts = []
		for pt in range(2):
			fts.append(torch.tensor(self.file.iloc[k, pt * self.meta_nums + 1:(pt + 1) * self.meta_nums + 1].to_numpy().astype("float"),
									dtype=torch.float))

		return pid, fts, label

	def __len__(self):
		return self.file.shape[0]

class GetData_s4(Dataset):
	def __init__(self, path, dft, transform):
		self.path = path
		self.transform = transform
		self.file = dft

	def __getitem__(self, k):
		pid = self.file.iloc[k,0]
		label_0 = torch.tensor([self.file.iloc[k, p] for p in range(3, self.file.shape[1]-4)],
							   dtype=torch.long)
		label_s1 = torch.tensor(self.file.iloc[k, self.file.shape[1]-4:self.file.shape[1]-2].to_numpy().astype("float"),
								dtype=torch.float)
		label_s2 = torch.tensor(self.file.iloc[k, self.file.shape[1] - 2:self.file.shape[1]].to_numpy().astype("float"),
								dtype=torch.float)
		label = [label_0, label_s1, label_s2]
		fts = []
		for pt in range(1,3):
			img_name = self.file.iloc[k, pt]
			img_item_path = os.path.join(self.path, img_name)
			img = Image.open(img_item_path)
			img = self.transform(img)
			fts.append(img)

		return pid, fts, label

	def __len__(self):
		return self.file.shape[0]

def build_dataset(is_train, dft, stage, meta_nums=None, data_path=None):
	transform = build_transform(is_train)
	if stage == 1 or stage == "e":
		dataset = GetData_s1e(data_path, dft, transform=transform)
	elif stage == 4:
		dataset = GetData_s4(data_path, dft, transform=transform)
	else:
		dataset = GetData_fts(dft, meta_nums)

	return dataset

def build_transform(is_train):
	if is_train=='train':
		transform = transforms.Compose([
			transforms.RandomRotation(degrees=180, fill=128),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.5,
																			   contrast=0.5,
																			   saturation=0.5,
																			   hue=0.5)]), p=0.5),
			transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(5, (5,9))]), p=0.3),
			transforms.ToTensor()
		])
		return transform
	else:
		transform = transforms.ToTensor()
		return transform
	
