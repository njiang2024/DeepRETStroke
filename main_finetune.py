import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import timm

assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit
import pickle

from engine_finetune import train_one_epoch, evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args_parser():
	parser = argparse.ArgumentParser('DeepRETStroke System Development', add_help=False)

	parser.add_argument('--batch_size', default=64, type=int,
						help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
	parser.add_argument('--epochs', default=30, type=int)
	parser.add_argument('--iterations', default=20, type=int)
	parser.add_argument('--accum_iter', default=1, type=int,
						help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

	# Model parameters
	parser.add_argument('--model', default='vit_large_patch16_s', type=str, metavar='MODEL',
						help='Name of model to train')
	parser.add_argument('--input_size', default=256, type=int,
						help='images input size')
	parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
						help='Drop path rate (default: 0.1)')
	parser.add_argument('--combined', action='store_true', default=False,
						help="Model selection")

	# Optimizer parameters
	parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
						help='Clip gradient norm (default: None, no clipping)')
	parser.add_argument('--weight_decay', type=float, default=0.05,
						help='weight decay (default: 0.05)')
	parser.add_argument('--lr', type=float, default=None, metavar='LR',
						help='learning rate (absolute lr)')
	parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
						help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
	parser.add_argument('--layer_decay', type=float, default=0.75,
						help='layer-wise lr decay from ELECTRA/BEiT')
	parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
						help='lower lr bound for cyclic schedulers that hit 0')
	parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
						help='epochs to warmup LR')

	# Augmentation parameters
	parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
						help='Color jitter factor (enabled only when not using Auto/RandAug)')
	parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
						help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
	parser.add_argument('--smoothing', type=float, default=0.1,
						help='Label smoothing (default: 0.1)')

	# * Random Erase params
	parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
						help='Random erase prob (default: 0.25)')
	parser.add_argument('--remode', type=str, default='pixel',
						help='Random erase mode (default: "pixel")')
	parser.add_argument('--recount', type=int, default=1,
						help='Random erase count (default: 1)')
	parser.add_argument('--resplit', action='store_true', default=False,
						help='Do not random erase first (clean) augmentation split')

	# * Mixup params
	parser.add_argument('--mixup', type=float, default=0,
						help='mixup alpha, mixup enabled if > 0.')
	parser.add_argument('--cutmix', type=float, default=0,
						help='cutmix alpha, cutmix enabled if > 0.')
	parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
						help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
	parser.add_argument('--mixup_prob', type=float, default=1.0,
						help='Probability of performing mixup or cutmix when either/both is enabled')
	parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
						help='Probability of switching to cutmix when both mixup and cutmix enabled')
	parser.add_argument('--mixup_mode', type=str, default='batch',
						help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

	# * development params
	parser.add_argument('--finetune', default='', type=str,
						help='finetune from checkpoint')
	parser.add_argument('--task', default='',type=str,
						help='finetune from checkpoint')
	parser.add_argument('--global_pool', action='store_true', default=True)
	parser.add_argument('--cls_token', action='store_false', dest='global_pool',
						help='Use class token instead of global pool for classification')

	# Dataset parameters
	parser.add_argument('--data_path', default='./data', type=str,
						help='dataset path')
	parser.add_argument('--nb_classes', default=1000, type=int,
						help='number of the classification types')
	parser.add_argument('--output_dir', default='./output_dir',
						help='path where to save, empty for no saving')
	parser.add_argument('--log_dir', default='./output_dir',
						help='path where to tensorboard log')
	parser.add_argument('--device', default='cuda',
						help='device to use for training / testing')
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--resume', default='',
						help='resume from checkpoint')
	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
						help='start epoch')
	parser.add_argument('--eval', action='store_true',
						help='Perform evaluation only')
	parser.add_argument('--dist_eval', action='store_true', default=False,
						help='Enabling distributed evaluation (recommended during training for faster monitor')
	parser.add_argument('--num_workers', default=10, type=int)
	parser.add_argument('--pin_mem', action='store_true',
						help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
	parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
	parser.set_defaults(pin_mem=True)

	# distributed training parameters
	parser.add_argument('--world_size', default=1, type=int,
						help='number of distributed processes')
	parser.add_argument('--local_rank', default=-1, type=int)
	parser.add_argument('--dist_on_itp', action='store_true')
	parser.add_argument('--dist_url', default='env://',
						help='url used to set up distributed training')

	return parser


def main(args):
	misc.init_distributed_mode(args)

	print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
	print("{}".format(args).replace(', ', ',\n'))

	device = torch.device(args.device)
	num_tasks = misc.get_world_size()
	global_rank = misc.get_rank()

	# fix the seed for reproducibility
	seed = args.seed + misc.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)

	cudnn.benchmark = True

	if global_rank == 0 and args.log_dir is not None and not args.eval:
		os.makedirs(args.log_dir, exist_ok=True)
		log_writer = SummaryWriter(log_dir=args.log_dir + args.task)
	else:
		log_writer = None

	mixup_fn = None
	mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
	if mixup_active:
		print("Mixup is activated!")
		mixup_fn = Mixup(
			mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
			prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
			label_smoothing=args.smoothing, num_classes=args.nb_classes)

	model = models_vit.__dict__[args.model](
		drop_path_rate=args.drop_path,
		global_pool=args.global_pool,
	)

	# two classifiers for SBI Detector
	det1 = LogisticRegression(class_weight="balanced", max_iter=2000)
	det2 = LogisticRegression(class_weight="balanced", max_iter=2000)

	if not args.eval:
		checkpoint = torch.load(args.finetune, map_location='cpu')
		print("Load checkpoint from: %s" % args.finetune)
		checkpoint_model = checkpoint['model']

		# interpolate position embedding
		interpolate_pos_embed(model, checkpoint_model)

		# load pre-trained model
		msg = model.load_state_dict(checkpoint_model, strict=False)
		print(msg)

		# manually initialize other layers
		trunc_normal_(model.hidden_layer_0.weight, std=2e-5)
		trunc_normal_(model.hidden_layer_1.weight, std=2e-5)
		trunc_normal_(model.head_2_0.weight, std=2e-5)
		trunc_normal_(model.head_2_1.weight, std=2e-5)
		trunc_normal_(model.hidden_layer_t.weight, std=2e-5)
		trunc_normal_(model.head_2_t.weight, std=2e-5)
		trunc_normal_(model.hidden_layer.weight, std=2e-5)
		trunc_normal_(model.head_5.weight, std=2e-5)

	model.to(device)
	model_without_ddp = model
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

	print("Model = %s" % str(model_without_ddp))
	print('number of params (M): %.2f' % (n_parameters / 1.e6))

	eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

	if args.lr is None:  # only base_lr is specified
		args.lr = args.blr * eff_batch_size / 256

	print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
	print("actual lr: %.2e" % args.lr)

	print("accumulate grad iterations: %d" % args.accum_iter)
	print("effective batch size: %d" % eff_batch_size)

	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
		model_without_ddp = model.module

	# build optimizer with layer-wise lr decay (lrd)
	param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
										no_weight_decay_list=model_without_ddp.no_weight_decay(),
										layer_decay=args.layer_decay
										)
	optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
	loss_scaler = NativeScaler()

#############################################################################################

	# csv files for constructing the training and validation set
	dft_sb_tr = pd.read_csv("data/df_sb_tr.csv")
	dft_st_tr = pd.read_csv("data/df_st_tr.csv")
	dft_sb_va = pd.read_csv("data/df_sb_va.csv")
	dft_st_va = pd.read_csv("data/df_st_va.csv")

	# define the weight of pos/neg sample and the standard of high confidence sample
	weight = [1, 4]
	pos_thr = 0.75
	neg_thr = 0.75

	path = args.data_path

	# construct the datasets for different stages of training
	dataset_st_tr_s1 = build_dataset('train', dft_st_tr, 1, data_path=path)
	dataset_st_va_s1 = build_dataset('test', dft_st_va, 1, data_path=path)
	dataset_st_tr_e = build_dataset('test', dft_st_tr, "e", data_path=path)
	dataset_sb_tr_e = build_dataset('test', dft_sb_tr, "e", data_path=path)
	dataset_sb_va_e = build_dataset('test', dft_sb_va, "e", data_path=path)

	sampler_st_tr_s1 = torch.utils.data.DistributedSampler(dataset_st_tr_s1, num_replicas=num_tasks,
														rank=global_rank, shuffle=True)
	sampler_st_va_s1 = torch.utils.data.SequentialSampler(dataset_st_va_s1)
	sampler_st_tr_e = torch.utils.data.SequentialSampler(dataset_st_tr_e)
	sampler_sb_tr_e = torch.utils.data.SequentialSampler(dataset_sb_tr_e)
	sampler_sb_va_e = torch.utils.data.SequentialSampler(dataset_sb_va_e)
	sample_size = int(args.batch_size / 2)

	data_loader_st_tr_s1 = torch.utils.data.DataLoader(
		dataset_st_tr_s1, sampler=sampler_st_tr_s1,
		batch_size=sample_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=True,
	)
	data_loader_st_va_s1 = torch.utils.data.DataLoader(
		dataset_st_va_s1, sampler=sampler_st_va_s1,
		batch_size=sample_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=False
	)
	data_loader_st_tr_e = torch.utils.data.DataLoader(
		dataset_st_tr_e, sampler=sampler_st_tr_e,
		batch_size=sample_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=False
	)
	data_loader_sb_tr_e = torch.utils.data.DataLoader(
		dataset_sb_tr_e, sampler=sampler_sb_tr_e,
		batch_size=sample_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=False
	)
	data_loader_sb_va_e = torch.utils.data.DataLoader(
		dataset_sb_va_e, sampler=sampler_sb_va_e,
		batch_size=sample_size,
		num_workers=args.num_workers,
		pin_memory=args.pin_mem,
		drop_last=False
	)

	# step 1, model initialization
	stage = 1

	print(f"Start training stage 1")
	start_time = time.time()
	max_auc_s1 = 0.0

	for epoch in range(30):
		if args.distributed:
			data_loader_st_tr_s1.sampler.set_epoch(epoch)
		train_stats = train_one_epoch(
			model, data_loader_st_tr_s1,
			optimizer, device, epoch, loss_scaler,
			stage, None, weight,
			args.clip_grad, mixup_fn,
			log_writer=log_writer,
			args=args
		)

		val_auc_roc = evaluate(data_loader_st_va_s1, model, device, args.task, stage, mode="eva")
		print("test performance ", val_auc_roc)

		if max_auc_s1 < val_auc_roc:
			max_auc_s1 = val_auc_roc

			if args.output_dir:
				misc.save_model(
					args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
					loss_scaler=loss_scaler, epoch=epoch, best_weight=True, stage=stage)

		if log_writer is not None:
			log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)

		log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
					 'epoch': epoch,
					 'n_parameters': n_parameters}

		if args.output_dir and misc.is_main_process():
			if log_writer is not None:
				log_writer.flush()
			with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
				f.write(json.dumps(log_stats) + "\n")

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print('Stage 1 Training time {}'.format(total_time_str))

##########################   model initialization end   ############################

	repeat_performance_sb = [[],[]]
	repeat_performance_st = []

	for repeat in range(30):
		# stage e, generate the features from the frozed encoder
		stage = "e"

		if repeat == 0:
			checkpoint = torch.load(args.task+'checkpoint-1-best.pth', map_location='cpu')
		else:
			checkpoint = torch.load(args.task + 'checkpoint-4-r%s-best.pth'%(str(repeat-1)), map_location='cpu')
		model.load_state_dict(checkpoint['model'], strict=False)
		model.to(device)

		model_without_ddp = model
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
			model_without_ddp = model.module

		dataset_st_tr_f0 = evaluate(data_loader_st_tr_e, model, device, args.task, stage, mode="test")
		dataset_sb_tr_f0 = evaluate(data_loader_sb_tr_e, model, device, args.task, stage, mode="test")
		dataset_sb_va_f0 = evaluate(data_loader_sb_va_e, model, device, args.task, stage, mode="test")

		# stage 2, semi-supervised learning
		stage = 2

		print(f"Start training stage 2")
		start_time = time.time()

		ft_shape = 1024

		dataset_sb_va_f = dataset_sb_va_f0.copy()
		dataset_sb_tr_f = dataset_sb_tr_f0.copy()
		dataset_st_tr_f = dataset_st_tr_f0.copy()
		while(True):
			fts_sb_tr_1 = dataset_sb_tr_f.iloc[:, 1:ft_shape+1].to_numpy()
			fts_sb_tr_2 = dataset_sb_tr_f.iloc[:, ft_shape + 1:ft_shape * 2 + 1].to_numpy()
			gt_sb_tr = dataset_sb_tr_f.iloc[:, dataset_sb_tr_f.shape[1]-1].to_numpy()
			det1.fit(fts_sb_tr_1, gt_sb_tr)
			det2.fit(fts_sb_tr_2, gt_sb_tr)

			fts_sb_va_1 = dataset_sb_va_f.iloc[:, 1:ft_shape+1].to_numpy()
			fts_sb_va_2 = dataset_sb_va_f.iloc[:, ft_shape + 1:ft_shape * 2 + 1].to_numpy()
			pre_sb_va_1 = det1.predict_proba(fts_sb_va_1)[:,1]
			pre_sb_va_2 = det2.predict_proba(fts_sb_va_2)[:,1]

			pos_t = []
			neg_t = []
			gt_sb_va = dataset_sb_va_f.iloc[:, dataset_sb_va_f.shape[1] - 1].to_numpy()
			print("performance ", roc_auc_score(gt_sb_va, pre_sb_va_1), roc_auc_score(gt_sb_va, pre_sb_va_2))

			for pre_sb_va in [pre_sb_va_1, pre_sb_va_2]:
				pre_srt = np.array(sorted(set(pre_sb_va)))
				pos_rate = []
				neg_rate = []
				for p in pre_srt:
					pos_srt = pre_sb_va[pre_sb_va > p]
					neg_srt = pre_sb_va[pre_sb_va <= p]
					if pos_srt.shape[0] > 0:
						gt_pos = gt_sb_va[pre_sb_va > p]
						pos_rate.append(gt_pos[gt_pos > 0].shape[0] / pos_srt.shape[0])
					else:
						pos_rate.append(pos_rate[-1])
					gt_neg = gt_sb_va[pre_sb_va <= p]
					neg_rate.append(gt_neg[gt_neg < 1].shape[0] / neg_srt.shape[0])
				pos_res = np.array(pos_rate) - pos_thr
				neg_res = np.array(neg_rate) - neg_thr
				if pos_res[pos_res > 0].shape[0] > 0:
					pre_pos = pre_srt[pos_res > 0]
					pos_t.append(pre_pos[np.argmin(pos_res[pos_res > 0])])
				else:
					pos_t.append(2)
				if neg_res[neg_res > 0].shape[0] > 0:
					pre_neg = pre_srt[neg_res > 0]
					neg_t.append(pre_neg[np.argmin(neg_res[neg_res > 0])])
				else:
					neg_t.append(2)

			fts_st_tr_1 = dataset_st_tr_f.iloc[:, 1:ft_shape + 1].to_numpy()
			fts_st_tr_2 = dataset_st_tr_f.iloc[:, ft_shape + 1:ft_shape * 2 + 1].to_numpy()
			pre_st_sb_1 = det1.predict_proba(fts_st_tr_1)[:,1]
			pre_st_sb_2 = det2.predict_proba(fts_st_tr_2)[:,1]

			pos_sft0 = set(np.where(pre_st_sb_1 >= pos_t[0])[0]) & set(np.where(pre_st_sb_2 >= pos_t[1])[0])
			neg_sft0 = set(np.where(pre_st_sb_1 <= neg_t[0])[0]) & set(np.where(pre_st_sb_2 <= neg_t[1])[0])
			pos_sft = sorted(pos_sft0 - neg_sft0)
			neg_sft = sorted(neg_sft0 - pos_sft0)
			res_sft = sorted(set(np.arange(dataset_st_tr_f.shape[0])) - set(pos_sft) - set(neg_sft))

			if len(pos_sft) == 0 or len(neg_sft) == 0 or len(pos_sft + neg_sft) == dataset_st_tr_f.shape[0]:
				repeat_performance_sb[0].append(roc_auc_score(gt_sb_va, pre_sb_va_1))
				repeat_performance_sb[1].append(roc_auc_score(gt_sb_va, pre_sb_va_2))
				break
			else:
				dataset_st_sb_0 = dataset_st_tr_f.iloc[pos_sft, 0:ft_shape * 2 + 1]
				dataset_st_sb_0["fake"] = 1
				dataset_st_sb_1 = dataset_st_tr_f.iloc[neg_sft, 0:ft_shape * 2 + 1]
				dataset_st_sb_1["fake"] = 0
				dataset_st_sb = pd.concat([dataset_st_sb_0, dataset_st_sb_1])
				dataset_st_sb.columns = dataset_sb_tr_f.columns.values

				dataset_sb_tr_f = pd.concat([dataset_sb_tr_f, dataset_st_sb])
				dataset_st_tr_f = dataset_st_tr_f.iloc[res_sft]

		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print('Stage 2 Training time {}'.format(total_time_str))

		# stage 3
		fts_st_tr_1 = dataset_st_tr_f0.iloc[:, 1:ft_shape + 1].to_numpy()
		fts_st_tr_2 = dataset_st_tr_f0.iloc[:, ft_shape + 1:ft_shape * 2 + 1].to_numpy()
		st_sb_sft_1 = det1.predict_proba(fts_st_tr_1)
		dataset_st_tr_fake_1 = pd.DataFrame(st_sb_sft_1)
		st_sb_sft_2 = det2.predict_proba(fts_st_tr_2)
		dataset_st_tr_fake_2 = pd.DataFrame(st_sb_sft_2)

		# stage 4
		stage = 4

		print(f"Start training stage 4")
		start_time = time.time()
		max_auc_s4 = 0.0
		ptimes = 5
		ctimes = 0
		dataset_st_tr_s40= pd.concat([dft_st_tr, dataset_st_tr_fake_1, dataset_st_tr_fake_2], axis=1)
		dataset_st_tr_s4 = build_dataset('train', dataset_st_tr_s40, 4, data_path=path)
		sampler_st_tr_s4 = torch.utils.data.DistributedSampler(dataset_st_tr_s4, num_replicas=num_tasks,
															   rank=global_rank, shuffle=True)
		data_loader_st_tr_s4 = torch.utils.data.DataLoader(
			dataset_st_tr_s4, sampler=sampler_st_tr_s4,
			batch_size=sample_size,
			num_workers=args.num_workers,
			pin_memory=args.pin_mem,
			drop_last=True,
		)

		for epoch in range(1):
			if args.distributed:
				data_loader_st_tr_s4.sampler.set_epoch(epoch)
			train_stats = train_one_epoch(
				model, data_loader_st_tr_s4,
				optimizer, device, epoch, loss_scaler,
				stage, None, weight,
				args.clip_grad, mixup_fn,
				log_writer=log_writer,
				args=args
			)

			val_auc_roc = evaluate(data_loader_st_va_s1, model, device, args.task, stage, mode="eva")
			print("test performance ", val_auc_roc)

			if max_auc_s4 < val_auc_roc:
				max_auc_s4 = val_auc_roc
				ctimes = 0

				if args.output_dir:
					misc.save_model(
						args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
						loss_scaler=loss_scaler, epoch=epoch, best_weight=True, stage=stage, repeat=str(repeat))
			else:
				ctimes = ctimes + 1
				if ctimes > ptimes:
					repeat_performance_st.append(max_auc_s4)
					break

			if log_writer is not None:
				#log_writer.add_scalar('perf/val_acc1', val_stats['acc1'], epoch)
				log_writer.add_scalar('perf/val_auc', val_auc_roc, epoch)
				#log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)

			log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
						 'epoch': epoch,
						 'n_parameters': n_parameters}

			if args.output_dir and misc.is_main_process():
				if log_writer is not None:
					log_writer.flush()
				with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
					f.write(json.dumps(log_stats) + "\n")

		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print('Stage 4 Training time {}'.format(total_time_str))

if __name__ == '__main__':
	args = get_args_parser()
	args = args.parse_args()

	if args.output_dir:
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	main(args)
