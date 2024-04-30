# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    f1_score, average_precision_score,multilabel_confusion_matrix,confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from pycm import *
import matplotlib.pyplot as plt
import numpy as np


def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_

def train_one_epoch_0(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, pids, meta_data, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        meta_data = meta_data.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.combined:
                _, outputs = model([samples, meta_data])
            else:
                outputs = model([samples, meta_data])

            if args.pre_train:
                loss = 0
                for k in range(3):
                    mask = torch.ones_like(targets, dtype=torch.float,  requires_grad=False).to(device, non_blocking=True)
                    num_0 = mask[targets == 0].sum()
                    num_1 = mask[targets == 1].sum()
                    if num_1 > 0:
                        class_weight = torch.tensor([1, num_0 / num_1], dtype=torch.float).to(device, non_blocking=True)
                    else:
                        class_weight = torch.tensor([1, 1], dtype=torch.float).to(device, non_blocking=True)
                    loss_p = torch.nn.CrossEntropyLoss(weight=class_weight)
                    loss += loss_p(outputs[k], targets[:,k])
            elif args.progression:
                mask = torch.ones_like(targets, dtype=torch.float, requires_grad=False).to(device, non_blocking=True)
                mask[targets == -1] = 0
                num_0 = mask[targets == 0].sum()
                num_1 = mask[targets == 1].sum()
                if num_1 > 0:
                    mask[targets == 1] = num_0 / num_1

                loss_p = torch.nn.BCEWithLogitsLoss(reduction="none")
                loss = torch.sum(loss_p(outputs, targets.float()) * mask) / mask.sum()
            else:
                mask = torch.ones_like(targets, dtype=torch.float, requires_grad=False).to(device, non_blocking=True)
                num_0 = mask[targets == 0].sum()
                num_1 = mask[targets == 1].sum()
                if num_1 > 0:
                    class_weight = torch.tensor([1, num_0 / num_1], dtype=torch.float).to(device, non_blocking=True)
                else:
                    class_weight = torch.tensor([1, 1], dtype=torch.float).to(device, non_blocking=True)
                loss_p = torch.nn.CrossEntropyLoss(weight=class_weight)
                loss = loss_p(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    stage, metadata = None, weight = [1, 4],
                    max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (pids, samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            if stage == 1 or stage == "fp":
                sps = torch.cat([samples[0].to(device, non_blocking=True),
                                 samples[1].to(device, non_blocking=True)], dim=0)
                tgs = torch.cat([targets.to(device, non_blocking=True)] * 2, dim=0)
                outputs = model([sps, None, stage])
                pt0 = []
                pt1 = []
                for k in range(tgs.shape[0]):
                    if tgs[k,-1] > 0:
                        pt1.append(k)
                    else:
                        pt0.append(k)
                mask = torch.ones_like(tgs, dtype=torch.float, requires_grad=False).to(device, non_blocking=True)
                mask[tgs == -1] = 0
                mask[pt0, :] = mask[pt0, :] * weight[0]
                mask[pt1, :] = mask[pt1, :] * weight[1]

                loss_p = torch.nn.BCEWithLogitsLoss(reduction="none")
                loss = torch.sum(loss_p(outputs, tgs.float()) * mask) / mask.sum()

            if stage == 2 or stage == "fd":
                for k in range(2):
                    samples[k] = samples[k].to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model([samples, metadata, stage])
                class_weight = torch.tensor(weight, dtype=torch.float).to(device, non_blocking=True)
                loss_p = torch.nn.CrossEntropyLoss(weight=class_weight)

                if False:
                    loss = []
                    for k in range(2):
                        if metadata[k] == 1:
                            loss.append(loss_p(outputs[k], targets))
                        else:
                            loss.append(0)
                else:
                    loss = 0
                    for k in range(2):
                        if metadata[k] == 1:
                            loss = loss + loss_p(outputs[k], targets)
                        else:
                            pass

            if stage == 4:
                sps = torch.cat([samples[0].to(device, non_blocking=True),
                                 samples[1].to(device, non_blocking=True)], dim=0)
                tgs_0 = torch.cat([targets[0].to(device, non_blocking=True)] * 2, dim=0)
                tgs_s = torch.cat([targets[1].to(device, non_blocking=True),
                                   targets[2].to(device, non_blocking=True)], dim=0)

                outputs = model([sps, None, stage])
                pre_soft_log = torch.nn.functional.log_softmax(outputs[0], dim=1)
                loss_s = torch.nn.KLDivLoss(reduction="batchmean")
                loss_soft = loss_s(pre_soft_log, tgs_s)

                pt0 = []
                pt1 = []
                for k in range(tgs_0.shape[0]):
                    if tgs_0[k,-1] > 0:
                        pt1.append(k)
                    else:
                        pt0.append(k)
                mask = torch.ones_like(tgs_0, dtype=torch.float, requires_grad=False).to(device, non_blocking=True)
                mask[tgs_0 == -1] = 0
                mask[pt0, :] = mask[pt0, :] * weight[0]
                mask[pt1, :] = mask[pt1, :] * weight[1]

                loss_p = torch.nn.BCEWithLogitsLoss(reduction="none")
                loss_pre = torch.sum(loss_p(outputs[1], tgs_0.float()) * mask) / mask.sum()

                # loss = loss_pre + args.alpha * loss_soft
                loss = loss_pre + 0.3 * loss_soft

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_0(data_loader, model, device, task, epoch, mode,
             pre_train=False, progression=False, combined=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    try:
        os.mkdir(task)
    except:
        pass

    pre_prediction_list = [[],[],[]]
    pre_true_label_decode_list = [[],[],[]]

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []

    pid_list = []
    output_list = []
    target_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        pid_list += batch[1]
        metadata = batch[2]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if combined:
                _, output = model([images, metadata])
            else:
                output = model([images, metadata])
            if pre_train == True:
                loss = criterion(output[0], target[:,0])
                acc1, _ = accuracy(output[0], target[:,0], topk=(1, 2))

                for k in range(3):
                    pre_softmax = nn.Softmax(dim=1)(output[k]) # p*2
                    pre_prediction_list[k].extend(pre_softmax[:,1].cpu().detach().numpy()) # p*1
                    pre_true_label_decode_list[k].extend(target[:,k].cpu().detach().numpy()) # p*1
                    if k == 0:
                        prediction_softmax = pre_softmax # p*2
                        true_label = F.one_hot(target[:,0].to(torch.int64), num_classes = 2) # p*2
            elif progression == True:
                flat_output = torch.flatten(output)
                flat_target = torch.flatten(target)
                pre_flat = torch.stack((1 - flat_output[flat_target > -1], flat_output[flat_target > -1])).T
                gt_flat = flat_target[flat_target > -1]
                loss = criterion(pre_flat, gt_flat)
                acc1, _ = accuracy(pre_flat, gt_flat, topk=(1, 2))

                prediction_softmax = nn.Softmax(dim=1)(pre_flat)
                true_label = F.one_hot(gt_flat.to(torch.int64), num_classes=2)
                bce_output = torch.nn.Sigmoid()(output)
                output_list.append(bce_output.cpu().detach().numpy())
                target_list.append(target.cpu().detach().numpy())
            else:
                loss = criterion(output, target)
                acc1, _ = accuracy(output, target, topk=(1, 2))

                prediction_softmax = nn.Softmax(dim=1)(output)
                true_label = F.one_hot(target.to(torch.int64), num_classes=2)
                sft_output = torch.nn.Softmax()(output)
                output_list.append(sft_output.cpu().detach().numpy())
                target_list.append(target.cpu().detach().numpy())

            prediction_decode = prediction_softmax[:,1]
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy()) # p*1
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy()) # p*1
            true_label_onehot_list.extend(true_label.cpu().detach().numpy()) # p*2
            prediction_list.extend(prediction_softmax.cpu().detach().numpy()) # p*2

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # gather the stats from all processes
    for p in range(3):
        pre_true_label_decode_list[p] = np.array(pre_true_label_decode_list[p])
        pre_prediction_list[p] = np.array(pre_prediction_list[p])
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)

    fpr, tpr, thr = roc_curve(true_label_decode_list, prediction_decode_list)
    yd_thr = thr[np.argmax(tpr-fpr)]
    prediction_decode_list = np.array([1 if p > yd_thr else 0 for p in prediction_decode_list])

    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(2)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')          
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    results_path = task+'_metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)

    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
        output_mat = np.concatenate(output_list, axis=0)
        target_mat = np.concatenate(target_list, axis=0)
        if not output_mat.shape == target_mat.shape:
            target_mat = np.expand_dims(target_mat, axis=1)
        data = np.concatenate([output_mat, target_mat], axis=1)
        dft = pd.DataFrame(data, index=pid_list)

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_roc, dft

    if pre_train == True:
        for p in range(3):
            pre_score = roc_auc_score(pre_true_label_decode_list[p], pre_prediction_list[p])
            if pre_score < 0.7:
                auc_roc = 0
                break

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc


@torch.no_grad()
def evaluate(data_loader, model, device, task_folder, stage, mode):
    #criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    try:
        os.mkdir(task_folder)
    except:
        pass

    # evaluation mode
    prediction_decode_list = []
    true_label_decode_list = []

    # test mode
    pid_list = []
    output_list = [[],[]]
    target_list = []

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        pid_list += batch[0]
        samples = [sample.to(device, non_blocking=True) for sample in batch[1]]
        targets = batch[2]

        if mode == "test":
            target_list.append(targets.cpu().detach().numpy())
        else:
            pre_flats = []

        # compute output
        with torch.cuda.amp.autocast():
            # stage e = mode test
            if stage == "e":
                for k in range(2):
                    outputs = model([samples[k], None, stage])
                    output_list[k].append(outputs.cpu().detach().numpy())
            if stage == 1 or stage == "fp":
                if mode == "eva":
                    flat_target = torch.flatten(targets)
                    gt_flat = flat_target[flat_target > -1].to(torch.int64)
                for k in range(2):
                    outputs = model([samples[k], None, stage])
                    bce_output = torch.nn.Sigmoid()(outputs)
                    if mode == "eva":
                        flat_output = torch.flatten(bce_output)
                        pre_flat = flat_output[flat_target > -1]
                        pre_flats.append(pre_flat)
                    else:
                        output_list[k].append(bce_output.cpu().detach().numpy())
                #loss = criterion(pre_flat, gt_flat)
                #acc1 = accuracy_score(gt_flat, pre_flat)
            if stage == 2 or stage == "fd":
                if mode == "eva":
                    gt_flat = targets.to(torch.int64)
                outputs = model([samples, [1,1], stage])
                for k in range(2):
                    sft_output = torch.nn.Softmax()(outputs[k])
                    if mode == "eva":
                        pre_flat = sft_output[:,1]
                        pre_flats.append(pre_flat)
                    else:
                        output_list[k].append(sft_output.cpu().detach().numpy())
                #loss = criterion(pre, gt)
                #acc1 = accuracy_score(gt, pre)
            if stage == 4:
                if mode == "eva":
                    flat_target = torch.flatten(targets)
                    gt_flat = flat_target[flat_target > -1].to(torch.int64)
                for k in range(2):
                    outputs = model([samples[k], None, stage])
                    bce_output = torch.nn.Sigmoid()(outputs[1])
                    if mode == "eva":
                        flat_output = torch.flatten(bce_output)
                        pre_flat = flat_output[flat_target > -1]
                        pre_flats.append(pre_flat)
                    else:
                        output_list[k].append(bce_output.cpu().detach().numpy())
                #loss = criterion(pre_flat, gt_flat)
                #acc1 = accuracy_score(gt_flat, pre_flat)

            if mode == "eva":
                for k in range(2):
                    print(pre_flats)
                    prediction_decode_list.extend(pre_flats[k].cpu().detach().numpy())  # p*1
                    true_label_decode_list.extend(gt_flat.cpu().detach().numpy())  # p*1


        #batch_size = samples.shape[0]
        #metric_logger.update(loss=loss.item())
        #metric_logger.meters['acc1'].update(acc1, n=batch_size)

    if mode == "eva":
        true_label_decode_list = np.array(true_label_decode_list)
        prediction_decode_list = np.array(prediction_decode_list)

        fpr, tpr, thr = roc_curve(true_label_decode_list, prediction_decode_list)
        auc_roc = roc_auc_score(true_label_decode_list, prediction_decode_list)
        auc_pr = average_precision_score(true_label_decode_list, prediction_decode_list)
        yd_thr = thr[np.argmax(tpr - fpr)]
        prediction_decode_list = np.array([1 if p > yd_thr else 0 for p in prediction_decode_list])

        confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,
                                                       labels=[i for i in range(2)])
        acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)

        metric_logger.synchronize_between_processes()

        print('Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc,
                                                                                               auc_pr, F1, mcc))
        results_path = os.path.join(task_folder, '_metrics_test.csv')
        with open(results_path, mode='a', newline='', encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            data2 = [[acc, sensitivity, specificity, precision, auc_roc, auc_pr, F1, mcc]]
            for i in data2:
                wf.writerow(i)

        return auc_roc
    else:
        pids = np.expand_dims(np.array(pid_list), axis=1)
        pre_res = []
        for k in range(2):
            pre_res.append(np.concatenate(output_list[k], axis=0))
        gt_res = np.concatenate(target_list, axis=0)
        if stage == 2 or stage == "fd":
            gt_res = np.expand_dims(gt_res, axis=1)
        data = np.concatenate([pids, pre_res[0], pre_res[1], gt_res], axis=1)
        dft = pd.DataFrame(data)

        return dft