import csv
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
import math
from torchstat import stat
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from dataset.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
# from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from dataset.blending import CutmixMixupBlending
from utils.config import get_config
from models import clipvqa
from sklearn import svm
import string
def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=r'D:\桌面\CLIPVQA\configs\k400\16_32.yaml')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",default=None,
        nargs='+',)
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config): 
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)

    model, _ = clipvqa.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    model = model.cpu()  

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = torch.nn.CosineEmbeddingLoss(reduce='mean')
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES, 
                                       smoothing=config.AUG.LABEL_SMOOTH, 
                                       mixup_alpha=config.AUG.MIXUP, 
                                       cutmix_alpha=config.AUG.CUTMIX, 
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        # criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
        criterion = torch.nn.CosineEmbeddingLoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy,_ = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)


    text_labels = generate_text(train_data)

    if config.TEST.ONLY_TEST:
        acc1 = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.4f}%")
        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)

        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn)

        acc1 = validate(val_loader, text_labels, model, config)
        if epoch==0:
            max_accuracy=acc1

        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.4f}%")
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.4f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)
    # config.defrost()
    # config.TEST.NUM_CLIP = 1
    # config.TEST.NUM_CROP = 3
    # config.freeze()
    # train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    # state_dict=torch.load('/home/fengchuang/VideoX-master/X-CLIP/distributed/exp/best.pth',map_location='cpu')
    # # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)
    # model.load_state_dict(state_dict,strict=False)
    # model_weight=load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)
    # acc1 = validate(val_loader, text_labels, model, config)
    # logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.4f}%")

def Pro_Lab(label):
    codebook=[0,1,2,3,4,5]
    # codebook = [0, 2.5, 5]
    # codebook = [0,0.5,1,1.5, 2,2.5,3,3.5, 4,4.5, 5]
    # codebook = [0, 0.25,0.5, 0.75,1,1.25, 1.5, 1.75,2,2.25, 2.5, 2.75,3, 3.25,3.5, 3.75,4, 4.25,4.5,4.75, 5]
    # codebook = [0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1.0, 1.1, 1.2, 1.3, 1.4, 1.5,1.6,1.7,1.8,1.9,2.0, 2.1, 2.2, 2.3, 2.4, 2.5,2.6,2.7,2.8,2.9,3.0, 3.1, 3.2, 3.3, 3.4, 3.5,3.6,3.7,3.8,3.9,4.0, 4.1, 4.2, 4.3, 4.4, 4.5,4.6,4.7,4.8,4.9,5.0]
    D=[]
    label=label   # If the range of MOS is an interval of 100 points, label=label/20.    
    for i in range(0,len(codebook)):
        dis_mos=math.exp(-(label-codebook[i])**2)
        D.append(dis_mos)
    return np.array(D)/sum(D)

def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    texts = text_labels.cuda(non_blocking=True)

    for idx, batch_data in enumerate(train_loader):
        # print("111112222")
        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True)
        label_id = label_id.reshape(-1)
        label_id=torch.as_tensor([Pro_Lab(label_id[i]) for i in range(len(label_id)) ])   #### new
        images = images.view((-1,config.DATA.NUM_FRAMES,3)+images.size()[-2:])

        # if mixup_fn is not None:
        #     images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        
        output = model(images, texts)

        # output=torch.nn.functional.softmax(output,1)

        total_loss= criterion(output, label_id.cuda(),torch.tensor([1]).cuda())
        # rank_loss = criterion(output1, output, torch.tensor([1]).cuda())
        # r_loss=rank_loss(output, label_id.cuda())
        # total_loss=loss+rank_loss
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()

        if config.TRAIN.OPT_LEVEL  == 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


svrdata_path='train.csv'
mos=[]
csv_reader=csv.reader(open(svrdata_path))
for row in csv_reader:
    mos.append(float(row[1])/20)
code_book=[0,1,2,3,4,5]
pro_label=[]
for j in range(0,len(mos)):
    D=[]
    for i in range(0,len(code_book)):
        D.append(math.exp(-(mos[j]-code_book[i])**2))
    pro_label.append(np.array(D)/sum(D))
svr = svm.SVR()
svr_model=svr.fit(pro_label,mos)



@torch.no_grad()
def validate(val_loader, text_labels, model, config):
    model.eval()

    result=[]
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        tot_similarity = []
        mos = []

        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id[0].reshape(-1)
            # label_id = torch.as_tensor([Pro_Lab(label_id[i]) for i in range(len(label_id))])  #### new


            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)
           
            # tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()

            for i in range(n):
                image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
                # label_id = label_id.cuda(non_blocking=True)
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                
                output = model(image_input, text_inputs)
                # output.sync()
                # similarity = output.view(b, -1)
                similarity = output.view(b, -1)
                # dist.all_gather(result, similarity)
                # similarity1=output1.view(b, -1).softmax(dim=-1)
                # similarity=similarity+similarity1
                predict_socre=svr_model.predict(similarity.cpu())
                for i in range(len(predict_socre.tolist())):
                    tot_similarity.append(predict_socre.tolist()[i])
                    mos.append(label_id.numpy()[i])
                # mos.append(label_id.numpy())
        SROCC=pd.DataFrame({'A':tot_similarity,'B':mos})
        plcc=SROCC.corr('pearson')
        srocc=SROCC.corr('spearman')
        result.append([srocc,plcc])
        plcc = [' '.join(repr(e) for e in result[i][0].loc['A'].iloc[[1]].to_list()) for i in range(len(result))]
        srocc = [' '.join(repr(e) for e in result[i][1].loc['A'].iloc[[1]].to_list()) for i in range(len(result))]
        acc=float(plcc[0])+float(srocc[0])
        print(float(plcc[0]))
        print(float(srocc[0]))
            # values_1, indices_1 = tot_similarity.topk(1, dim=-1)
            # values_5, indices_5 = tot_similarity.topk(5, dim=-1)
    #         acc1, acc5 = 0, 0
    #         for i in range(b):
    #             if indices_1[i] == label_id[i]:
    #                 acc1 += 1
    #             if label_id[i] in indices_5[i]:
    #                 acc5 += 1
    #
    #         acc1_meter.update(float(acc1) / b * 100, b)
    #         acc5_meter.update(float(acc5) / b * 100, b)
    #         if idx % config.PRINT_FREQ == 0:
    #             logger.info(
    #                 f'Test: [{idx}/{len(val_loader)}]\t'
    #                 f'Acc@1: {acc1_meter.avg:.3f}\t'
    #             )
    # acc1_meter.sync()
    # acc5_meter.sync()
    # logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '56781'
    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = 0
        world_size = 2

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://localhost:57842 ',rank=rank , world_size=world_size)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    # seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    # logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")

    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)    
    main(config)
    