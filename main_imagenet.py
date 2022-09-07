import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from utils_ import *
# from utils import KD_loss
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models
from models.cifar.resnetcs50 import ResNet50
from models.cifar.resnetcs18 import ResNet18

# sys.path.append("../../")

parser = argparse.ArgumentParser("BinealNet")
parser.add_argument('--data', default='/home/datasets/imagenet', help='path to dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--teacher', type=str, default='resnet50', help='path of ImageNet')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--lmbda', type=float, default=1e-8, help='lambda for L1 mask regularization (default: 1e-8)')
parser.add_argument('--final-temp', type=float, default=200, help='temperature at the end of each round (default: 200)')
parser.add_argument('--act', type=int, default=0, help='quantization bitwidth for activation')
parser.add_argument('--target-Nbit', type=int, default=5, help='Target Nbit')
parser.add_argument('--Nbits', type=int, default=6, help='quantization bitwidth for weight')
parser.add_argument('--save', type=str, default='train_result/0908/IMG_A0T5N6H8T8', help='path for saving trained models')
parser.add_argument('--log_file', type=str, default='train.log', help='save path of weight and log files')

args = parser.parse_args()

CLASSES = 1000
if not os.path.exists(args.save):
    os.makedirs(args.save)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def main():
    writer = SummaryWriter(args.save)
    train_log_filepath = os.path.join(args.save, args.log_file)
    logger = get_logger(train_log_filepath)

    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True
    logging.info("args = %s", args)

    # load model
    model_teacher = None
    # model_teacher = models.__dict__[args.teacher](pretrained=True)
    # model_teacher = nn.DataParallel(model_teacher).cuda()
    # for p in model_teacher.parameters():
    #     p.requires_grad = False
    # model_teacher.eval()

    model_student = ResNet18(
        num_classes=CLASSES,
        Nbits=args.Nbits,
        act_bit = args.act,
        bin=True
        )
    logging.info('student:')
    logging.info(model_student)
    model_student = nn.DataParallel(model_student).cuda()

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    criterion_kd = nn.KLDivLoss()

    all_parameters = model_student.parameters()
    weight_parameters = []
    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.learning_rate,)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    start_epoch = 0
    best_top1_acc= 0

    checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        logger.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model_student.load_state_dict(checkpoint['state_dict'], strict=False)
        logger.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()

    # load training data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # train the model
    epoch = start_epoch
    best_acc = 0
    temp_increase = 100**(1./args.epochs)
    while epoch < args.epochs:
        train_obj, train_top1_acc,  train_top5_acc, ratio_ones = train(epoch,  train_loader, model_student, model_teacher, criterion, optimizer, scheduler, writer,logger)
        # update temp_s based on sampled_iter per epoch
        if epoch == args.epochs-1:
            for m in model_student.module.mask_modules:
                logger.info("prune int before last test......")
                m.mask= torch.where(m.mask >= 0.5, torch.full_like(m.mask, 1), m.mask)
                m.mask= torch.where(m.mask < 0.5, torch.full_like(m.mask, 0), m.mask)
        
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model_student, criterion, args, writer,logger)
        
        for m in model_student.module.mask_modules:
            m.mask_discrete = torch.bernoulli(m.mask)
            m.sampled_iter += m.mask_discrete
            m.temp_s = temp_increase**m.sampled_iter
            # print('sample_iter:', m.sampled_iter.tolist(), '  |  temp_s:', [round(item,3) for item in m.temp_s.tolist()])

        # save latest model
        save_name = os.path.join(*[args.save, 'model_latest.pt'])
        torch.save({
                'model': model_student.state_dict(),
                'epoch': epoch,
                'valid_acc': valid_top1_acc,
            }, save_name)

        # save the model of the best epoch
        best_model_path = os.path.join(*[args.save_dir, 'model_best.pt'])
        if valid_top1_acc > best_acc:
            torch.save({
                'model': model_student.state_dict(),
                'epoch': epoch,
                'valid_acc': valid_top1_acc,
            }, best_model_path)
            best_acc = valid_top1_acc
        logger.info('Best Accuracy is %.3f%% at Epoch %d' %  (best_acc, epoch))

        epoch += 1
    
    avg_bit = args.Nbits * ratio_ones
    logger.info('average bit is: %.3f ' % avg_bit)

    training_time = (time.time() - start_t) / 3600
    logger.info('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model_student, model_teacher, criterion, optimizer, scheduler, writer,logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    ratio_one = AverageMeter('ratio_one', ':6.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, ratio_one],
        prefix="Epoch: [{}]".format(epoch))

    model_student.train()
    # model_teacher.eval()
    end = time.time()
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate::%.3f'% cur_lr)

    # update global temp
    temp_increase = 100**(1./args.epochs)
    if epoch > 0: model_student.module.temp = temp_increase**epoch
    logger.info('Current global temp:%.3f'% round(model_student.module.temp,3))

    
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits_student = model_student(images)
        # logits_teacher = model_teacher(images)

        masks = [m.mask for m in model_student.module.mask_modules]
        mask_discrete = [m.mask_discrete for m in model_student.module.mask_modules]
        # for analyze bit mask
        total_ele = 0
        ones = 0
        for iter in range(len(mask_discrete)):
            t = mask_discrete[iter].numel()
            o = (mask_discrete[iter] == 1).sum().item()
            z = (mask_discrete[iter] == 0).sum().item()
            total_ele += t
            ones += o
        ratio_ones = ones/total_ele
        writer.add_scalar('Ratio of ones in bit mask', ratio_ones, epoch) 

        entries_sum = sum(m.sum() for m in masks)
        
        # Budget-aware adjusting lmbda according to Eq(4)
        Tsparsity = args.target_Nbit / args.Nbits
        loss = criterion(logits_student, target)+(args.lmbda*(Tsparsity - (1-ratio_ones))) * entries_sum

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        ratio_one.update(ratio_ones)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)
        writer.add_scalar('train loss', losses.avg, epoch)

    return losses.avg, top1.avg, top5.avg, ratio_ones

def validate(epoch, val_loader, model, criterion, args, writer,logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        logger.info(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        writer.add_scalar('Test Acc', top1.avg, epoch)

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()