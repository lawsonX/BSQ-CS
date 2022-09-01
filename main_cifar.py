import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random
from models.cifar.resnetcs import ResNet
import logging
import torchvision.models as models
from utils_ import *
import KD_loss
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description='Training a ResNet on CIFAR-10 with Continuous Sparsification')
# parser.add_argument('--which-gpu', type=int, default=0, help='which GPU to use')
parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training/val/test (default: 128)')
parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train (default: 300)')
parser.add_argument('--class', type=int, default=10, help='class of output')
parser.add_argument('--Nbits', type=int, default=2, help='quantization bitwidth for weight')
parser.add_argument('--target-Nbit', type=float, default=2, help='Target Nbit')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 2)')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--lmbda', type=float, default=0.1, help='lambda for L1 mask regularization (default: 0.01)')
parser.add_argument('--final-temp', type=float, default=200, help='temperature at the end of each round (default: 200)')
parser.add_argument('--save_dir', type=str, default='train_result/0901/CIFAR_T2N2_LR005_LD01_newT', help='save path of weight and log files')
parser.add_argument('--log_file', type=str, default='train.log', help='save path of weight and log files')
parser.add_argument('--teacher', type=str, default='resnet18', help='path of ImageNet')
args = parser.parse_args()


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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    writer = SummaryWriter(args.save_dir)

    train_log_filepath = os.path.join(args.save_dir, args.log_file)
    logger = get_logger(train_log_filepath)

    #prepare dataset and preprocessing
    transform_train = transform.Compose([
        transform.RandomCrop(32, padding=4),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transform.Compose([
        transform.ToTensor(),
        transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)

    #labels in CIFAR10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #define ResNet20
    model = ResNet(
        num_classes=10,
        Nbits=args.Nbits, 
        bin=True
        ).to(device)
    logger.info(model)

    # print("Loading teacher model....", args.teacher)
    # model_teacher = models.__dict__[args.teacher](pretrained=True).to(device)
    # fc_features = model_teacher.fc.in_features
    # model_teacher.fc = nn.Linear(fc_features,10).to(device)

    # for p in model_teacher.parameters():
    #     p.requires_grad = False
    # model_teacher.eval()

    #define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    # criterion_smooth = CrossEntropyLabelSmooth(10, 0.1)
    # criterion_smooth = criterion_smooth.to(device)
    # criterion_kd = KD_loss.DistributionLoss()
    # criterion_kd = nn.KLDivLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    TP = model.total_param()

    #train
    logger.info('start training!')
    best_acc = 0
    iters_per_reset = args.epochs-1
    # temp_increase = args.final_temp**(1./iters_per_reset)
    temp_increase = 100**(1./args.epochs)
    for epoch in range(0, args.epochs):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        
        # update global temp
        if epoch > 0: model.temp = min(model.temp * temp_increase**epoch, args.final_temp)
        print('Current global temp:', round(model.temp,3))

        for i, data in enumerate(trainloader, 0):
            #prepare dataset
            length = len(trainloader)
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            #forward & backward
            logits_student = model(inputs)
            # logits_teacher = model_teacher(inputs)

            masks = [m.mask for m in model.mask_modules]
            mask_discrete = [m.mask_discrete for m in model.mask_modules]

            # for analyze only
            total_ele = 0
            ones = 0
            for iter in range(len(mask_discrete)):
                t = mask_discrete[iter].numel()
                o = (mask_discrete[iter] == 1).sum().item()
                z = (mask_discrete[iter] == 0).sum().item()
                total_ele += t
                ones += o
            ratio_one = ones/total_ele
            writer.add_scalar('Ratio of ones in bit mask', ratio_one, epoch)
            if i % 100 == 0 :
                print('Ratio of ones in bit mask', ratio_one)

            entries_sum = sum(m.sum() for m in masks)
            # Budget-aware adjusting lmbda according to Eq(4)
            Tsparsity = args.target_Nbit / args.Nbits
            loss = criterion(logits_student, labels)  + (args.lmbda*(Tsparsity - (1-ratio_one))) * entries_sum  #/TP # L1 Reg on mask
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            
            #print ac & loss in each batch
            sum_loss += loss.item()
            _, predicted = torch.max(logits_student.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            train_acc = 100. * correct / total
            if i % 100 == 0:
                logger.info('Epoch:[{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch+1 , sum_loss / (i + 1), train_acc ))
            writer.add_scalar('train loss', sum_loss / (i + 1), epoch)

        #get the ac with testdataset in each epoch
        print('Waiting Test...')
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                test_acc = (100 * correct / total)
            logger.info('Test\'s ac is: %.3f%%' % test_acc )
            writer.add_scalar('Test Acc', test_acc, epoch)

        # update temp_s based on sampled_iter per epoch
        for m in model.mask_modules:
            m.mask_discrete = torch.bernoulli(m.mask)
            m.sampled_iter += m.mask_discrete
            m.temp_s = temp_increase**m.sampled_iter
            print('sample_iter:', m.sampled_iter.tolist(), '  |  temp_s:', [round(item,3) for item in m.temp_s.tolist()])

        #save the model of the best epoch
        best_model_path = os.path.join(*[args.save_dir, 'model_best.pt'])
        if test_acc > best_acc:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_acc': test_acc,
            }, best_model_path)
            best_acc = test_acc
            best_epoch = epoch+1
        logger.info('Best Accuracy is %.3f%% at Epoch %d' %  (best_acc, best_epoch))
    TP = model.total_param()
    print('Train has finished, weight and log saved at ', str(args.save_dir))
    logger.info('model size is:', str(TP))