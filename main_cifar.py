from lib2to3.pgen2.grammar import opmap_raw
import os
import argparse
import torch
import torch.nn as nn
from torch.nn import Parameter
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import random
from models.cifar.resnetcs import ResNet
from torch.utils.tensorboard import SummaryWriter
import logging
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description='Training a ResNet on CIFAR-10 with Continuous Sparsification')
# parser.add_argument('--which-gpu', type=int, default=0, help='which GPU to use')
parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training/val/test (default: 128)')
parser.add_argument('--epochs', type=int, default=600, help='number of epochs to train (default: 300)')
parser.add_argument('--class', type=int, default=10, help='class of output')
parser.add_argument('--Nbits', type=int, default=6, help='quantization bitwidth for weight')
parser.add_argument('--target', type=int, default=3, help='Target Nbit')
parser.add_argument('--act', type=int, default=0, help='quantization bitwidth for activation')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--goal_acc', type=float, default=91, help='weight decay (91)')
parser.add_argument('--lmbda', type=float, default=0.001, help='lambda for L1 mask regularization (default: 1e-8)')
parser.add_argument('--final-temp', type=float, default=200, help='temperature at the end of each round (default: 200)')
parser.add_argument('--final-temps', type=float, default=160, help='temperature of temp_s based on estimated maximum value of sampled iter')
parser.add_argument('--save_dir', type=str, default='train_result/0919/test', help='save path of weight and log files')
parser.add_argument('--log_file', type=str, default='train.log', help='save path of weight and log files')
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

def get_ratio_one(model):
    mask_discrete = [m.mask_discrete for m in model.mask_modules]
    total_ele = 0
    ones = 0
    for iter in range(len(mask_discrete)):
        t = mask_discrete[iter].numel()
        o = (mask_discrete[iter] == 1).sum().item()
        # z = (mask_discrete[iter] == 0).sum().item()
        total_ele += t
        ones += o
    ratio_one = ones/total_ele

    return ratio_one

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    writer = SummaryWriter(args.save_dir)

    train_log_filepath = os.path.join(args.save_dir, args.log_file)
    logger = get_logger(train_log_filepath)

    logger.info("args = %s", args)

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
        act_bit = args.act,
        bin=True
        ).to(device)

    #define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    # optimizer = optim.Adam(model.parameters(),lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs/4, eta_min=0.001, last_epoch=-1)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100, 150], last_epoch=-1)
    #train
    logger.info('start training!')
    best_acc = 0
    solid_best_acc =0
    # temp_increase = 200**(1./args.epochs)
    temp_increase = 200**(1./(args.epochs/2))
    lr=[]
    for epoch in range(0, args.epochs):
        print('\nEpoch: %d' % (epoch + 1))
        # adjust_learning_rate(optimizer, epoch, args)

        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        
        # update global temp
        if epoch <= args.epochs/2:
            model.temp = temp_increase**epoch
        else:
            _epoch = epoch - (args.epochs/2)
            model.temp = temp_increase**_epoch

        logger.info('Current global temp:%.3f'% round(model.temp,3))

        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            lr.append(scheduler.get_lr()[0])
            
            outputs = model(inputs)

            # get sparsity of network at current epoch
            ratio_one = get_ratio_one(model)
            writer.add_scalar('Ratio of ones in bit mask', ratio_one, epoch)

            # Budget-aware adjusting lmbda according to Eq(4)
            TS = args.target / args.Nbits  # target ratio of ones of masks in the network
            regularization_loss = 0
            for m in model.mask_modules:
                regularization_loss += torch.sum(torch.abs(m.mask).sum())
            classify_loss = criterion(outputs, labels)
            loss = classify_loss + (args.lmbda*((1-TS)-(1-ratio_one))) * regularization_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
       
            
            #print ac & loss in each batch
            lrr = optimizer.state_dict()['param_groups'][0]['lr']
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            train_acc = 100. * correct / total
            if i % 50 == 0:
                logger.info('Epoch:[{}]\t lr={:.4f}\t Ratio_ones={:.5f}\t loss={:.5f}\t acc={:.3f}'.format(epoch+1,lrr,ratio_one,sum_loss/(i+1),train_acc ))
            writer.add_scalar('train loss', sum_loss / (i + 1), epoch)
        scheduler.step()
        save_name = os.path.join(*[args.save_dir, 'model_latest.pt'])
        torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_acc': train_acc,
                'ratio_one':ratio_one,
            }, save_name)

        # test with soft mask
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
        
        if epoch > args.epochs/2 and test_acc > args.goal_acc:
            # Turn soft mask to discrete
            for m in model.mask_modules:
                m.mask= torch.where(m.mask >= 0.5, torch.full_like(m.mask, 1), m.mask)
                m.mask= torch.where(m.mask < 0.5, torch.full_like(m.mask, 0), m.mask)
                m.mask_discrete = torch.bernoulli(m.mask)
            # test again after finalizing the soft bitmask to 0&1
            with torch.no_grad():
                _correct = 0
                _total = 0
                for data in testloader:
                    model.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, _predicted = torch.max(outputs.data, 1)
                    _total += labels.size(0)
                    _correct += (_predicted == labels).sum()
                    _test_acc = (100 * _correct / total)
                logger.info('Solid Test\'s ac is: %.3f%%' % _test_acc )
            ratio_one = get_ratio_one(model)
            solid_best_model_path = os.path.join(*[args.save_dir, 'solid_model_best.pt'])
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_acc': _test_acc,
                'solid_ratio_one': ratio_one,
            }, solid_best_model_path)
            solid_best_acc = _test_acc
            _best_epoch = epoch+1
            avg_bit_ = ratio_one * args.Nbits
            logger.info('best Solid Accuracy is %.3f%% , average bit is %.2f%% at epoch %d' %  (solid_best_acc, avg_bit_, _best_epoch))
        
        # update temp_s based on sampled_iter per epoch
        if epoch < args.epochs/2:
            for m in model.mask_modules:
                m.mask_discrete = torch.bernoulli(m.mask)
                m.sampled_iter += m.mask_discrete
                m.temp_s = temp_increase**m.sampled_iter
                if epoch == args.epochs/2:
                    print('sample_iter:', m.sampled_iter.tolist(), '  |  temp_s:', [round(item,3) for item in m.temp_s.tolist()])
        
    TP = model.total_param()
    avg_bit = args.Nbits * ratio_one
    logger.info('model size is: %.3f' % TP)
    logger.info('average bit is: %.3f ' % avg_bit)
    plt.plot(np.arange(len(lr)), lr)
    plt.savefig('aa.jpg')