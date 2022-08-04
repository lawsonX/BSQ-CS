import os
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
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Training a ResNet on CIFAR-10 with Continuous Sparsification')
# parser.add_argument('--which-gpu', type=int, default=0, help='which GPU to use')
parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='input batch size for training/val/test (default: 128)')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 85)')
parser.add_argument('--class', type=int, default=10, help='class of output')
parser.add_argument('--Nbits', type=int, default=8, help='quantization bitwidth for weight')
# parser.add_argument('--act_bit', type=int, default=4, help='quantization bitwidth for activation')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--lmbda', type=float, default=1e-8, help='lambda for L1 mask regularization (default: 1e-8)')
parser.add_argument('--final-temp', type=float, default=200, help='temperature at the end of each round (default: 200)')
# parser.add_argument('--mask-initial-value', type=float, default=0., help='initial value for mask parameters')
parser.add_argument('--save_dir', type=str, default='train_result/0804/withoutL1norm', help='save path of weight and log files')
args = parser.parse_args()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    writer = SummaryWriter(args.save_dir)
            
    iters_per_reset = args.epochs-1
    temp_increase = args.final_temp**(1./iters_per_reset)

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    #labels in CIFAR10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #define ResNet20
    model = ResNet(
        num_classes=10,
        Nbits=args.Nbits, 
        bin=True
        ).to(device)
    print(model)

    #define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)

    #train
    best_acc = 0
    for epoch in range(0, args.epochs):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        if epoch > 0: model.temp *= temp_increase
        print('Current epoch temp:', model.temp)
        print('current learning rate:', optimizer.param_groups[0]['lr'])
        # print('Ratio of ones in mask', ratio_one)

        for i, data in enumerate(trainloader, 0):
            #prepare dataset
            length = len(trainloader)
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            #forward & backward
            outputs = model(inputs)
            masks = [m.mask for m in model.mask_modules]
            mask_sample = [m.mask_sample for m in model.mask_modules]

            # for analyze only
            total_ele = 0
            ones = 0
            for iter in range(len(mask_sample)):
                t = mask_sample[iter].numel()
                o = (mask_sample[iter] == 1).sum().item()
                z = (mask_sample[iter] == 0).sum().item()
                total_ele += t
                ones += o
            ratio_one = ones/total_ele
            writer.add_scalar('Ratio of ones in bit mask', ratio_one, epoch)

            entries_sum = sum(m.sum() for m in masks)
            loss = criterion(outputs, labels)  #+ args.lmbda * entries_sum # L1 Reg on mask
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            #print ac & loss in each batch
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            train_acc = 100. * correct / total
            if i % 1000 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), train_acc))
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
            print('Test\'s ac is: %.3f%%' % test_acc )

        #save the model of the best epoch
        best_model_path = os.path.join(*[args.save_dir, 'model_best.pt'])
        if test_acc > best_acc:
            # torch.save(model.state_dict(), best_model_path)
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_acc': test_acc,
            }, best_model_path)
            best_acc = test_acc
            best_epoch = epoch+1
        print('Best Accuracy is %.3f%% at Epoch %d' %  (best_acc, best_epoch))
    print('Train has finished, weight and log saved at ', args.save_dir)
    