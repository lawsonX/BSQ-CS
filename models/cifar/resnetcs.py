from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch
import torch.nn as nn
import math
from .bitcs import BitLinear, BitConv2d
# from .bitcss import BitLinear, BitConv2d
import numpy as np
import copy


class PACTFunction(torch.autograd.Function):
    """
    Parametrized Clipping Activation Function
    https://arxiv.org/pdf/1805.06085.pdf
    Code from https://github.com/obilaniu/GradOverride
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.clamp(min=0.0).min(alpha)
    @staticmethod
    def backward(ctx, dLdy):
        x, alpha = ctx.saved_variables
        lt0 = x < 0
        gta = x > alpha
        gi = 1.0-lt0.float()-gta.float()
        dLdx = dLdy*gi
        dLdalpha = torch.sum(dLdy*x.ge(alpha).float()) 
        return dLdx, dLdalpha

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit):
        if bit==0:
            # No quantization
            act = x
        else:
            S = torch.max(torch.abs(x))
            if S==0:
                act = x*0
            else:
                step = 2 ** (bit)-1
                R = torch.round(torch.abs(x) * step / S)/step
                act =  S * R * torch.sign(x)
        return act

    @staticmethod
    def backward(ctx, g):
        return g, None

class PACT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.relu = nn.ReLU6(inplace=True)
	
    def forward(self, x):
	        return PACTFunction.apply(x, self.alpha)#


__all__ = ['resnet']

#def conv3x3(in_planes, out_planes, stride=1):
#    "3x3 convolution with padding"
#    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                     padding=1, bias=False)
                     
def conv3x3(in_planes, out_planes, stride=1, Nbits=4, bin=True):
    "3x3 convolution with padding"
    return BitConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, Nbits = Nbits, bin=bin)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, Nbits=4, act_bit=4, bin=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, Nbits=Nbits, bin=bin)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, Nbits=Nbits, bin=bin)
        self.bn2 = nn.BatchNorm2d(planes)
        if act_bit>3:
            self.relu1 = nn.ReLU6(inplace=True) 
            self.relu2 = nn.ReLU6(inplace=True) 
        else:
            self.relu1 = PACT()
            self.relu2 = PACT()
        self.downsample = downsample
        self.stride = stride
        self.act_bit = act_bit

    def forward(self, x, epoch, temp, ticket):
        # x, temp = input
        residual = x

        out = self.conv1(x, epoch, temp, ticket)
        out = self.bn1(out)
        out = self.relu1(out)

        # out = STE.apply(out,self.act_bit)

        out = self.conv2(out, epoch, temp)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        
        # out = STE.apply(out,self.act_bit)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MaskedNet(nn.Module):
    def __init__(self):
        super(MaskedNet, self).__init__()
        self.ticket = False

    def checkpoint(self):
        for m in self.mask_modules: m.checkpoint()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules: m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)
                
    def prune(self):
        for m in self.mask_modules: m.prune(self.temp)

class ResStage(nn.Module):
    def __init__(self, in_planes, out_planes, stride, padding, Nbits=4, bin=True, bias=False):
        super(ResStage, self).__init__()
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            
        self.block1 = BasicBlock(in_planes, out_planes, stride=stride, downsample=downsample, Nbits=Nbits, act_bit=4, bin=False)
        self.block2 = BasicBlock(out_planes, out_planes, stride=1, downsample=None, Nbits=Nbits, act_bit=4, bin=False)
        self.block3 = BasicBlock(out_planes, out_planes, stride=1, downsample=None, Nbits=Nbits, act_bit=4, bin=False)

    def forward(self, x, epoch, temp, ticket):
        out = self.block1(x, epoch, temp, ticket)
        out = self.block2(out, epoch, temp, ticket)
        out = self.block3(out, epoch, temp, ticket)
        return out

class ResNet(MaskedNet):
    def __init__(self, num_classes=10, Nbits=8, bin=True):
        super(ResNet, self).__init__()

        self.conv1 = BitConv2d(3, 16, kernel_size=3, padding=1,bias=False, Nbits=Nbits, bin=bin)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResStage(16,16,1,1,Nbits,bin=bin)
        self.layer2 = ResStage(16,32,2,1,Nbits,bin=bin)
        self.layer3 = ResStage(32,64,2,1,Nbits,bin=bin)
        # self.layer4 = ResStage(64,128,2,1,Nbits,bin=bin)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = BitLinear(64, out_features=num_classes, Nbits=Nbits, bin=bin)
        self.mask_modules = [m for m in self.modules() if type(m) in [BitConv2d, BitLinear] ]
        self.temp = 1
        self.epoch = 0

        for m in self.modules():
            if isinstance(m, BitConv2d):
                if m.bin:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    ini_w = torch.full_like(m.pweight[...,0], 0)
                    ini_w.normal_(0, math.sqrt(2. / n))
                    m.ini2bit(ini_w)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x, self.epoch, self.temp, self.ticket)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x, self.epoch, self.temp, self.ticket)
        x = self.layer2(x, self.epoch, self.temp, self.ticket)
        x = self.layer3(x, self.epoch, self.temp, self.ticket)
        # x = self.layer4(x, self.temp)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, self.epoch, self.temp)

        return x
    
    def total_param(self):
        N = 0
        for name, m in self.named_modules():
            if (isinstance(m,BitLinear) or isinstance(m,BitConv2d)) and 'downsample' not in name:
                if m.bin:
                    param = m.pweight-m.nweight
                    N += np.prod(param.data.cpu().numpy().shape)/m.Nbits
                    if m.pbias is not None:
                        param = m.pbias-m.nbias
                        N += np.prod(param.data.cpu().numpy().shape)/m.bNbits
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape)
        return N
        
    def total_bit(self):
        N = 0
        for name, m in self.named_modules():
            if (isinstance(m,BitLinear) or isinstance(m,BitConv2d)) and 'downsample' not in name:
                if m.bin:
                    param = m.pweight-m.nweight
                    N += np.prod(param.data.cpu().numpy().shape)
                    if m.pbias is not None:
                        param = m.pbias-m.nbias
                        N += np.prod(param.data.cpu().numpy().shape)
                else:
                    param = m.weight
                    N += np.prod(param.data.cpu().numpy().shape)*m.Nbits
                    if m.bias is not None:
                        param = m.bias
                        N += np.prod(param.data.cpu().numpy().shape)*m.bNbits
        return N
    
    def get_Nbits(self):
        Nbit_dict = {}
        for name, m in self.named_modules():
            if isinstance(m,BitLinear) or isinstance(m,BitConv2d):
                Nbit_dict[name] = [m.Nbits, 0]
                if m.pbias is not None or m.bias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
        return Nbit_dict

    def set_Nbits(self, Nbit_dict):
        for name, m in self.named_modules():
            if isinstance(m,BitLinear) or isinstance(m,BitConv2d):
                N = Nbit_dict[name]
                N0 = N[0]
                N1 = N[1]
                ex = np.arange(N0-1, -1, -1)
                m.exps = torch.Tensor((2**ex)/(2**(N0)-1)).float()
                m.Nbits = N0
                if N1:
                    ex = np.arange(N1-1, -1, -1)
                    m.bexps = torch.Tensor((2**ex)/(2**(N1)-1)).float()         
                    m.bNbits = N1
                if m.bin:
                    m.pweight.data = m.pweight.data[...,0:N0]
                    m.nweight.data = m.nweight.data[...,0:N0]
                    if N1:
                        m.pbias.data = m.pbias.data[...,0:N1]
                        m.nbias.data = m.nbias.data[...,0:N1]
                        
    def set_zero(self):
        for name, m in self.named_modules():
            if isinstance(m,BitLinear) or isinstance(m,BitConv2d):
                weight = m.pweight.data-m.nweight.data
                if m.Nbits==1 and (np.count_nonzero(weight.cpu().numpy())==0):
                    m.zero=True
                else:
                    m.zero=False
                if m.pbias is not None:
                    weight = m.pbias.data-m.nbias.data
                    if m.bNbits==1 and (np.count_nonzero(weight.cpu().numpy())==0):
                        m.bzero=True
                    else:
                        m.bzero=False                    
    
    def pruning(self, threshold=1.0, drop=True):   #Use drop to control whether 0 bit after pruning will be removed, 0 bit before pruning will always be removed
        Nbit_dict = {}
        for name, m in self.named_modules():
            # print(name,m)
            if isinstance(m,BitLinear) or isinstance(m,BitConv2d):
                # import pdb; pdb.set_trace()
                if m.Nbits>1:
                    # Remove MSB
                    weight = m.pweight.data.cpu().numpy()-m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape)/m.Nbits
                    nonz_weight = [np.count_nonzero(weight[...,i])*100 for i in range(m.Nbits)]
                    nonz_weight = nonz_weight/total_weight
                    N = m.Nbits
                    N0 = m.Nbits
                    pweight = m.pweight.data
                    nweight = m.nweight.data
                    for i in range(N):
                        if nonz_weight[i]==0:
                            m.pweight.data = pweight[...,i+1:N]
                            m.nweight.data = nweight[...,i+1:N]
                            m.Nbits -= 1
                            if m.Nbits==1:
                                break
                        elif nonz_weight[i]<threshold: # set MSB to 0, remove MSB if "drop"
                            if drop:
                                m.pweight.data = pweight[...,i+1:N]+pweight[...,i].unsqueeze(-1)
                                m.nweight.data = nweight[...,i+1:N]+nweight[...,i].unsqueeze(-1)
                                m.Nbits -= 1
                                if m.Nbits==1:
                                    break
                            else:
                                m.pweight.data = pweight[...,i:N]+pweight[...,i].unsqueeze(-1)
                                m.nweight.data = nweight[...,i:N]+nweight[...,i].unsqueeze(-1)
                                m.pweight.data[...,0] = 0.
                                m.nweight.data[...,0] = 0.
                            m.pweight.data = torch.where(m.pweight.data < 1, m.pweight.data, torch.full_like(m.pweight.data, 1.))
                            m.nweight.data = torch.where(m.nweight.data < 1, m.nweight.data, torch.full_like(m.nweight.data, 1.))
                        else:
                            break
                    # Remove LSB                  
                    weight = m.pweight.data.cpu().numpy()-m.nweight.data.cpu().numpy()
                    total_weight = np.prod(weight.shape)/m.Nbits
                    nonz_weight = [np.count_nonzero(weight[...,i])*100 for i in range(m.Nbits)]
                    nonz_weight = nonz_weight/total_weight
                    N = m.Nbits
                    pweight = m.pweight.data
                    nweight = m.nweight.data
                    if m.Nbits>1:
                        for i in range(N):
                            if nonz_weight[N-1-i]<=threshold:
                                m.pweight.data = pweight[...,0:N-1-i]
                                m.nweight.data = nweight[...,0:N-1-i]
                                m.Nbits -= 1
                                m.scale.data = m.scale.data*2
                                if m.Nbits==1:
                                    break
                            else:
                                break
                    # Reset exps
                    N = m.Nbits 
                    ex = np.arange(N-1, -1, -1)
                    m.exps = torch.Tensor((2**ex)/(2**(N)-1)).float()
                    m.scale.data = m.scale.data*(2**(N)-1)/(2**(N0)-1)
                    ## Match the shape of grad to data
                    if m.pweight.grad is not None:
                        m.pweight.grad.data = m.pweight.grad.data[...,0:N]
                        m.nweight.grad.data = m.nweight.grad.data[...,0:N]
                # For bias
                if m.pbias is not None and m.bNbits>1:
                    # Remove MSB
                    weight = m.pbias.data.cpu().numpy()-m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape)/m.bNbits
                    nonz_weight = [np.count_nonzero(weight[...,i])*100 for i in range(m.bNbits)]
                    nonz_weight = nonz_weight/total_weight
                    N = m.bNbits
                    N0 = m.bNbits
                    pweight = m.pbias.data
                    nweight = m.nbias.data
                    for i in range(N):
                        if nonz_weight[i]==0:
                            m.pbias.data = pweight[...,i+1:N]
                            m.nbias.data = nweight[...,i+1:N]
                            m.bNbits -= 1
                            if m.bNbits==1:
                                break
                        elif nonz_weight[i]<threshold:
                            m.pbias.data = pweight[...,i+1:N]+pweight[...,i].unsqueeze(-1)
                            m.nbias.data = nweight[...,i+1:N]+nweight[...,i].unsqueeze(-1)
                            m.pbias.data = torch.where(m.pbias.data < 1, m.pbias.data, torch.full_like(m.pbias.data, 1.))
                            m.nbias.data = torch.where(m.nbias.data < 1, m.nbias.data, torch.full_like(m.nbias.data, 1.))
                            m.bNbits -= 1
                            if m.bNbits==1:
                                break
                        else:
                            break
                    # Remove LSB                    
                    weight = m.pbias.data.cpu().numpy()-m.nbias.data.cpu().numpy()
                    total_weight = np.prod(weight.shape)/m.bNbits
                    nonz_weight = [np.count_nonzero(weight[...,i])*100 for i in range(m.bNbits)]
                    nonz_weight = nonz_weight/total_weight
                    if m.bNbits>1:
                        N = m.bNbits
                        pweight = m.pbias.data
                        nweight = m.nbias.data
                        for i in range(N):
                            if nonz_weight[N-1-i]<=threshold:
                                m.pbias.data = pweight[...,0:N-1-i]
                                m.nbias.data = nweight[...,0:N-1-i]
                                m.bNbits -= 1
                                m.biasscale.data = m.biasscale.data*2
                                if m.bNbits==1:
                                    break
                            else:
                                break
                    # Reset exps
                    N = m.bNbits 
                    ex = np.arange(N-1, -1, -1)
                    m.bexps = torch.Tensor((2**ex)/(2**(N)-1)).float()
                    m.biasscale.data = m.biasscale.data*(2**(N)-1)/(2**(N0)-1)
                    ## Match the shape of grad to data
                    if m.pbias.grad is not None:
                        m.pbias.grad.data = m.pbias.grad.data[...,0:N]
                        m.nbias.grad.data = m.nbias.grad.data[...,0:N]
                if m.pbias is not None:
                    Nbit_dict[name] = [m.Nbits, m.bNbits]
                else:
                    Nbit_dict[name] = [m.Nbits, 0]
        return Nbit_dict

def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

if __name__ == '__main__':
    import os
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transform
    from torch.utils.tensorboard import SummaryWriter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCH = 600
    pre_epoch = 0
    BATCH_SIZE = 512
    LR = 0.1
    gamma = 0.1
    LMBDA = 1e-8
    Nbit = 4
    save_dir = 'C:/Users/102/Documents/GitHub/BSQ-CS/train_result/0803/ep600-lr01'
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    writer = SummaryWriter(save_dir)
            
    iters_per_reset = EPOCH-1
    temp_increase = 200**(1./iters_per_reset)

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    #labels in CIFAR10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #define ResNet20
    model = ResNet(num_classes=10,Nbits=Nbit, bin=True).to(device)
    print(model)

    #define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH, eta_min=0, last_epoch=-1)

    #train
    best_acc = 0
    for epoch in range(pre_epoch, EPOCH):
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
                total_ele += t
                ones += o
            ratio_one = ones/total_ele
            writer.add_scalar('Ratio of ones in mask', ratio_one, epoch)

            entries_sum = sum(m.sum() for m in masks)
            loss = criterion(outputs, labels)  + LMBDA * entries_sum # L1 Reg on mask
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
        best_model_path = os.path.join(*[save_dir, 'model_best.pt'])
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
    print('Train has finished, weight and log saved at', save_dir)
    
    