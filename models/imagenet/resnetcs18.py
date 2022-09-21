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
                     
def conv3x3(in_planes, out_planes, stride=1, Nbits=4, bin=True):
    "3x3 convolution with padding"
    return BitConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, Nbits = Nbits, bin=bin)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, Nbits=6, act_bit=4, bin=True):
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

    def forward(self, x, temp):
        # x, temp = input
        residual = x

        out = self.conv1(x, temp)
        out = self.bn1(out)
        out = self.relu1(out)

        out = STE.apply(out,self.act_bit)

        out = self.conv2(out, temp)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)
        
        out = STE.apply(out,self.act_bit)

        return out

class MaskedNet(nn.Module):
    def __init__(self):
        super(MaskedNet, self).__init__()

    def checkpoint(self):
        for m in self.mask_modules: m.checkpoint()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())      
    def prune(self):
        for m in self.mask_modules: m.prune(self.temp)

class ResStage(nn.Module):
    def __init__(self, in_planes, out_planes, stride, padding, Nbits=4,act_bit=0, bin=True, bias=False):
        super(ResStage, self).__init__()
        downsample = None
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            
        self.block1 = BasicBlock(in_planes, out_planes, stride=stride, downsample=downsample, Nbits=Nbits, act_bit=act_bit, bin=bin)
        self.block2 = BasicBlock(out_planes, out_planes, stride=1, downsample=None, Nbits=Nbits, act_bit=act_bit, bin=bin)
        self.block3 = BasicBlock(out_planes, out_planes, stride=1, downsample=None, Nbits=Nbits, act_bit=act_bit, bin=bin)

    def forward(self, x, temp):
        out = self.block1(x, temp)
        out = self.block2(out, temp)
        out = self.block3(out, temp)
        return out

class ResNet18(MaskedNet):
    def __init__(self, num_classes=10, Nbits=8, act_bit=0, bin=True):
        super(ResNet18, self).__init__()

        self.conv1 = BitConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, Nbits=8, bin=bin)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResStage(64,64,1,1,Nbits,act_bit,bin=bin)
        self.layer2 = ResStage(64,128,2,1,Nbits,act_bit,bin=bin)
        self.layer3 = ResStage(128,256,2,1,Nbits,act_bit,bin=bin)
        self.layer4 = ResStage(256,512,2,1,Nbits,bin=bin)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = BitLinear(512, out_features=num_classes, Nbits=8, bin=bin)

        self.mask_modules = [m for m in self.modules() if type(m) in [BitConv2d, BitLinear]]
        self.temp = 1
        self.temp_s = torch.Tensor(Nbits)

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
        x = self.conv1(x, self.temp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x, self.temp)
        x = self.layer2(x, self.temp)
        x = self.layer3(x, self.temp)
        x = self.layer4(x, self.temp)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x, self.temp)

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
            if isinstance(m,BitLinear) or isinstance(m,BitConv2d):
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

def resnet18(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet18(**kwargs)


    
 