import torch
import torch.nn as nn
from logger import log


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(double_conv, self).__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def layers_list(self):
        return [self.conv1,self.bn1,self.conv2,self.bn2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class down(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super(down, self).__init__()
        self.pool = pool
        if pool:
            self.mp = nn.MaxPool2d(2)
        self.conv = double_conv(in_ch, out_ch)

    def layers_list(self):
        return self.conv.layers_list()

    def forward(self, x):
        if self.pool:
            x = self.mp(x)
        x = self.conv(x)
        return x
    
class up(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch):
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = double_conv(in_ch, out_ch, mid_ch=mid_ch)

    def layers_list(self):
        return self.conv.layers_list()

    def forward(self, x1, x2=None, x3=None):
        x1 = self.up(x1)
        if x2 is None and x3 is None:
            x = x1
        elif x3 is None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = torch.cat([x2, x3, x1], dim=1)
        x = self.conv(x)
        return x

class mid(nn.Module):
    def __init__(self, in_ch, out_ch, small_ch=None):
        super(mid, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        if small_ch is None:
            self.conv2 = nn.Conv2d(out_ch, in_ch, 3, padding=1)
        else:
            self.conv2 = nn.Conv2d(out_ch, small_ch, 3, padding=1)

    def layers_list(self):
        return [self.conv1,self.conv2]

    def forward(self, x):
        x = self.mp(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.sigm = nn.Sigmoid()

    def layers_list(self):
        return [self.conv]

    def forward(self, x):
        x = self.conv(x)
        return self.sigm(x)

class UNet(nn.Module):
    def __init__(self, n_classes=1, base=10):
        super(UNet, self).__init__()
        self.down1 = down(6, 2**base, pool=False)
        self.down2 = down(2**base, 2**(base+1))
        self.down3 = down(2**(base+1), 2**(base+2))
        self.down4 = down(2**(base+2), 2**(base+3))
        self.mid = mid(2**(base+3), 2**(base+4))
        self.up1 = up(2**(base+4), 2**(base+2), 2**(base+3))
        self.up2 = up(2**(base+3), 2**(base+1), 2**(base+2))
        self.up3 = up(2**(base+2), 2**base, 2**(base+1))
        self.up4 = up(2**(base+1), 2**base, 2**base)
        self.outc = outconv(2**base, n_classes)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.mid(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def backpropagation(self, prediction, target, optimizer):
        loss = self.dice_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def dice_loss(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class UNetRGBD(nn.Module):
    def __init__(self, n_classes=1, base=10):
        super(UNetRGBD, self).__init__()
        self.rgb_down1 = down(5, 2**base, pool=False)
        self.rgb_down2 = down(2**base, 2**(base + 1))
        self.rgb_down3 = down(2**(base + 1), 2**(base + 2))
        self.rgb_down4 = down(2**(base + 2), 2**(base + 3))
        self.depth_down1 = down(1, 2**base, pool=False)
        self.depth_down2 = down(2**base, 2**(base + 1))
        self.depth_down3 = down(2**(base + 1), 2**(base + 2))
        self.depth_down4 = down(2**(base + 2), 2**(base + 3))
        self.mid  = mid(2**(base + 4), 2**(base + 5), 2**(base + 4))
        self.up1 = up(2**(base + 4), 2**(base + 3), 2**(base + 4))
        self.up2 = up(2**(base + 4), 2**(base + 2), 2**(base + 3))
        self.up3 = up(2**(base + 3), 2**(base + 1), 2**(base + 2))
        self.up4 = up(2**(base + 2), 2**(base + 1), 2**(base + 1))
        self.outc = outconv(2**(base + 1), n_classes)

    def forward(self, x):
        rgb, depth = x
        rgb1 = self.rgb_down1(rgb)
        rgb2 = self.rgb_down2(rgb1)
        rgb3 = self.rgb_down3(rgb2)
        rgb4 = self.rgb_down4(rgb3)
        
        depth1 = self.depth_down1(depth)
        depth2 = self.depth_down2(depth1)
        depth3 = self.depth_down3(depth2)
        depth4 = self.depth_down4(depth3)

        x5 = self.mid(torch.cat([rgb4, depth4], dim=1))
        x = self.up1(x5)
        x = self.up2(x, rgb3, depth3)
        x = self.up3(x, rgb2, depth2)
        x = self.up4(x, rgb1, depth1)
        x = self.outc(x)
        return x
    
    def backpropagation(self, prediction, target, optimizer):
        loss = self.dice_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def dice_loss(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice