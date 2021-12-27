import torch
import torch.nn as nn


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
        return [self.conv1, self.bn1, self.conv2, self.bn2]

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

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is None:
            x = x1
        else:
            x = torch.cat([x2, x1], dim=1)
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
        return [self.conv1, self.conv2]

    def forward(self, x):
        x = self.mp(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def layers_list(self):
        return [self.conv]

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)


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
        x = x.reshape((1, x.shape[2], x.shape[0], x.shape[1]))
        x = torch.tensor(x, dtype=torch.float32)

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
        target = target.reshape((1, 1, target.shape[0], target.shape[1]))
        target = torch.tensor(target, dtype=torch.float32)

        loss = self.dice_loss(prediction, target)
        print("Loss: {}".format(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def dice_loss(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
