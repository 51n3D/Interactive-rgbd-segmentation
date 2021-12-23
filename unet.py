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

    def forward(self, x1, x2=None, x3=None, x4=None, x5=None):
        x1 = self.up(x1)
        if x2 is None and x3 is None:
            x = x1
        elif x3 is None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = torch.cat([x2, x3, x4, x5, x1], dim=1)
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
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def layers_list(self):
        return [self.conv]

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.rgb_down1 = down(3, 32, pool=False)
        self.rgb_down2 = down(32, 64)
        self.rgb_down3 = down(64, 128)
        self.rgb_down4 = down(128, 256)

        self.fneg_interactions_down1 = down(1, 32, pool=False)
        self.fneg_interactions_down2 = down(32, 64)
        self.fneg_interactions_down3 = down(64, 128)
        self.fneg_interactions_down4 = down(128, 256)

        self.fpos_interactions_down1 = down(1, 32, pool=False)
        self.fpos_interactions_down2 = down(32, 64)
        self.fpos_interactions_down3 = down(64, 128)
        self.fpos_interactions_down4 = down(128, 256)

        self.disparity_down1 = down(1, 32, pool=False)
        self.disparity_down2 = down(32, 64)
        self.disparity_down3 = down(64, 128)
        self.disparity_down4 = down(128, 256)

        self.mid = mid(1024, 2048, 1024)
        self.up1 = up(1024, 256, 1024)
        self.up2 = up(768, 128, 256)
        self.up3 = up(384, 64, 128)
        self.up4 = up(192, 64, 64)

        self.outc = outconv(64, n_classes)

    def forward(self, x):
        image, fneg_interactions, fpos_interactions, disparity = x
        rgb = torch.tensor(image, dtype=torch.float32)
        fneg_interactions = torch.tensor(fneg_interactions, dtype=torch.float32)
        fpos_interactions = torch.tensor(fpos_interactions, dtype=torch.float32)
        disparity = torch.tensor(disparity, dtype=torch.float32)

        rgb1 = self.rgb_down1(rgb)
        rgb2 = self.rgb_down2(rgb1)
        rgb3 = self.rgb_down3(rgb2)
        rgb4 = self.rgb_down4(rgb3)

        fneg_interactions_1 = self.fneg_interactions_down1(fneg_interactions)
        fneg_interactions_2 = self.fneg_interactions_down2(fneg_interactions_1)
        fneg_interactions_3 = self.fneg_interactions_down3(fneg_interactions_2)
        fneg_interactions_4 = self.fneg_interactions_down4(fneg_interactions_3)

        fpos_interactions_1 = self.fneg_interactions_down1(fpos_interactions)
        fpos_interactions_2 = self.fneg_interactions_down2(fpos_interactions_1)
        fpos_interactions_3 = self.fneg_interactions_down3(fpos_interactions_2)
        fpos_interactions_4 = self.fneg_interactions_down4(fpos_interactions_3)

        disparity1 = self.disparity_down1(disparity)
        disparity2 = self.disparity_down2(disparity1)
        disparity3 = self.disparity_down3(disparity2)
        disparity4 = self.disparity_down4(disparity3)

        x5 = self.mid(torch.cat([rgb4, fneg_interactions_4, fpos_interactions_4, disparity4], dim=1))
        x = self.up1(x5)
        x = self.up2(x, rgb3, fneg_interactions_3, fpos_interactions_3, disparity3)
        x = self.up3(x, rgb2, fneg_interactions_2, fpos_interactions_2, disparity2)
        x = self.up4(x, rgb1, fneg_interactions_1, fpos_interactions_1, disparity1)
        x = self.outc(x)

        return x
