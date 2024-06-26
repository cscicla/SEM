import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_PV(nn.Module):
    def __init__(self):
        super(UNet_PV, self).__init__()
        ######################## DEFINE THE LAYERS ########################
        self.dropout = nn.Dropout(p=0.3)
        # encoder layers (convolution)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.enc1 = nn.Conv2d(64, 3, kernel_size=1, padding=1)
        self.enc1b = nn.Conv2d(64, 3, kernel_size=1, padding=1)
        self.enc2 = nn.Conv2d(128, 3, kernel_size=1, padding=1)
        self.enc2b = nn.Conv2d(128, 3, kernel_size=1, padding=1)
        self.enc3 = nn.Conv2d(256, 3, kernel_size=1, padding=1)
        self.enc3b = nn.Conv2d(256, 3, kernel_size=1, padding=1)
        self.enc4 = nn.Conv2d(512, 3, kernel_size=1, padding=1)
        self.enc4b = nn.Conv2d(512, 3, kernel_size=1, padding=1)

        # bottleneck
        self.enc5 = nn.Conv2d(1024, 3, kernel_size=1, padding=1)
        self.enc5b = nn.Conv2d(1024, 3, kernel_size=1, padding=1)

        # decoder layers (deconvolution)
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1b = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2b = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3b = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec4b = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # convolution (3x3)
        self.conv1a = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.conv2a = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        self.conv3a = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.conv4a = nn.Conv2d(512, 3, kernel_size=3, padding=1)
        self.conv4b = nn.Conv2d(512, 3, kernel_size=3, padding=1)
        self.conv5a = nn.Conv2d(1024, 3, kernel_size=3, padding=1)
        self.conv5b = nn.Conv2d(1024, 3, kernel_size=3, padding=1)

        # output map (6 classes)
        self.out = nn.Conv2d(6, 1, kernel_size=1, padding=0)

    def double_conv_block1(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc1b(x))
        return x
    
    def double_conv_block2(self, x):
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc2b(x))
        return x
    
    def double_conv_block3(self, x):
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc3b(x))
        return x
    
    def double_conv_block4(self, x):
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc4b(x))
        return x
    
    def double_conv_block5(self, x):
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc5b(x))
        return x
    
    def up_double_conv_block1(self, x):
        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        return x
    
    def up_double_conv_block2(self, x):
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        return x
    
    def up_double_conv_block3(self, x):
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        return x
    
    def up_double_conv_block4(self, x):
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        return x

    def downsample_block1(self, x):
        f = self.double_conv_block1(x)
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p
    
    def downsample_block2(self, x):
        f = self.double_conv_block2(x)
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p
    
    def downsample_block3(self, x):
        f = self.double_conv_block3(x)
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p
    
    def downsample_block4(self, x):
        f = self.double_conv_block4(x)
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p

    def bottleneck_conv(self, x):
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc5b(x))
        return x

    def upsample_block1(self, x, conv_features):
        x = self.dec1(x)
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv_features], dim=1)
        x = self.dropout(x)
        x = self.up_double_conv_block1(x)
        return x
    
    def upsample_block2(self, x, conv_features):
        x = self.dec2(x)
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv_features], dim=1)
        x = self.dropout(x)
        x = self.up_double_conv_block2(x)
        return x
    
    def upsample_block3(self, x, conv_features):
        x = self.dec3(x)
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv_features], dim=1)
        x = self.dropout(x)
        x = self.up_double_conv_block3(x)
        return x
    
    def upsample_block4(self, x, conv_features):
        x = self.dec4(x)
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX - diffX // 2,
                  diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, conv_features], dim=1)
        x = self.dropout(x)
        x = self.up_double_conv_block4(x)
        return x

    ########################### PUT IT ALL TOGETHER ###########################
    def build_unet_model(self, input):
        # inputs
        inputs = input
        # encoder: contracting path - downsample
        # 1 - downsample
        f1, p1 = self.downsample_block1(inputs)
        # 2 - downsample
        f2, p2 = self.downsample_block2(p1)
        # 3 - downsample
        f3, p3 = self.downsample_block3(p2)
        # 4 - downsample
        f4, p4 = self.downsample_block4(p3)
        # 5 - bottleneck
        bottleneck = self.double_conv_block5(p4)
        # decoder: expanding path - upsample
        # 6 - upsample
        u6 = self.upsample_block1(bottleneck, f4)
        # 7 - upsample
        u7 = self.upsample_block2(u6, f3)
        # 8 - upsample
        u8 = self.upsample_block3(u7, f2)
        # 9 - upsample
        u9 = self.upsample_block4(u8, f1)
        # outputs
        outputs = self.out(u9)

        return outputs