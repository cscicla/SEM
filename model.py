import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_PV(nn.Module):
    def __init__(self):
        super(UNet_PV, self).__init__()
        self.enc1 = nn.LazyConv2d(64, 3, 1, 1)
        self.enc1b = nn.LazyConv2d(64, 3, 1, 1)
        self.enc2 = nn.LazyConv2d(128, 3, 1, 1)
        self.enc2b = nn.LazyConv2d(128, 3, 1, 1)
        self.enc3 = nn.LazyConv2d(256, 3, 1, 1)
        self.enc3b = nn.LazyConv2d(256, 3, 1, 1)
        self.enc4 = nn.LazyConv2d(512, 3, 1, 1)
        self.enc4b = nn.LazyConv2d(512, 3, 1, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.enc5 = nn.LazyConv2d(1024, 3, 1, 1)
        self.enc5b = nn.LazyConv2d(1024, 3, 1, 1)

        self.dec1 = nn.LazyConvTranspose2d(512, 2, 2, 0)
        self.dec1b = nn.LazyConvTranspose2d(512, 2, 2, 0)
        self.dec2 = nn.LazyConvTranspose2d(256, 2, 2, 0)
        self.dec2b = nn.LazyConvTranspose2d(256, 2, 2, 0)
        self.dec3 = nn.LazyConvTranspose2d(128, 2, 2, 0)
        self.dec3b = nn.LazyConvTranspose2d(128, 2, 2, 0)
        self.dec4 = nn.LazyConvTranspose2d(64, 2, 2, 0)
        self.dec4b = nn.LazyConvTranspose2d(64, 2, 2, 0)

        self.conv1a = nn.LazyConv2d(64, 3, 1, 1)
        self.conv1b = nn.LazyConv2d(64, 3, 1, 1)
        self.conv2a = nn.LazyConv2d(128, 3, 1, 1)
        self.conv2b = nn.LazyConv2d(128, 3, 1, 1)
        self.conv3a = nn.LazyConv2d(256, 3, 1, 1)
        self.conv3b = nn.LazyConv2d(256, 3, 1, 1)
        self.conv4a = nn.LazyConv2d(512, 3, 1, 1)
        self.conv4b = nn.LazyConv2d(512, 3, 1, 1)
        self.conv5a = nn.LazyConv2d(1024, 3, 1, 1)
        self.conv5b = nn.LazyConv2d(1024, 3, 1, 1)


        self.out = nn.LazyConv2d(6, 1, 1, 0)

        self.forward = self.build_unet_model


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def double_conv_block1(self, x):
        # Conv2D then ReLU activation
        x = F.relu(self.enc1(x))
        # Conv2D then ReLU activation
        x = F.relu(self.enc1b(x))
        return x
    def double_conv_block2(self, x):
        # Conv2D then ReLU activation
        x = F.relu(self.enc2(x))
        # Conv2D then ReLU activation
        x = F.relu(self.enc2b(x))
        return x
    def double_conv_block3(self, x):
        # Conv2D then ReLU activation
        x = F.relu(self.enc3(x))
        # Conv2D then ReLU activation
        x = F.relu(self.enc3b(x))
        return x
    def double_conv_block4(self, x):
        # Conv2D then ReLU activation
        # x = self.H_pad(x)
        # x = self.W_pad(x)
        x = F.relu(self.enc4(x))
        # Conv2D then ReLU activation
        x = F.relu(self.enc4b(x))
        return x

    def double_conv_block5(self, x):
        # Conv2D then ReLU activation
        x = F.relu(self.enc5(x))
        # Conv2D then ReLU activation
        x = F.relu(self.enc5b(x))
        return x

    def up_double_conv_block1(self, x):
        # Conv2D then ReLU activation
        x = F.relu(self.conv5a(x))
        # Conv2D then ReLU activation
        x = F.relu(self.conv5b(x))
        return x
    def up_double_conv_block2(self, x):
        # Conv2D then ReLU activation
        x = F.relu(self.conv4a(x))
        # Conv2D then ReLU activation
        x = F.relu(self.conv4b(x))
        return x
    def up_double_conv_block3(self, x):
        # Conv2D then ReLU activation
        x = F.relu(self.conv3a(x))
        # Conv2D then ReLU activation
        x = F.relu(self.conv3b(x))
        return x
    def up_double_conv_block4(self, x):
        # Conv2D then ReLU activation
        x = F.relu(self.conv2a(x))
        # Conv2D then ReLU activation
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

    def conv_block1(self, x):
        self.dec1(x)

    def upsample_block1(self, x, conv_features):
        # upsample
        x = self.dec1(x)
        # concatenate
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                  diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, conv_features], dim=1)
        # dropout
        x = self.dropout(x)
        # Conv2D twice with ReLU activation
        x = self.up_double_conv_block1(x)
        return x
    def upsample_block2(self, x, conv_features):
        # upsample
        x = self.dec2(x)
        # concatenate
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                  diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, conv_features], dim=1)
        # dropout
        x = self.dropout(x)
        # Conv2D twice with ReLU activation
        x = self.up_double_conv_block2(x)
        return x
    def upsample_block3(self, x, conv_features):
        # upsample
        x = self.dec3(x)
        # concatenate
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                  diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, conv_features], dim=1)
        # dropout
        x = self.dropout(x)
        # Conv2D twice with ReLU activation
        x = self.up_double_conv_block3(x)
        return x
    def upsample_block4(self, x, conv_features):
        # upsample
        x = self.dec4(x)
        # concatenate
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                  diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, conv_features], dim=1)
        # dropout
        x = self.dropout(x)
        # Conv2D twice with ReLU activation
        x = self.up_double_conv_block4(x)
        return x

    def bottleneck_conv(self, x):
        # Conv2D then ReLU activation
        x = F.relu(self.enc5(x))
        # Conv2D then ReLU activation
        x = F.relu(self.enc5b(x))
        return x

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