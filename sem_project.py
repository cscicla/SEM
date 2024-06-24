from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from PIL import Image
from matplotlib.pyplot import imsave, imshow

from sklearn.metrics import f1_score

import shutil
from sklearn.model_selection import train_test_split

import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import gc
from pytorch_toolbelt import losses as L

from IPython.display import display
from PIL.ImageOps import grayscale
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import cv2 as cv
import cv2 

# import time
# from torchvision.io import read_image
# from skimage.transform import resize
# # pip install pytorch-toolbelt
# # from google.colab.patches import cv2.imshow
# import json
# from PIL import Image
# import shutil
# from sklearn.metrics import f1_score
# import glob

class UNet_PV(nn.Module):
    def __init__(self):
        super(UNet_PV, self).__init__()
        ######################## DEFINE THE LAYERS ########################
        self.dropout = nn.Dropout(p=0.3)
        # encoder layers (convolution)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.enc1 = nn.LazyConv2d(64, 3, 1, 1)
        self.enc1b = nn.LazyConv2d(64, 3, 1, 1)
        self.enc2 = nn.LazyConv2d(128, 3, 1, 1)
        self.enc2b = nn.LazyConv2d(128, 3, 1, 1)
        self.enc3 = nn.LazyConv2d(256, 3, 1, 1)
        self.enc3b = nn.LazyConv2d(256, 3, 1, 1)
        self.enc4 = nn.LazyConv2d(512, 3, 1, 1)
        self.enc4b = nn.LazyConv2d(512, 3, 1, 1)

        # bottleneck
        self.enc5 = nn.LazyConv2d(1024, 3, 1, 1)
        self.enc5b = nn.LazyConv2d(1024, 3, 1, 1)

        # decoder layers (deconvolution)
        # up-convolution (2x2)
        self.dec1 = nn.LazyConvTranspose2d(512, 2, 2, 0)
        self.dec1b = nn.LazyConvTranspose2d(512, 2, 2, 0)
        self.dec2 = nn.LazyConvTranspose2d(256, 2, 2, 0)
        self.dec2b = nn.LazyConvTranspose2d(256, 2, 2, 0)
        self.dec3 = nn.LazyConvTranspose2d(128, 2, 2, 0)
        self.dec3b = nn.LazyConvTranspose2d(128, 2, 2, 0)
        self.dec4 = nn.LazyConvTranspose2d(64, 2, 2, 0)
        self.dec4b = nn.LazyConvTranspose2d(64, 2, 2, 0)
        # convolution (3x3)
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

        # output map (6 classes)
        self.out = nn.LazyConv2d(6, 1, 1, 0)
        self.forward = self.build_unet_model

    # essentially just adding relu to all of our conv (enc and conv)
        # (the dec is convTranspose)
    def double_conv_block1(self, x):
        # Conv2D then ReLU activation
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

    # Now that we have defined our layers (with ReLu), we can build our blocks
    #################### Contracting path (Encoder) ####################
    '''
    * captures contextual information and reduces spatial resolution of the input
    * downsampling: converting a high resolution image to a low resolution image
          * helps the model better understand "WHAT" is present in the image, but it loses the information of "WHERE" it is present
            the pooling (and convolution) operation both reduce the size of the image
          # identifies relevant info
    * structure: double convolution, max pooling, dropout
    # double the number of filters in each layer
    '''
    def downsample_block1(self, x):
        f = self.double_conv_block1(x) # f: feature map after the double conv block
        p = self.max_pool(f)           # p: downsampled feature map
            # reduces the spatial dimenstions but not the depth
        p = self.dropout(p)
            # regularization technique that randomly "drops" some weights in training to prevent overfitting/ boost generalization
        return f, p
            # f: provides detailed, high-res features
                  # later passed (through skip connections) to recoder to maintain detail and improve accuracy
            # p: allows network to learn more abstract and general features while preventing overfitting
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

    #################### Bottleneck ####################
    def bottleneck_conv(self, x):
      x = F.relu(self.enc5(x))
      x = F.relu(self.enc5b(x))
      return x

    #################### Expansive path (Decoder) ####################
    '''
    * takes the bottleneck feature map and converts it back to size of original input image
    * upsampling: converting the low resolution image to a high resolution image
          * Since downsampling results in us losing the "WHERE" information, we recover the "WHERE" information by upsampling
          * methods: bi-linear interpolation, cubic interpolation, nearest neighbor interpolation, unpooling, transposed convolution (preferred)
    '''
    def upsample_block1(self, x, conv_features):
        # ********* UPSAMPLE *********
        x = self.dec1(x)                              # deconvolution/ upconvolution
        # ********* CONCATENATE *********
        # conv_features.size(): gets the size (dimensions) of the tensor
            # the tensor is from the corresponding layer in the encoder
        # x.size: size of x, the upsampled feature map from the prev layer in decoder
        diffY = conv_features.size()[2] - x.size()[2] # difference in height
        diffX = conv_features.size()[3] - x.size()[3] # difference in width

        # pad x to match the dimensions of conv_features (we want to return img back to OG size)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, # pads elements on l and r to match width
                  diffY // 2, diffY - diffY // 2])    # pads elts on top and bottom to match height
        x = torch.cat([x, conv_features], dim=1)      # concatenates x and conv_features
        # ********* DROPOUT *********
        x = self.dropout(x)
        # ********* Conv2D twice with ReLU activation *********
        x = self.up_double_conv_block1(x)             # integrate the high-level features
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
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
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
   
class PVDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir # (NumPy)
        self.image_list = glob.glob(img_dir + '*.png')
        self.image_list.extend(glob.glob(img_dir + '*.jpg'))
        self.image_list.extend(glob.glob(img_dir + '*.tif'))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert("L") # grayscale
        image = image.resize((1024, 768))
        label_name = self.label_dir + os.path.basename(os.path.splitext(img_name)[0]) + '_label.npy'
        label_np = np.load(label_name)
        label = torch.from_numpy(label_np) # convert label to tensor

        if self.transform: # random cropping and flipping to help generalizability
            image = self.transform(image)
            label = label.unsqueeze(0)
            if random.random() > 0.5:
              image = transforms.functional.hflip(image)
              label = transforms.functional.hflip(label)
            if random.random() > 0.5:
              i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 384))
              image = transforms.functional.crop(image, i, j, h, w)
              label = transforms.functional.crop(label, i, j, h, w)
            image = transforms.functional.resize(image, size=(768, 1024))
            label = transforms.functional.resize(label, size=(768, 1024))
        return image, label
    
def get_mask(all_mask):
    '''
    convert the mask to color based on class
    '''
    # set classes = colors
    unlabeled = np.array([0, 0, 0])
    silver = np.array([128, 64, 128])
    glass = np.array([189, 201, 18])
    silicon = np.array([0, 102, 0])
    void = np.array([0, 76, 130])
    interfacialvoid = np.array([36, 18, 201])
    height = all_mask.shape[0]
    width = all_mask.shape[1]

    #initialize empty mask
    final_mask = np.zeros((height, width, 3))

    # for each pixel, give it a color
    for i in range(len(all_mask)):
        for j in range(len(all_mask[i])):
            if all_mask[i][j] == 0: # was not changed from the initalized zero
                final_mask[i][j] = unlabeled
            elif all_mask[i][j] == 1:
                final_mask[i][j] = silver
            elif all_mask[i][j] == 2:
                final_mask[i][j] = glass
            elif all_mask[i][j] == 3:
                final_mask[i][j] = silicon
            elif all_mask[i][j] == 4:
                final_mask[i][j] = void
            elif all_mask[i][j] == 5:
                final_mask[i][j] = interfacialvoid
    # convert it to standard image format
    return final_mask.astype(np.uint8)

def save_images(data, output, target, epoch, testflag):
    '''
        save the original image, ground_truth/ target mask, and predicted mask
    '''
    # save_folder = "/home/crcvreu.student10/SEM_python/masks/"
    save_folder = "/home/crcvreu.student10/SEM/masks/"
    
    # Convert tensors to numpy arrays
    for i in range(FLAGS.batch_size):
        # data[i]: ith image in the batch
        # .cpu(): converting tensors to NumPy arrays require the data to be on the CPU
                # numpy does not support gpu tensors
        # .numpy(): convert the tensor to NumPy
        #.squeeze(): remove unecessary 1D
        data_np = data[i].cpu().numpy().squeeze()       # convert input images
        output_np = output[i].cpu().numpy().squeeze()   # convert predicted masks
        target_np = target[i].cpu().numpy().squeeze()   # convert ground truth

        # get_mask(): convert the mask to color
        pred_mask = get_mask(output_np)
        label_mask = get_mask(target_np)

        data_np = data_np * 255
        data_np = data_np.astype(np.uint8)

        # Save images as PNG files with epoch number in the filename
        imsave(os.path.join(save_folder, f'epoch_{epoch}_predicted.png'), pred_mask)

        # save original and target images at beginning of run
        if epoch == 1:
            imsave(os.path.join(save_folder, 'original.png'), data_np)
            imsave(os.path.join(save_folder, 'target.png'), label_mask)

def multi_acc(pred, label):
    '''
    pred: predicted labels
    label: true labels
    gets the number of correctly predicted labels
    '''
    corrects = torch.eq(pred, label).float()  # tensor where elts are...1 if pred=label, 0 if pred not= label
    acc = corrects.sum() / corrects.numel()   # correct/ total
    acc = acc * 100
    return acc

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size, file, class_weights):
    '''
    Trains the model for an epoch and optimizes it.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    losses = []     # loss of each image in the epoch
    accs = []       # accuracy of each image in the epoch
    f1_scores_class = []  # f1_score for each class of each image in the epoch
    f1_scores = []  # weighted f1_score of each image in the epoch

    # Set model to train mode before each epoch
    model.train()

    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample # data: input, target: ground truth

        # Push data/label to correct device
        data = data.to(device)
        target = target.to(device)

        # Reset optimizer gradients. Avoids gradient accumulation
              # gradient accumulation is useful in recurrent neural networds (RNNs) but
              # may lead to incorrect gradient values for CNNs (Unet)
        optimizer.zero_grad()

        #forward pass
        # after softmax this provides the proability of each element being in a certain class
        output = model(data)
        output = F.softmax(output, dim=1)
              # eg. output after softmax:
              #     tensor([[0.2153, 0.0456, 0.7391],
              #             [0.1042, 0.6682, 0.2276],
              #             [0.4914, 0.3333, 0.1752]])

        # computes how different the output and target are
        loss = criterion(y_pred=output, y_true=target.long())

        # backward pass: Computes gradient based on final loss
              # how much each neuron and weght contributed to the error
        loss.backward()
        output = torch.argmax(output, dim=1)
              # eg. output after finding which had class had the highest probability for each elt
              #     output = tensor([2, 1, 0])

        losses.append(loss.item())      # store loss
        acc = multi_acc(output, target).cpu()
        accs.append(acc)                # calculate and store accuracy

        # GET F1 SCORES
        # flatten tensors and convert to numpy arrays for f1_score calculation
        target = target.long().cpu().view(-1)
        output = output.view(-1)

        f1 = f1_score(y_true=target.numpy(), y_pred=output.cpu().numpy(), average=None)
              # eg. f1 = array([0.62111801, 0.33333333, 0.26666667, 0.13333333, 0.26666667, 0.13333333,])
        f1_scores_class.append(f1)

        f1_weighted = f1_score(y_true=target.numpy(), y_pred=output.cpu().numpy(), average='weighted')
        f1_scores.append(f1_weighted)

        optimizer.step()                # Optimize model parameters based on lr and gradient
        gc.collect()                    # free memory blocks not in use anymore

    train_loss = float(np.mean(losses)) # get average loss of the batch
    train_acc = np.mean(accs)

    # if some classes are not present in the images, pad them with 0s
    for i in range(len(f1_scores_class)):
        if len(f1_scores_class[i]) < 6: #6 = number of classes
            f1_scores_class[i] = np.pad(f1_scores_class[i], (0, 6 - len(f1_scores_class[i])), mode='constant')

    train_f1_class = np.mean(f1_scores_class, axis=0)
    train_f1 = np.mean(f1_scores)

    print("Epoch: ", epoch)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    print('Average Train Accuracy: {:.4f}'.format(train_acc))
    # print('Train F1 Scores Per Class: ', f1_scores_class)
    print('Train F1 Score: ', train_f1)
    file.write('\n' + "Epoch: " + str(epoch) + '\n')
    file.write('Train set: Average loss: {:.4f}\n'.format(train_loss))
    # file.write('Train F1 Scores Per Class: {}\n'.format(f1_scores_class))
    file.write('Train F1 Score: {}\n'.format(train_f1))
    return train_loss, train_acc, train_f1_class, train_f1

def test(model, device, test_loader, file, epoch, class_weights):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''

    # Set model to eval mode to notify all layers.
    model.eval()

    losses = []     # loss of each image in the epoch
    accs = []       # accuracy of each image in the epoch
    f1_scores_class = []  # f1_score for each class of each image in the epoch
    f1_scores = []  # weighted f1_score of each image in the epoch

    with torch.no_grad():     # disable the gradient computation and backpropoagation used in training
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            # Forward pass: get model's prediction
            output = model(data)
            output = F.softmax(output, dim=1)

            # Compute loss: how dissimlar predictedmask  was from ground_truth mask
            criterion=L.DiceLoss('multiclass')
            loss = criterion(y_pred=output, y_true=target.long())
            output = torch.argmax(output, dim=1)

            # store f1, loss, and acc
            losses.append(loss.item())
            acc = multi_acc(output, target).cpu()
            accs.append(acc)

            # GET F1 SCORES
            # flatten tensors and convert to numpy arrays for f1_score calculation
            target1 = target.long().cpu().view(-1)
            output1 = output.view(-1)

            f1 = f1_score(y_true=target1.numpy(), y_pred=output1.cpu().numpy(), average=None)
                  # eg. f1 = array([0.62111801, 0.33333333, 0.26666667, 0.13333333, 0.26666667, 0.13333333,])
            f1_scores_class.append(f1)            # store f1

            f1_weighted = f1_score(y_true=target1.numpy(), y_pred=output1.cpu().numpy(), average='weighted')
            f1_scores.append(f1_weighted)         # store weighted f1


    save_images(data, output, target, epoch, True)  # save predicted masks
    test_loss = float(np.mean(losses))
    test_acc = np.mean(accs)

    # if some classes are not present in the images, pad them with 0s
    for i in range(len(f1_scores_class)):
        if len(f1_scores_class[i]) < 6: #6 = number of classes
            f1_scores_class[i] = np.pad(f1_scores_class[i], (0, 6 - len(f1_scores_class[i])), mode='constant')

    test_f1_class = np.mean(f1_scores_class, axis=0)
    test_f1 = np.mean(f1_scores)

    print('Test set: Average loss: {:.4f}'.format(test_loss))
    print('Average Test Accuracy: {:.4f}'.format(test_acc))
    # print('Test F1 Scores Per Class: ', f1_scores_class)
    print('Test F1 Score: ', test_f1)
    file.write('\n' + "Epoch: " + str(epoch) + '\n')
    file.write('Test set: Average loss: {:.4f}\n'.format(float(np.mean(losses))))
    # file.write('Test F1 Scores Per Class: {}\n'.format(f1_scores_class))
    file.write('Test F1 Score: {}\n'.format(test_f1))
    return test_loss, test_acc, test_f1_class, test_f1

def split_data(image_dir, label_dir, train_image_dir, train_label_dir, test_image_dir, test_label_dir, test_size=0.2):
    images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    labels = [f for f in os.listdir(label_dir) if f.endswith('.npy')]

    image_bases = {os.path.splitext(f)[0] for f in images}
    label_bases = {os.path.splitext(f)[0].replace('_label', '') for f in labels}
    common_bases = list(image_bases & label_bases)

    print(f"Found {len(common_bases)} common images and labels.")

    if len(common_bases) == 0:
        raise ValueError("No common images and labels found to split.")

    images = [f"{base}.jpg" if f"{base}.jpg" in images else f"{base}.png" for base in common_bases]
    labels = [f"{base}_label.npy" for base in common_bases]

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    # os.makedirs(train_image_dir, exist_ok=True)
    # os.makedirs(train_label_dir, exist_ok=True)
    # os.makedirs(test_image_dir, exist_ok=True)
    # os.makedirs(test_label_dir, exist_ok=True)

    for image, label in zip(train_images, train_labels):
        shutil.move(os.path.join(image_dir, image), os.path.join(train_image_dir, image))
        shutil.move(os.path.join(label_dir, label), os.path.join(train_label_dir, label))

    for image, label in zip(test_images, test_labels):
        shutil.move(os.path.join(image_dir, image), os.path.join(test_image_dir, image))
        shutil.move(os.path.join(label_dir, label), os.path.join(test_label_dir, label))

    print(f"Moved {len(train_images)} images to training set and {len(test_images)} images to test set.")

def run_main(FLAGS, file):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    # Set proper device based on cuda availability
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    file.write("\n\nTorch device selected: " + str(device) + '\n')
    # Initialize the model and send to device
    model = UNet_PV().to(device)
    save_folder = "/home/crcvreu.student10/SEM/masks"
    # Create the folder if it doesn't exist
    # os.makedirs(save_folder, exist_ok=True)
    checkpoint_folder = "/home/crcvreu.student10/SEM/checkpoints"
    # os.makedirs(checkpoint_folder, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-5)
            # higher weight decay = lower overfitting

    # Create transformations to apply to each data sample
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
         ])
    criterion=L.DiceLoss('multiclass')
    class_weights = torch.Tensor([1, 1, 1.5, 1, 1.5, 1.5]).to(device)

    #################### LOAD DATA ####################
    image_dir = '/home/crcvreu.student10/SEM/data/'
    label_dir = '/home/crcvreu.student10/SEM/output'
    train_image_dir = '/home/crcvreu.student10/SEM/train/data/'
    train_label_dir = '/home/crcvreu.student10/SEM/train/output/'
    test_image_dir = '/home/crcvreu.student10/SEM/test/data/'
    test_label_dir = '/home/crcvreu.student10/SEM/test/output/'

    # Split data into training and test sets
    # split_data(image_dir, label_dir, train_image_dir, train_label_dir, test_image_dir, test_label_dir)

    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = PVDataset(train_image_dir, train_label_dir, transform=transform)
    dataset2 = PVDataset(test_image_dir, test_label_dir, transform=transform)

    train_loader = DataLoader(dataset1, batch_size=FLAGS.batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset2, batch_size=FLAGS.batch_size,
                             shuffle=False, num_workers=2)

    best_accuracy = 0.0
    checkpoint_path = '/home/crcvreu.student10/SEM/checkpoints/model_filled2.pth'
    train_losses = []           # store losses per epoch
    train_accs = []             # store accuracy per epoch
    train_f1_class_scores = []  # store f1_score of each class in each image per epoch
    train_f1_scores = []        # store f1_score of each class in each image per epoch

    test_losses = []
    test_accs = []
    test_f1_class_scores = []
    test_f1_scores = []

    classes = ["background", "silver", "glass", "silicon", "void", "interfacial void"]
    f1_dict = {"background":[], "silver":[], "glass":[], "silicon":[], "void":[], "interfacial void":[]}

    # Early stopping parameters
    early_stopping_patience = FLAGS.early_stopping_patience
    best_loss = float('inf')
    epochs_wo_improve = 0

    # if the checkpoint file does not exist, initialize a new trainign process starting from epoch1
    for epoch in range(1, FLAGS.num_epochs + 1):
        # train
        train_loss, train_acc, train_f1_class, train_f1 = train(model, device, train_loader,
                                      optimizer, criterion, epoch,
                                      FLAGS.batch_size, file, class_weights)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1_class_scores.append(train_f1_class)
        train_f1_scores.append(train_f1)

        # test
        test_loss, test_acc, test_f1_class, test_f1 = test(model, device, test_loader,
                                    file, epoch, class_weights)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        test_f1_class_scores.append(test_f1_class)
        test_f1_scores.append(test_f1)

        # Check for early stopping
        if early_stopping_patience < 1:
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'train_accs': train_accs,
                'test_accs': test_accs,
                'train_f1_class_scores': train_f1_class,
                'test_f1_class_scores': test_f1_class,
                'train_f1_scores': train_f1_scores,
                'test_f1_scores': test_f1_scores
            }, checkpoint_path)
        else:
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_wo_improve = 0
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_accs': train_accs,
                    'test_accs': test_accs,
                    'train_f1_class_scores': train_f1_class,
                    'test_f1_class_scores': test_f1_class,
                    'train_f1_scores': train_f1_scores,
                    'test_f1_scores': test_f1_scores
                }, checkpoint_path)
            else:
                epochs_wo_improve += 1
                if epochs_wo_improve >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

    torch.save(model, '/home/crcvreu.student10/SEM/final_model_filled2.pth')
    print("Training and evaluation finished")

    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Train and Test Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/crcvreu.student10/SEM/graphs/Train_Test_loss_model.jpg")
    plt.clf()
    # plt.show()

    plt.plot(train_accs, label='train accuracy')
    plt.plot(test_accs, label='test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("/home/crcvreu.student10/SEM/graphs/Train_Test_acc_model.jpg")
    plt.clf()
    plt.show()

    num_classes = len(train_f1_class_scores[0]) # 6
    epochs = range(1, len(train_f1_class_scores) + 1)
    for class_idx in range(num_classes):
        class_scores = [epoch_scores[class_idx] for epoch_scores in train_f1_class_scores]
        plt.plot(epochs, train_f1_class_scores, label=classes[class_idx])
    # plt.plot(train_f1_class_scores)
    plt.xlabel('Epochs')
    plt.ylabel('Train F1')
    plt.title('Train F1 per class')
    plt.legend(classes, loc='upper left')
    plt.grid(True)
    plt.savefig("/home/crcvreu.student10/SEM/graphs/Train_f1_per_class_model.jpg")
    plt.clf()
    # plt.show()

    # plt.plot(test_f1_class_scores)
    for class_idx in range(num_classes):
        class_scores = [epoch_scores[class_idx] for epoch_scores in test_f1_class_scores]
        plt.plot(epochs, test_f1_class_scores, label=classes[class_idx])
    plt.xlabel('Epochs')
    plt.ylabel('Test F1')
    plt.title('Test F1 per class')
    plt.legend(classes, loc='upper left')
    plt.grid(True)
    plt.savefig("/home/crcvreu.student10/SEM/graphs/Test_f1_per_class_model.jpg")
    plt.clf()
    # plt.show()

    plt.plot(train_f1_scores)
    plt.xlabel('Epochs')
    plt.ylabel('Train F1')
    plt.title('Train F1 Scores')
    plt.grid(True)
    plt.savefig("/home/crcvreu.student10/SEM/graphs/Train_f1_model.jpg")
    plt.clf()
    # plt.show()

    plt.plot(test_f1_scores)
    plt.xlabel('Epochs')
    plt.ylabel('Test F1')
    plt.title('Test F1 Scores')
    plt.grid(True)
    plt.savefig("/home/crcvreu.student10/SEM/graphs/Test_f1_model.jpg")
    plt.clf()
    # plt.show()

    file.write("\nTraining and evaluation finished")

if __name__ == '__main__':
    torch.set_printoptions(threshold=10000)
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=300,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=2,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    parser.add_argument('--early_stopping_patience',
                        type=float, default=0,
                        help= 'Num epochs to wait for lower loss before ending training')
    
    with open('output.txt', 'a') as f:
        # FLAGS = None
        gc.collect()
        FLAGS, unparsed = parser.parse_known_args()

        run_main(FLAGS, f)

print('main done')
# Check if cuda is available
use_cuda = torch.cuda.is_available()
# Set proper device based on cuda availability
device = torch.device("cuda" if use_cuda else "cpu")

model = UNet_PV().to(device)
model = torch.load('/home/crcvreu.student10/SEM/final_model_filled2.pth')
model.eval()

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
 ])

classes = ["background", "silver", "glass", "silicon", "void", "interfacial void"]
f1_dict = {"background":[], "silver":[], "glass":[], "silicon":[], "void":[], "interfacial void":[]}

img_dir = '/home/crcvreu.student10/SEM/train/data/'
for filename in os.listdir(img_dir):
  file_path = Path(img_dir + filename)
  if file_path.suffix == '.png' or file_path.suffix == '.tif' or file_path.suffix == '.jpg':
        # Print OG photo
        print(filename)
        image = Image.open(file_path)
        image = image.resize((1024, 768))
        plt.clf()
        # display(image)

        #print target mask
        mask_dir = "/home/crcvreu.student10/SEM/train/output/"
        label_name = mask_dir + os.path.basename(os.path.splitext(filename)[0]) + '_label.npy'
        label_np = np.load(label_name)
        label = torch.from_numpy(label_np) # convert label to tensor
        target_np = label.cpu().numpy().squeeze()
        label_mask = get_mask(target_np)
        label_mask = cv.resize(cv.cvtColor(label_mask, cv.COLOR_BGR2RGB), (1024, 768))
        # label_mask = cv.resize(label_mask, (1024, 768))
        # cv2.imshow('ground truth mask', label_mask)
        imsave(os.path.join("/home/crcvreu.student10/SEM/masks/target", os.path.splitext(os.path.basename(file_path))[0] + "_target.png"), pred_mask)

        # print predicted mask
        image = grayscale(image)
        image = trans(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        # print('predicted')
        # print("IMAGE DIMS: ", image.shape)
        pred = model(image)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy().squeeze()
        pred_mask = get_mask(pred)
        # pred_mask = cv.cvtColor(pred_mask, cv.COLOR_BGR2RGB)
        # cv2.imshow('predicted mask', pred_mask)
        # Save images as PNG files with epoch number in the filename
        # os.makedirs("/home/crcvreu.student10/SEM/masks/final", exist_ok=True)
        imsave(os.path.join("/home/crcvreu.student10/SEM/masks/final", os.path.splitext(os.path.basename(file_path))[0] + "_final.png"), pred_mask)

        plt.clf()
        label = np.load(Path(mask_dir + "/" + os.path.splitext(os.path.basename(file_path))[0] + "_label.npy"))
        m = MultiLabelBinarizer().fit(label)
        f1 = f1_score(m.transform(label), m.transform(pred), average=None)
        if len(f1) != 6:
            continue
        for i, class_name in enumerate(classes):
            f1_dict[class_name].append(f1[i])
        print(f1_dict)
for class_name in classes:
    print(class_name, np.mean(f1_dict[class_name]))