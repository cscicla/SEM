import os
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import random
from torchvision import transforms
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from pytorch_toolbelt import losses as L
from matplotlib.pyplot import imsave, imshow
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import gc
import argparse
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import cv2 as cv
from PIL.ImageOps import grayscale

class PVDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir # (NumPy)
        # self.image_list = glob.glob(img_dir + '*.png')
        # self.image_list.extend(glob.glob(img_dir + '*.jpg'))
        # self.image_list.extend(glob.glob(img_dir + '*.tif'))
        self.image_list = glob.glob(os.path.join(img_dir, '*.png'))
        self.image_list.extend(glob.glob(os.path.join(img_dir, '*.jpg')))


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
            if random.random() > 0.5:
              image = transforms.functional.hflip(image)
              label = transforms.functional.hflip(label)
            if random.random() > 0.5:
              i, j, h, w = transforms.RandomCrop.get_params(
              image, output_size=(698, 364))
              image = transforms.functional.crop(image, i, j, h, w)
              label = transforms.functional.crop(label, i, j, h, w)
        return image, label
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

def multi_acc(pred, label):
    # _, tags = torch.argmax(pred, dim = 1)
    corrects = torch.eq(pred, label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc

def get_mask(all_mask):
    unlabeled = np.array([0, 0, 0])
    silver = np.array([128, 64, 128])
    glass = np.array([189, 201, 18])
    silicon = np.array([0, 102, 0])
    void = np.array([0, 76, 130])
    interfacialvoid = np.array([36, 18, 201])
    height = all_mask.shape[0]
    width = all_mask.shape[1]

    final_mask = np.zeros((height, width, 3))

    for i in range(len(all_mask)):
        for j in range(len(all_mask[i])):
            if all_mask[i][j] == 0:
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

    return final_mask.astype(np.uint8)

def save_images(data, output, target, epoch, testflag):

    save_folder = "/content/drive/MyDrive/SEM_project/masks"
    # Convert tensors to numpy arrays
    # for i in range(FLAGS.batch_size):
    for i in range(1):
        data_np = data[i].cpu().numpy().squeeze()
        output_np = output[i].cpu().numpy().squeeze()
        target_np = target[i].cpu().numpy().squeeze()

        pred_mask = get_mask(output_np)
        label_mask = get_mask(target_np)
        data_np = data_np * 255
        data_np = data_np.astype(np.uint8)

        # Save images as PNG files with epoch number in the filename
        imsave(os.path.join(save_folder, f'epoch_{epoch}_predicted.png'), pred_mask)
        if epoch == 1:
          imsave(os.path.join(save_folder, 'original.png'), data_np)
          imsave(os.path.join(save_folder, 'target.png'), label_mask)

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

    # Set model to train mode before each epoch
    model.train()
    # Empty list to store losses
    losses = []
    # correct = 0
    accs = []
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample

        # Push data/label to correct device
        # data, target = data.to(device), target.to(device)
        data = data.to(device)
        target = target.to(device)

        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()

        # Do forward pass for current set of data
        output = model(data)
        output = F.softmax(output, dim=1)
        criterion=L.DiceLoss('multiclass')
        loss = criterion(y_pred=output, y_true=target.long())
        # Computes gradient based on final loss
        loss.backward()
        output = torch.argmax(output, dim=1)

        # Store loss
        losses.append(loss.item())

        # Optimize model parameters based on learning rate and gradient
        optimizer.step()
        acc = multi_acc(output, target).cpu()
        accs.append(acc)
        gc.collect()

    train_loss = float(np.mean(losses))
    # train_acc = correct / ((batch_idx+1) * batch_size)
    print("Epoch: ", epoch)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    print('Average Train Accuracy: {:.4f}'.format(np.mean(accs)))
    file.write('\n' + "Epoch: " + str(epoch) + '\n')
    file.write('Train set: Average loss: {:.4f}\n'.format(float(np.mean(losses))))
    return train_loss, np.mean(accs)

def test(model, device, test_loader, file, epoch, class_weights):
    '''
    Tests the model.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''

    # Set model to eval mode to notify all layers.
    model.eval()

    losses = []
    accs=[]
    correct = 0
    data_last = 0
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            # Predict for data by doing forward pass
            output = model(data)
            output = F.softmax(output, dim=1)
            criterion=L.DiceLoss('multiclass')
            loss = criterion(y_pred=output, y_true=target.long())
            output = torch.argmax(output, dim=1)
            # Append loss to overall test loss
            losses.append(loss.item())

            # Get predicted index by selecting maximum log-probability
            # pred = output.argmax(dim=1, keepdim=True)
            acc = multi_acc(output, target).cpu()
            accs.append(acc)
    save_images(data, output, target, epoch, True)
    test_loss = float(np.mean(losses))
    
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    print('Average Test Accuracy: {:.4f}'.format(np.mean(accs)))
    return test_loss, np.mean(accs)

def split_data(image_dir, label_dir, train_image_dir, train_label_dir, test_image_dir, test_label_dir, test_size=0.2):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    train_image_dir = Path(train_image_dir)
    train_label_dir = Path(train_label_dir)
    test_image_dir = Path(test_image_dir)
    test_label_dir = Path(test_label_dir)


    images = [f for f in image_dir.iterdir() if f.suffix in {'.png', '.jpg'}]
    labels = [f for f in label_dir.iterdir() if f.suffix == '.npy']

    image_bases = {f.stem for f in images}
    label_bases = {f.stem.replace('_label', '') for f in labels}
    common_bases = list(image_bases & label_bases)

    print(f"Found {len(common_bases)} common images and labels.")

    if len(common_bases) == 0:
        raise ValueError("No common images and labels found to split.")

    images = [image_dir / (f"{base}.jpg") if (image_dir / f"{base}.jpg").exists() else image_dir / f"{base}.png" for base in common_bases]
    labels = [label_dir / f"{base}_label.npy" for base in common_bases]

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    for image, label in zip(train_images, train_labels):
        shutil.move(str(image), str(train_image_dir / image.name))
        shutil.move(str(label), str(train_label_dir / label.name))

    for image, label in zip(test_images, test_labels):
        shutil.move(str(image), str(test_image_dir / image.name))
        shutil.move(str(label), str(test_label_dir / label.name))

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
    save_folder = Path("/Users/claire/Downloads/SEM_project_python/masks")
    # Create the folder if it doesn't exist
    checkpoint_folder = Path("/Users/claire/Downloads/SEM_project_python/checkpoints")
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # Create transformations to apply to each data sample
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
         ])
    criterion = nn.CrossEntropyLoss()
    class_weights = torch.Tensor([1, 1, 1.5, 1, 1.5, 1.5]).to(device)

    #################### LOAD DATA ####################
    image_dir = Path('/Users/claire/Downloads/SEM_project_python/data/')
    label_dir = Path('/Users/claire/Downloads/SEM_project_python/output')
    train_image_dir = Path('/Users/claire/Downloads/SEM_project_python/train/data/')
    train_label_dir = Path('/Users/claire/Downloads/SEM_project_python/train/output/')
    test_image_dir = Path('/Users/claire/Downloads/SEM_project_python/test/data/')
    test_label_dir = Path('/Users/claire/Downloads/SEM_project_python/test/output/')

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

    # best_accuracy = 0.0
    checkpoint_path = '/Users/claire/Downloads/SEM_project_python/checkpoints/model_filled2.pth'
    # checkpoint_path = '/Users/claire/Downloadsclai_project_pythonre/Downloads/SEM_project_python/checkpoints/model_filled2.pth'
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    if not os.path.isfile(checkpoint_path):
        # Run training for n_epochs specified in config
        for epoch in range(1, FLAGS.num_epochs + 1):
            train_loss, train_acc = train(model, device, train_loader,
                                            optimizer, criterion, epoch,
                                            FLAGS.batch_size, file, class_weights)
            test_loss, test_acc = test(model, device, test_loader,
                                        file, epoch, class_weights)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs
            }, checkpoint_path)

    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        train_accs = checkpoint['train_accs']
        test_accs = checkpoint['test_accs']

        for epoch_i in range(epoch+1, FLAGS.num_epochs + 1):
            train_loss, train_acc = train(model, device, train_loader,
                                            optimizer, criterion, epoch_i,
                                            FLAGS.batch_size, file,class_weights)
            test_loss, test_acc = test(model, device, test_loader, file,
                                        epoch_i, class_weights)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs
            }, checkpoint_path)
    torch.save(model, '/Users/claire/Downloads/SEM_project_python/final_model_filled2.pth')
    
    # print("Training and evaluation finished")

    plt.plot(train_losses)
    plt.savefig("/Users/claire/Downloads/SEM_project_python/graphs/Train_loss_model.jpg")
    plt.clf()
    plt.plot(train_accs)
    plt.savefig("/Users/claire/Downloads/SEM_project_python/graphs/Train_acc_model.jpg")
    plt.clf()
    plt.plot(test_losses)
    plt.savefig("/Users/claire/Downloads/SEM_project_python/graphs/Test_loss_model.jpg")
    plt.clf()
    plt.plot(test_accs)
    plt.savefig("/Users/claire/Downloads/SEM_project_python/graphs/Test_acc_model.jpg")
    plt.clf()

    file.write("\nTraining and evaluation finished")


if __name__ == '__main__':
    torch.set_printoptions(threshold=10000)
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--mode',
                        type=int, default=1,
                        help='Select mode between 1-5.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.00005,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=5,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=1,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    with open('output.txt', 'a') as f:
        # FLAGS = None
        gc.collect()
        FLAGS, unparsed = parser.parse_known_args()

        run_main(FLAGS, f)

# Check if cuda is available
use_cuda = torch.cuda.is_available()
# Set proper device based on cuda availability
device = torch.device("cuda" if use_cuda else "cpu")

model = UNet_PV().to(device)
model = torch.load('/Users/claire/Downloads/SEM_project_python/checkpoints/final_model_filled2.pth')
model.eval()

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

classes = ["background", "silver", "glass", "silicon", "void", "interfacial void"]
f1_dict = {"background":[], "silver":[], "glass":[], "silicon":[], "void":[], "interfacial void":[]}
flag = 1
counter = 0

img_dir = Path('/Users/claire/Downloads/SEM_project_python/train/data')
mask_dir = Path('/Users/claire/Downloads/SEM_project_python/train/output')
output_dir = Path('/Users/claire/Downloads/SEM_project_python/masks/final')

for filename in os.listdir(img_dir):
    file_path = Path(img_dir + filename)
    if file_path.suffix == '.png' or file_path.suffix == '.tif' or file_path.suffix == '.jpg':
        # Print OG photo
        print(filename)
        image = Image.open(file_path)
        image = image.resize((1024, 768))
        plt.clf()
        # display(image)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        #print target mask
        mask_dir = "/content/drive/MyDrive/SEM_project/train/output/"
        label_name = mask_dir + os.path.basename(os.path.splitext(filename)[0]) + '_label.npy'
        label_np = np.load(label_name)
        label = torch.from_numpy(label_np) # convert label to tensor
        target_np = label.cpu().numpy().squeeze()
        label_mask = get_mask(target_np)
        label_mask = cv.resize(cv.cvtColor(label_mask, cv.COLOR_BGR2RGB), (1024, 768))
        # cv2_imshow(label_mask)
        plt.imshow(label_mask)
        plt.axis('off')
        plt.show()


        # print predicted mask
        image = grayscale(image)
        image = trans(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        print('predicted')
        print("IMAGE DIMS: ", image.shape)
        pred = model(image)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy().squeeze()
        pred_mask = get_mask(pred)
        pred_mask = cv.cvtColor(pred_mask, cv.COLOR_BGR2RGB)
        # cv2_imshow(pred_mask)
        plt.imshow(pred_mask)
        plt.axis('off')
        plt.show()

        # Save images as PNG files with epoch number in the filename
        imsave(os.path.join("/content/drive/MyDrive/SEM_project/masks/final", os.path.splitext(os.path.basename(file_path))[0] + "_final.png"), pred_mask)
        plt.clf()
        label = np.load(Path(mask_dir + "/" + os.path.splitext(os.path.basename(file_path))[0] + "_label.npy"))
        m = MultiLabelBinarizer().fit(label)
        f1 = f1_score(m.transform(label), m.transform(pred), average=None)
        if len(f1) != 6:
            continue
        for i, class_name in enumerate(classes):
            f1_dict[class_name].append(f1[i])
        print(f1)
for class_name in classes:
    print(class_name, np.mean(f1_dict[class_name]))

