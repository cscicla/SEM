from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob

import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.encoders import get_preprocessing_fn

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, IterableDataset
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

# preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')

class PVDataset(IterableDataset):
    def __init__(self, img_dir, label_dir, transform=None, shuffle=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_list = glob.glob(img_dir + '*.png')
        self.image_list.extend(glob.glob(img_dir + '*.jpg'))
        self.image_list.extend(glob.glob(img_dir + '*.tif'))
        self.transform = transform
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.image_list)
        for img_name in self.image_list:
            image = Image.open(img_name).convert("L")  # grayscale
            image = image.resize((1024, 768))
            label_name = self.label_dir + os.path.basename(os.path.splitext(img_name)[0]) + '_label.npy'
            label_np = np.load(label_name)
            label = torch.from_numpy(label_np)  # convert label to tensor

            if self.transform:
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

            yield image, label

def get_mask(all_mask):
    '''
    convert the mask to color based on class
    '''
    #initialize empty mask
    height = all_mask.shape[0]
    width = all_mask.shape[1]
    final_mask = np.zeros((height, width, 3))

    # for each pixel, give it a color
    for i in range(len(all_mask)):
        for j in range(len(all_mask[i])):
            if all_mask[i][j] == 0: # was not changed from the initalized zero
                final_mask[i][j] = np.array([0, 0, 0])      # unlabeled
            elif all_mask[i][j] == 1:
                final_mask[i][j] = np.array([128, 64, 128]) # silver
            elif all_mask[i][j] == 2:
                final_mask[i][j] = np.array([189, 201, 18]) # glass
            elif all_mask[i][j] == 3:
                final_mask[i][j] = np.array([0, 102, 0])    # silicon
            elif all_mask[i][j] == 4:
                final_mask[i][j] = np.array([0, 76, 130])   # void
            elif all_mask[i][j] == 5:
                final_mask[i][j] = np.array([36, 18, 201])  # interfacialvoid
    # convert it to standard image format
    return final_mask.astype(np.uint8)

def save_images(data, output, target, epoch, testflag):
    '''
        save the original image, ground_truth/ target mask, and predicted mask
    '''
    # save_folder = "/home/crcvreu.student10/SEM_python/masks/"
    save_folder = Path("/home/crcvreu.student10/run/sem_batches/masks/")
    
    # Convert tensors to numpy arrays
    for i in range(len(data)):
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
        label_mask = cv.resize(label_mask, (1024, 768))

        data_np = data_np * 255
        data_np = data_np.astype(np.uint8)

        # Save images as PNG files with epoch number in the filename
        predicted_filename = save_folder / f'epoch_{epoch}_predicted.png'
        imsave(predicted_filename, pred_mask)

        # Save original and target images at beginning of run
        if epoch == 1:
            original_filename = save_folder / 'original.png'
            target_filename = save_folder / 'target.png'
            
            imsave(original_filename, data_np)
            imsave(target_filename, label_mask)

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
        if len(f1_scores_class[i]) < 6: # 5 classes + background
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
    # Get images and labels
    images = [f for f in image_dir.iterdir() if f.suffix in ('.png', '.jpg')]
    labels = [f for f in label_dir.iterdir() if f.suffix == '.npy']

    # Get base names
    image_bases = {f.stem for f in images}
    label_bases = {f.stem.replace('_label', '') for f in labels}

    # Find common bases
    common_bases = list(image_bases & label_bases)

    print(f"Found {len(common_bases)} common images and labels.")

    if len(common_bases) == 0:
        raise ValueError("No common images and labels found to split.")

    images = [f"{base}.jpg" if f"{base}.jpg" in images else f"{base}.png" for base in common_bases]
    labels = [f"{base}_label.npy" for base in common_bases]

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    for image, label in zip(train_images, train_labels):
        shutil.move(image_dir / image, train_image_dir / image)
        shutil.move(label_dir / label, train_label_dir / label)

    # Move test images and labels
    for image, label in zip(test_images, test_labels):
        shutil.move(image_dir / image, test_image_dir / image)
        shutil.move(label_dir / label, test_label_dir / label)
    print(f"Moved {len(train_images)} images to training set and {len(test_images)} images to test set.")

def run_main(FLAGS, file):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    # Set proper device based on cuda availability
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    file.write("\n\nTorch device selected: " + str(device) + '\n')

    # Initialize the model and send to device
    UNet_PV = smp.UnetPlusPlus(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=6,                      # model output channels (number of classes in your dataset)
    )
    model = UNet_PV.to(device)
    save_folder = "/home/crcvreu.student10/run/sem_batches/masks"
    checkpoint_folder = "/home/crcvreu.student10/run/sem_batches/checkpoints"
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
            # higher weight decay = lower overfitting

    # Create transformations to apply to each data sample
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
         ])
    criterion=L.DiceLoss('multiclass')
    class_weights = torch.Tensor([1, 1, 1.5, 1, 1.5, 1.5]).to(device)

    #################### LOAD DATA ####################
    # image_dir = '/home/crcvreu.student10/run/sem_batches/data/'
    # label_dir = '/home/crcvreu.student10/run/sem_batches/output'
    train_image_dir = '/home/crcvreu.student10/run/sem_batches/train/data/'
    train_label_dir = '/home/crcvreu.student10/run/sem_batches/train/output/'
    test_image_dir = '/home/crcvreu.student10/run/sem_batches/test/data/'
    test_label_dir = '/home/crcvreu.student10/run/sem_batches/test/output/'

    # Split data into training and test sets
    # split_data(image_dir, label_dir, train_image_dir, train_label_dir, test_image_dir, test_label_dir)

    # Load datasets for training and testing
    train_dataset = PVDataset(train_image_dir, train_label_dir, transform=transform, shuffle=True)
    test_dataset = PVDataset(test_image_dir, test_label_dir, transform=transform, shuffle=False)

    best_accuracy = 0.0
    checkpoint_path = '/home/crcvreu.student10/run/sem_batches/checkpoints/model_filled2.pth'
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
        train_loss, train_acc, train_f1_class, train_f1 = train(model, device, train_dataset,
                                      optimizer, criterion, epoch,
                                      FLAGS.batch_size, file, class_weights)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_f1_class_scores.append(train_f1_class)
        train_f1_scores.append(train_f1)

        # test
        test_loss, test_acc, test_f1_class, test_f1 = test(model, device, test_dataset,
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

    torch.save(model, '/home/crcvreu.student10/run/sem_batches/final_model_filled2.pth')
    print("Training and evaluation finished")

    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('Train and Test Losses')
    plt.legend()
    plt.grid(True)
    plt.ylim(0,1)
    plt.savefig("/home/crcvreu.student10/run/sem_batches/graphs/Train_Test_loss_model.jpg")
    plt.clf()

    plt.plot(train_accs, label='train accuracy')
    plt.plot(test_accs, label='test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0,1)
    plt.savefig("/home/crcvreu.student10/run/sem_batches/graphs/Train_Test_acc_model.jpg")
    plt.clf()

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
    plt.ylim(0,1)
    plt.savefig("/home/crcvreu.student10/run/sem_batches/graphs/Train_f1_per_class_model.jpg")
    plt.clf()

    # plt.plot(test_f1_class_scores)
    for class_idx in range(num_classes):
        class_scores = [epoch_scores[class_idx] for epoch_scores in test_f1_class_scores]
        plt.plot(epochs, test_f1_class_scores, label=classes[class_idx])
    plt.xlabel('Epochs')
    plt.ylabel('Test F1')
    plt.title('Test F1 per class')
    plt.legend(classes, loc='upper left')
    plt.grid(True)
    plt.ylim(0,1)
    plt.savefig("/home/crcvreu.student10/run/sem_batches/graphs/Test_f1_per_class_model.jpg")
    plt.clf()

    plt.plot(train_f1_scores, label = 'train f1 accuracy')
    plt.plot(test_f1_scores, label='test f1 accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()
    plt.title('Train and Test F1 Scores')
    plt.grid(True)
    plt.ylim(0,1)
    plt.savefig("/home/crcvreu.student10/run/sem_batches/graphs/Train_Test_f1_model.jpg")
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
                        default=300,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=1,
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

UNet_PV = smp.UnetPlusPlus(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=6,                      # model output channels (number of classes in your dataset)
)
model = UNet_PV.to(device)
# model = smp.Unet()
# model = model.to(device)
model = torch.load('/home/crcvreu.student10/run/sem_batches/final_model_filled2.pth')
model.eval()

classes = ["background", "silver", "glass", "silicon", "void", "interfacial void"]
f1_dict = {"background":[], "silver":[], "glass":[], "silicon":[], "void":[], "interfacial void":[]}

img_dir = Path('/home/crcvreu.student10/run/sem_batches/test/data/')
mask_dir = Path("/home/crcvreu.student10/run/sem_batches/test/output/")
target_mask_dir = Path("/home/crcvreu.student10/run/sem_batches/masks/target/")
final_mask_dir = Path("/home/crcvreu.student10/run/sem_batches/masks/final/")

# Process images
with open('f1results.txt', 'a') as file_output:
    # Initialize f1_dict
    f1_dict = {class_name: [] for class_name in classes}
    for file_path in img_dir.iterdir():
        if file_path.suffix in {'.png', '.tif', '.jpg'}:
            # Print OG photo
            print(file_path.name)
            file_output.write(str(file_path.name) + '\n')
            image = Image.open(file_path)
            image = image.resize((1024, 768))
            plt.clf()
            # display(image)  # Uncomment if display is needed

            # Print target mask
            label_name = mask_dir / (file_path.stem + '_label.npy')
            label_np = np.load(label_name)
            label = torch.from_numpy(label_np)  # Convert label to tensor
            target_np = label.cpu().numpy().squeeze()
            label_mask = get_mask(target_np)
            # label_mask = cv.resize(cv.cvtColor(label_mask, cv.COLOR_BGR2RGB), (1024, 768))
            label_mask = cv.resize(label_mask, (1024, 768))
            target_mask_path = target_mask_dir / (file_path.stem + "_target.png")
            imsave(target_mask_path, label_mask)

            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])

            # Print predicted mask
            image = image.convert("L")
            image = trans(image)
            image = image.to(device)
            image = image.unsqueeze(0)
            pred = model(image)
            pred = torch.nn.functional.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy().squeeze()
            pred_mask = get_mask(pred)
            final_mask_path = final_mask_dir / (file_path.stem + "_final.png")
            imsave(final_mask_path, pred_mask)

            plt.clf()
            label = np.load(mask_dir / (file_path.stem + "_label.npy"))
            m = MultiLabelBinarizer().fit(label)
            f1 = f1_score(m.transform(label), m.transform(pred), average=None)
            if len(f1) != 6:
                continue
            for i, class_name in enumerate(classes):
                f1_dict[class_name].append(f1[i])
            print(f1_dict)
            file_output.write(str(f1_dict) + '\n')

    for class_name in classes:
        mean_f1 = np.mean(f1_dict[class_name])
        print(class_name, mean_f1)
        file_output.write(f'{class_name}: {mean_f1}\n')
        