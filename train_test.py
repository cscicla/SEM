# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
import gc
from pytorch_toolbelt import losses as L
from matplotlib.pyplot import imsave, imshow
from model import *
from data import *

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
    for i in range(FLAGS.batch_size):
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