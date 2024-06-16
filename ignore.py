from data import*
from model import *
from train_test import *
from eval import *

def split_data(image_dir, label_dir, train_image_dir, train_label_dir, test_image_dir, test_label_dir, test_size=0.2):
    images = [f for f in image_dir.iterdir() if f.suffix in {'.png', '.jpg'}]
    labels = [f for f in label_dir.iterdir() if f.suffix == '.npy']

    image_bases = {os.path.splitext(f)[0] for f in images}
    label_bases = {os.path.splitext(f)[0].replace('_label', '') for f in labels}
    common_bases = list(image_bases & label_bases)

    print(f"Found {len(common_bases)} common images and labels.")

    if len(common_bases) == 0:
        raise ValueError("No common images and labels found to split.")

    images = [f"{base}.jpg" if f"{base}.jpg" in images else f"{base}.png" for base in common_bases]
    labels = [f"{base}_label.npy" for base in common_bases]

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=42)

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

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
    save_folder = "/content/drive/MyDrive/SEM_project/masks"
    # Create the folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    checkpoint_folder = "/content/drive/MyDrive/SEM_project/checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)

    # ======================================================================
    # Define loss function.
    # ----------------- YOUR CODE HERE ----------------------
    #
    # Remove NotImplementedError and assign correct loss function.

    # ======================================================================
    # Define optimizer function.
    # ----------------- YOUR CODE HERE ----------------------
    #
    # Remove NotImplementedError and assign appropriate optimizer with learning rate and other parameters.
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    # Create transformations to apply to each data sample
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        transforms.Normalize((0.5), (0.5))
         ])
    criterion = nn.CrossEntropyLoss()
    class_weights = torch.Tensor([1, 1, 1.5, 1, 1.5, 1.5]).to(device)

    #################### LOAD DATA ####################
    image_dir = '/content/drive/MyDrive/SEM_project/data/'
    label_dir = '/content/drive/MyDrive/SEM_project/output'
    train_image_dir = '/content/drive/MyDrive/SEM_project/train/data/'
    train_label_dir = '/content/drive/MyDrive/SEM_project/train/output/'
    test_image_dir = '/content/drive/MyDrive/SEM_project/test/data/'
    test_label_dir = '/content/drive/MyDrive/SEM_project/test/output/'

    # Split data into training and test sets
    split_data(image_dir, label_dir, train_image_dir, train_label_dir, test_image_dir, test_label_dir)

    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = PVDataset(train_image_dir, train_label_dir, transform=transform)
    dataset2 = PVDataset(test_image_dir, test_label_dir, transform=transform)
    train_loader = DataLoader(dataset1, batch_size=FLAGS.batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset2, batch_size=FLAGS.batch_size,
                             shuffle=False, num_workers=2)

    best_accuracy = 0.0
    checkpoint_path = '/content/drive/MyDrive/SEM_project/checkpoints/model_filled2.pth'
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

        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
    # print("accuracy is {:2.2f}".format(best_accuracy))
    # file.write("accuracy is {:2.2f}".format(best_accuracy))
    torch.save(model, '/content/drive/MyDrive/SEM_project/final_model_filled2.pth')

    print("Training and evaluation finished")

    plt.plot(train_losses)
    plt.savefig("/content/drive/MyDrive/SEM_project/graphs/Train_loss_model.jpg")
    plt.clf()
    plt.plot(train_accs)
    plt.savefig("/content/drive/MyDrive/SEM_project/graphs/Train_acc_model.jpg")
    plt.clf()
    plt.plot(test_losses)
    plt.savefig("/content/drive/MyDrive/SEM_project/graphs/Test_loss_model.jpg")
    plt.clf()
    plt.plot(test_accs)
    plt.savefig("/content/drive/MyDrive/SEM_project/graphs/Test_acc_model.jpg")
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
                        default=280,
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
