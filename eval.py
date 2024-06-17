from data import *
from model import *
from train_test import *
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import cv2 as cv
from PIL.ImageOps import grayscale

def eval():
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    # Set proper device based on cuda availability
    device = torch.device("cuda" if use_cuda else "cpu")

    model = UNet_PV().to(device)
    model = torch.load('/home/crcvreu.student10/SEM/checkpoints/final_model_filled2.pth')
    model.eval()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    classes = ["background", "silver", "glass", "silicon", "void", "interfacial void"]
    f1_dict = {"background":[], "silver":[], "glass":[], "silicon":[], "void":[], "interfacial void":[]}
    flag = 1
    counter = 0

    img_dir = Path('/home/crcvreu.student10/SEM/train/data')
    mask_dir = Path('/home/crcvreu.student10/SEM/train/output')
    output_dir = Path('/home/crcvreu.student10/SEM/masks/final')

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