import numpy as np
import os
from datasets import Dataset
from PIL import Image
# from huggingface_hub import hf_authenticate

#mount google drive
from google.colab import drive
drive.mount('/content/drive')

# Get .npy mask_files from masks_directory
masks_directory = '/content/drive/MyDrive/SEM_project_old/output1/labels_good'
images_directory = '/content/drive/MyDrive/SEM_project_old/png_images'
api_token ='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'
hf_dataset_identifier = 'chugz/SEM'

mask_files = [f for f in os.listdir(masks_directory) if f.endswith('.npy')]

masks = []        # List to store masks
images = []
image_ids = []    # List to store image name

for file_name in mask_files:
    mask_path = os.path.join(masks_directory, file_name)
    mask = np.load(mask_path)
    masks.append(mask)
    image_id = file_name[:-4]   # Remove .npy extension
     # Try to load the image with .jpg extension
    image_path_jpg = os.path.join(images_directory, f"{image_id}.jpg")
    image_path_png = os.path.join(images_directory, f"{image_id}.png")
    
    if os.path.exists(image_path_jpg):
        image = Image.open(image_path_jpg)
    elif os.path.exists(image_path_png):
        image = Image.open(image_path_png)
    else:
        raise FileNotFoundError(f"No image found for {image_id} with .jpg or .png extension")
    images.append(image)
    image_ids.append(image_id)

# Create a list of dictionaries where each dictionary represents one data sample
dataset_dict = {
    'image_id': image_ids,
    'image': images,
    'label': masks  # Assuming masks are already in the correct format (e.g., category IDs for each pixel)
}

# Create a Hugging Face dataset
hf_dataset = Dataset.from_dict(dataset_dict)

# Rename columns if necessary
hf_dataset = hf_dataset.rename_column('image', 'pixel_values')  # Assuming this is required by your model
# hf_dataset = hf_dataset.rename_column('label', 'label')

# Remove any unnecessary columns
hf_dataset = hf_dataset.remove_columns(['image_id'])

# Push to the Hugging Face Hub
hf_dataset.push_to_hub(repo_id = hf_dataset_identifier,
                      token = api_token,
                      private = True)

