import numpy as np
from PIL import Image, ImageDraw
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

json_name = '/content/drive/MyDrive/SEM_project_old/output1/06_05_2024_master_annotations.json'
image_directory = '/content/drive/MyDrive/SEM_project_old/png_images'
labels_directory = '/content/drive/MyDrive/SEM_project_old/output1/labels'
masks_directory = '/content/drive/MyDrive/SEM_project_old/output1/masks'

print('here')
def create_mask(width, height, polygons):
    # Create a new image mask
    img = Image.new('L', (width, height), 0)

    # Draw polygons on the image mask
    for poly in polygons:
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)

    # Convert image mask to numpy array
    return np.array(img)

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

def get_numpys(json_name):
    # open JSON file
    with open(json_name) as f:
        pv_json = json.load(f)

    # parse JSON
    ################# FOR EACH IMAGE #################
    for i in pv_json:
        filename = pv_json[i]['filename']
        # print(f"Processing image '{filename}'...")

        # Check if the image file exists
        image_path = os.path.join(image_directory, filename)
        if not os.path.exists(image_path):
            print(f"Image file '{filename}' not found in '{image_directory}'. Skipping...")
            continue

        # create empty lists to append polygons to
        silver_polygons = []
        glass_polygons = []
        silicon_polygons = []
        void_polygons = []
        interfacialvoid_polygons = []

        ################# FOR EACH REGION #################
        for j in pv_json[i]['regions']:
            region_type = j['region_attributes']['type']
            polygon = []
            for k in range(len(j['shape_attributes']['all_points_x'])):
                polygon.append((j['shape_attributes']['all_points_x'][k], j['shape_attributes']['all_points_y'][k]))

            # append the polygon to the appropriate typed list
            if region_type == 'silver':
                silver_polygons.append(polygon)
            elif region_type == 'glass':
                glass_polygons.append(polygon)
            elif region_type == 'silicon':
                silicon_polygons.append(polygon)
            elif region_type == 'void':
                void_polygons.append(polygon)
            else:
                interfacialvoid_polygons.append(polygon)

        ################# GET OG IMAGE INFORMATION #################
        # Open the image using PIL
        img = Image.open(image_path)
        width, height = img.size

        # Create masks for each type
        silver_mask = create_mask(width, height, silver_polygons)
        glass_mask = create_mask(width, height, glass_polygons)
        silicon_mask = create_mask(width, height, silicon_polygons)
        void_mask = create_mask(width, height, void_polygons)
        interfacialvoid_mask = create_mask(width, height, interfacialvoid_polygons)

        # remove voids from silver and silicon and glass
        silver_mask = silver_mask.astype(np.float32)
        silicon_mask = silicon_mask.astype(np.float32)
        glass_mask = glass_mask.astype(np.float32)
        void_mask = void_mask.astype(np.float32)
        interfacialvoid_mask = interfacialvoid_mask.astype(np.float32)

        silver_mask = np.maximum(silver_mask - void_mask, 0)
        silver_mask = np.maximum(silver_mask - interfacialvoid_mask, 0)
        silver_mask = np.maximum(silver_mask - glass_mask, 0)
        silver_mask = np.maximum(silver_mask - silicon_mask, 0)

        silicon_mask = np.maximum(silicon_mask - void_mask, 0)
        silicon_mask = np.maximum(silicon_mask - interfacialvoid_mask, 0)
        silicon_mask = np.maximum(silicon_mask - glass_mask, 0)

        glass_mask = np.maximum(glass_mask - void_mask, 0)
        glass_mask = np.maximum(glass_mask - interfacialvoid_mask, 0)

        void_mask = np.maximum(void_mask - interfacialvoid_mask, 0)

        interfacialvoid_mask = np.maximum(interfacialvoid_mask - void_mask, 0)

        # Combine masks into a single mask
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        combined_mask[silver_mask > 0] = 1
        combined_mask[glass_mask > 0] = 2
        combined_mask[silicon_mask > 0] = 3
        combined_mask[void_mask > 0] = 4
        combined_mask[interfacialvoid_mask > 0] = 5

        # Save the numpy array
        output_path = os.path.join(labels_directory, f'{Path(pv_json[i]["filename"]).stem}.npy')
        np.save(output_path, combined_mask)

        combined_img = get_mask(combined_mask)
        combined_img = Image.fromarray(combined_img)
        
        # # Save combined mask as PNG
        # combined_img = Image.fromarray(combined_mask, mode='L')
        combined_img.save(os.path.join(masks_directory, f"{Path(pv_json[i]['filename']).stem}_masks.png"))

        print(f"Saved PNG mask for '{filename}'")

get_numpys(json_name)
