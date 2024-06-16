# SEM
The goal of this model is to create accurate semantic segmentation masks that reflect the material composition in the cross-section SEM images. To do this, the current model uses a UNET architecture. 

# Dataset
The provided dataset consists of 88 images to train on. (I was originally given 89 images, but one image did not have the corresponding labels and was thus deleted from the dataset.) Each image (.png or .jpeg) corresponds to labels in a json file. Each image/file is identifiable using a filename and has a file_size, file_attributes, and region_count. Each region in the image has a region_id, region_shape_attributes (name:, all_points_x, all_points_y), and region_attributes. The region attributes is the material of the region (eg. silver)

# Changes:
* Split data between training (80% dataset = 70 images) and test (20% dataset = 18 images). This will move the images from the image_dir to either test_data or train_data and their corresponding npy labels from label_dir to either test_output or train_output.

# Problems

# Note
* Since the code moves the images from one file to another, it can be annoying to delete the .npy files and move the SEM images back to the image_dir. If you want to avoid doing this everytime, comment out the split_data code. This will allow the code to work directly with the code in the train/ test folders and keeps the training and test data consistent in each run.
