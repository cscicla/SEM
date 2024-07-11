<<<<<<< HEAD
# SEM
=======
# SEM
The goal of this model is to create accurate semantic segmentation masks that reflect the material composition in the cross-section SEM images. To do this, the current model uses a UNET architecture. 

# Dataset
The provided dataset consists of 66 images to train on. Each image (.png or .jpeg) corresponds to labels in a json file. Each image/file is identifiable using a filename and has a file_size, file_attributes, and region_count. Each region in the image has a region_id, region_shape_attributes (name:, all_points_x, all_points_y), and region_attributes. The region attributes is the material of the region (eg. silver)

# Changes:
* Split data between training (80% dataset) and test (20% dataset). This will move the images from the image_dir to either test_data or train_data and their corresponding npy labels from label_dir to either test_output or train_output.

# Problems

# Directories
* masks/target directory: stores the masks ground truth masks as images for references
* masks/predicted directory: stores the results of the model's predictions after evaluation
* output1 directory: stores the labels as json/ csv files
* test directory: stores the testing dataset
    * 'data' (.png/.jpeg)
    * labels/'output' (.npy)
* train directory: stores the training dataset
    * 'data' (.png/.jpeg)
    * labels/'output' (.npy)
* .slurm
    * run_sem1.slurm: used to run the python code
    * example.slurm: temporary file that is currently being used as a template .slurm file
    * segformer_practice.slurm: used to run the tutorial code
* old_models directory: used to keep track of older models/ python codes

# Files
* segformer_hugdata.py
    * used to add own dataset to HuggingFace
* sem_unet_plus.py
    * best performance of older models
    * UNet++
* sem_batche.py
    * better performance than UNet++
    * changes made to hyperparameters:
        * Increased learning rate to 0.0001
        * Increased batch size to 6
        * Early Stopping (91 epochs)
* cm_sem.py
    * ran sem_batche code with confusion matrix
        * need to improve/ normalize confusion matrix outputed

# Credit
* For the older models (UNet++, Linknet, MANet) used    
    * @misc{Iakubovskii:2019,
        Author = {Pavel Iakubovskii},
        Title = {Segmentation Models Pytorch},
        Year = {2019},
        Publisher = {GitHub},
        Journal = {GitHub repository},
        Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
        }
* For SegFormer information used
    * https://huggingface.co/blog/fine-tune-segformer#create-your-own-dataset 