from datasets import load_dataset

import json
from huggingface_hub import hf_hub_download, HfApi

import transformers

from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor

from transformers import SegformerForSemanticSegmentation

import accelerate
from transformers import TrainingArguments, Trainer

import torch
from torch import nn
import evaluate

import os
from huggingface_hub import HfFolder
from transformers import Trainer

from PIL import Image
import numpy as np

'''
used this code (modified to work wth own dataset):
https://huggingface.co/blog/fine-tune-segformer#create-your-own-dataset
'''

################ LOAD THE DATASET ################
print('\n===========================================')
print('\nloading dataset...\n')
api_token ='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'
hf_dataset_identifier = 'chugz/SEM'
ds = load_dataset(hf_dataset_identifier)

# shuffle and split into training and testing data
ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]

# get labels
# id2label = {0: "background", 1: "silver", 2: "glass", 3: "silicon", 4: "void", 5: "interfacial void"}
# with open("id2label.json", "w") as f:
#     json.dump(id2label, f)
api = HfApi()
api.upload_file(
    path_or_fileobj="id2label.json",
    path_in_repo="id2label.json",
    repo_id=hf_dataset_identifier,
    repo_type="dataset",
    use_auth_token=api_token
)

filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=hf_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
print("Id2label:", id2label)
print("Label2id:", label2id)

################ PROCESS AND AUGMENT IMAGES ################
print('\n===========================================')
print('\nstarting data processing and augmentation...')
processor = SegformerImageProcessor()
# jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
jitter = ColorJitter(brightness=0.25, contrast=0.25)
    # color jitter is designed to work with RGB, so convert to RGB first then change back to grayscale
        # def train_transforms(example_batch):
        #     images = [jitter(x) for x in example_batch['pixel_values']]
        #     labels = [x for x in example_batch['label']]
        #     inputs = processor(images, labels)
        #     return inputs


        # def val_transforms(example_batch):
        #     images = [x for x in example_batch['pixel_values']]
        #     labels = [x for x in example_batch['label']]
        #     inputs = processor(images, labels)
        #     return inputs

def transform_image(img):
    img = Image.fromarray(np.uint8(img))  # Assuming img is a numpy array
    img = img.convert('RGB')
    img = jitter(img)
    # img = img.convert('L')
    return img

# Define dataset transforms
def train_transforms(example_batch):
    images = [transform_image(img) for img in example_batch['pixel_values']]
    labels = [np.array(x) for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

def val_transforms(example_batch):
    images = [transform_image(img) for img in example_batch['pixel_values']]
    labels = [np.array(x) for x in example_batch['label']]
    inputs = processor(images, labels)
    return inputs

# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)


print('\n===========================================')
print('\nfinetunting SegFormer')
pretrained_model_name = "nvidia/mit-b0"
print(f'\nloading pretrained model:{pretrained_model_name}')
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)

print('\n===========================================')
print('\nSetting up the trainer...')
epochs = 50
lr = 0.00006
batch_size = 2


hub_model_id = "segformer-b0-practice-7-11"
training_args = TrainingArguments(
    "sformer-practice-outputs",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_model_id=hub_model_id,
    hub_strategy="end",
)

print('after Training Args')
metric = evaluate.load("mean_iou")
print('after evaluate.load')

############## CALCULATE EVALUATION METRICS ##############
''' function that computes the evaluation metric we want to work with'''
def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=len(id2label),
        ignore_index=0,
        reduce_labels=processor.do_reduce_labels,
    )

    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics
  
HfFolder.save_token(api_token)

# instantiate a Tariner object #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
# and train
print('before train')
trainer.train()
print('after train')

# push fine-tuned model to image processor and create a model card
kwargs = {
    "tags": ["vision", "image-segmentation"],
    "finetuned_from": pretrained_model_name,
    "dataset": hf_dataset_identifier,
}

processor.push_to_hub(hub_model_id)
trainer.push_to_hub(**kwargs)

# processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# model = SegformerForSemanticSegmentation.from_pretrained(hf_dataset_identifier)

