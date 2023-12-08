import os
import pandas as pd
import numpy as np
import gzip
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pydicom
from torchvision import transforms
import zipfile
import re

path = '/scratch/ny675/jpg-data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz'
df_meta = pd.read_csv(path, compression='gzip')

def create_image_path(row):
    base = '/scratch/ny675/jpg-data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
    p1 = 'p' + str(row['subject_id'])[:2]
    if int(str(row['subject_id'])[:2]) == 10:
        p2 = 'p' + str(row['subject_id'])
        s = 's' + str(row['study_id'])
        dcm = row['dicom_id'] + '.jpg' # can change to + '.dcm'
        return f"{base}{p1}/{p2}/{s}/{dcm}"
    else:
        return None

df_meta['image_path'] = df_meta.apply(create_image_path, axis=1)
df_meta = df_meta[df_meta['image_path'].notna()]

df_front = df_meta[df_meta['ViewPosition'].isin(['PA', 'AP'])]


path = '/scratch/ny675/jpg-data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz'
df_symptoms = pd.read_csv(path, compression='gzip')
df_symptoms.fillna(0, inplace=True)
df_symptoms.replace(-1, 0, inplace=True)

front_symptoms = df_front.merge(df_symptoms, on=['subject_id', 'study_id'], how='inner')


# List of your label column names
label_columns = ['Atelectasis', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
       'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']  # Add all label column names
for col in label_columns:
    front_symptoms[col] = front_symptoms[col].astype(int)
front_symptoms['combined_labels'] = front_symptoms[label_columns].values.tolist()



class MIMIC_CXR(Dataset):
    def __init__(self, metadata_df, augment=None, img_depth=3, image_format='jpg'): # can change to jpg format
        self.metadata_df = metadata_df
        self.augment = augment
        self.img_depth = img_depth
        self.image_format = image_format

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, index):
        # Getting image path and loading image based on format
        file = self.metadata_df.iloc[index]['image_path']
        
        try:
            if self.image_format == 'dcm':
                dicom_file = pydicom.dcmread(file)
                imageData = dicom_file.pixel_array
                imageData = Image.fromarray(imageData).convert("L")
            else:  # Assuming it's a format readable by PIL (like jpg)
                imageData = Image.open(file)
                if self.img_depth == 1:
                    imageData = imageData.convert('L')
                else:
                    imageData = imageData.convert('RGB')

            # Applying augmentations if any
            if self.augment:
                imageData = self.augment(imageData)

            label = self.metadata_df.iloc[index]['combined_labels']
            label_tensor = torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing image {file}: {e}")
            return None, None  # or handle the error in a different way

        return imageData, label_tensor
    
    
# Define transformations/augmentations
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std) # Convert PIL image to PyTorch tensor
])

dataset_with_transforms = MIMIC_CXR(front_symptoms, augment=transform)

import pickle

# Assuming dataset_with_transforms is your dataset object

with open('dataset.pkl', 'wb') as file:
    pickle.dump(dataset_with_transforms, file)

# Use DataLoader to create a loader for batch processing
batch_size = 32

def my_collate_fn(batch):
    batch = [(img, lbl) for img, lbl in batch if img is not None]
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

dataloader = DataLoader(dataset_with_transforms, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
print("dataset of image is ready!")

# for batch_images, batch_reports in dataloader:
#     print("Batch labels:", batch_reports)
#     print("First label in Batch:", batch_reports[0])
#     print("Batch labels Length:", len(batch_reports))
#     print(batch_images.shape, 'image')
# torch.Size([1, 1, 224, 224]) if dcm
# torch.Size([1, 3, 224, 224]) if jpg

# only 2 example jpgs used here