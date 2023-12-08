
import ast
import torch

from PIL import Image
import numpy as np

import cv2
import numpy as np
import pandas as pd

import ast

import fire

from llama import Llama
from typing import List

from llama.data_preprocessing import *

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch import nn, optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pydicom
from torchvision import transforms
import zipfile
import pickle


    
llama = Llama.build(
    ckpt_dir='llama-2-7b',
    tokenizer_path='tokenizer.model',
    max_seq_len=2000,
    max_batch_size=8,
)

path = '/scratch/ny675/jpg-data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz'
df_split = pd.read_csv(path, compression='gzip')

path = '/scratch/ny675/jpg-data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz'
df_meta = pd.read_csv(path, compression='gzip')

zip_file_path = '/scratch/ny675/mimic-cxr-reports.zip'
extraction_path = '/scratch/ny675/mimic-cxr-reports'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

df_meta['image_path'] = df_meta.apply(create_image_path, axis=1)
# Apply the function to create report paths
df_meta['report_path'] = df_meta.apply(lambda row: create_report_path(row), axis=1)

# Now you have report paths that should match the extracted structure
df_meta = df_meta[df_meta['image_path'].notna() & df_meta['report_path'].notna()]

df_front = df_meta[df_meta['ViewPosition'].isin(['PA', 'AP'])]

path = '/scratch/ny675/jpg-data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz'
df_symptoms = pd.read_csv(path, compression='gzip')
df_symptoms.fillna(0, inplace=True)
df_symptoms.replace(-1, 0, inplace=True)

# DataLoader setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_with_transforms = MIMIC_CXR(llama, df_front, augment=transform)

data = []
for i in range(len(dataset_with_transforms)):
    try:
        data.append(dataset_with_transforms[i])
    except Exception as e:
        print(f"Error processing index {i}: {e}")

with open('dataset6.pkl', 'wb') as file:
    pickle.dump(data, file)
