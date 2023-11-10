import ast
import torch

from PIL import Image
import numpy as np

import cv2
import numpy as np

import ast

import fire

from llama import Llama
from typing import List

from llama.data_preprocessing import *

from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pydicom
from torchvision import transforms
import zipfile

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        labels = torch.arange(image_features.size(0), dtype=torch.long, device=image_features.device)
        loss = self.criterion(logits, labels) + self.criterion(logits.t(), labels)
        return loss / 2

def main(
    data_loader,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    epochs: int = 10
):
    
    llama = Llama.build(
        ckpt_dir='llama-2-7b',
        tokenizer_path='tokenizer.model',
        max_seq_len=256,
        max_batch_size=32,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llama.to(device)
    llama.get_trainable_params()

    
    optimizer = optim.Adam(
    lr=learning_rate,
    weight_decay=weight_decay
)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader) * epochs, eta_min=0, last_epoch=-1)
    
    epochs = 10  # Define your epochs
    for epoch in range(epochs):
        for images, texts in loader:
            images, texts = images.to(device), texts.to(device)

            optimizer.zero_grad()

            # Forward pass through LLaMA and visual forward model
            text_features, image_features = llama.forward_train(texts, images)

            # Compute contrastive loss
            loss = criterion(image_features, text_features)

            # Backward and optimize
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print('Training completed.')

    
if __name__ == "__main__":
    path = 'physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv.gz'
    df_split = pd.read_csv(path, compression='gzip')
    
    path = 'physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz'
    df_meta = pd.read_csv(path, compression='gzip')

    zip_file_path = 'mimic-cxr-reports.zip'
    extraction_path = 'mimic-cxr-reports'

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    df_meta['image_path'] = df_meta.apply(create_image_path, axis=1)
    # Apply the function to create report paths
    df_meta['report_path'] = df_meta.apply(lambda row: create_report_path(row), axis=1)

    # Now you have report paths that should match the extracted structure

    df_front = df_meta[df_meta['ViewPosition'].isin(['PA', 'AP'])]
    
    path = 'physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz'
    df_symptoms = pd.read_csv(path, compression='gzip')
    df_symptoms.fillna(0, inplace=True)
    df_symptoms.replace(-1, 0, inplace=True)
        
        
    # DataLoader setup
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_with_transforms = MIMIC_CXR(df_front, augment=transform)
    loader = DataLoader(dataset_with_transforms, batch_size=32, shuffle=True, collate_fn=my_collate_fn)
    
    main(loader)