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
from llama.main_model import med_mae

from llama.data_preprocessing import *

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch import nn, optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pydicom
from torchvision import transforms
import zipfile
import pickle

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

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
    
    def show_logits(self, image_features, text_features):
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        return logits


def save_weights(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model weights saved to {filename}")

def load_weights(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    print(f"Model weights loaded from {filename}")
    
def load_checkpoint(filename, model, optimizer, scheduler=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    return start_epoch
    
def main(
    data_loader,
    checkpoint_path=None,
    learning_rate: float = 3e-8,
    weight_decay: float = 0.02,
    epochs: int = 70
):
    main_model = med_mae()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_model.to(device)
    
    if checkpoint_path:
        load_weights(main_model, checkpoint_path)

    optimizer = optim.Adam(
        main_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    main_model.load_state_dict(torch.load('med_mae_model.pth'))
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader) * epochs, eta_min=0, last_epoch=-1)
    criterion = ContrastiveLoss()
    print('Start Training...')
#     global_loss = np.array([])
    
#     initial_weights = {
#         "attention_pooling_key": main_model.attention_pooling.key.weight.data.clone(),
#         "attention_pooling_value": main_model.attention_pooling.value.weight.data.clone(),
#         "patch_embed_proj": main_model.visual_forward.patch_embed.proj.weight.data.clone()
#     }
#     for name, weights in initial_weights.items():
#         print(f"Initial weights for {name}: weights {weights}")
    global_loss = np.load('loss_array_main_model.npy')
#     start_epoch = load_checkpoint('main_pretrain_checkpoint/checkpoint_epoch_50.pth.tar', main_model, optimizer, scheduler)
    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        epoch_loss = []
        for images, texts in data_loader:
            images, texts = images.to(device), texts.to(device)
#             print(images.shape)
#             print(texts.shape)

            optimizer.zero_grad()

            # Forward pass through LLaMA and visual forward model
            image_features, text_features = main_model.forward(images, texts)

            # Compute contrastive loss
            loss = criterion(image_features, text_features)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        global_loss = np.append(global_loss, epoch_loss)
            
        if (epoch + 1) % 2 == 0:
            scheduler.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        np.save(f'main_pretrain_checkpoint/third_checkpoint_epoch_{epoch+1}_loss_array_main_model.npy', global_loss)
        
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': main_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),  # Save scheduler state if you are using a scheduler
    }, filename=f"main_pretrain_checkpoint/third_checkpoint_epoch_{epoch+1}.pth.tar")
    
    save_weights(main_model, 'med_mae_model.pth')
    np.save('loss_array_main_model.npy', global_loss)
    print('Training completed.')
    
#     torch.save(main_model.state_dict(), 'main_model.pth')
    
#     new_weights = {
#         "attention_pooling_key": main_model.attention_pooling.key.weight.data.clone(),
#         "attention_pooling_value": main_model.attention_pooling.value.weight.data.clone(),
#         "patch_embed_proj": main_model.visual_forward.patch_embed.proj.weight.data.clone()
#     }
#     for name, weights in new_weights.items():
#         print(f"New weights for {name}: weights {weights}")

    
if __name__ == "__main__":
#     with open('dataset1.pkl', 'rb') as file:
#         loaded_dataset = pickle.load(file)
#     loader = DataLoader(loaded_dataset, batch_size=16, collate_fn=my_collate_fn)
    datasets = []

    #Loop through the dataset files
    for i in range(6, 7):  
        file_name = f'dataset{i}.pkl'
        print(file_name)
        with open(file_name, 'rb') as file:
            loaded_dataset = pickle.load(file)
            datasets.append(loaded_dataset)

#     Assuming you want to concatenate all datasets into a single dataset
#     You might need to modify this part depending on how you want to use the datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)

    # Create a DataLoader for the combined dataset
    loader = DataLoader(combined_dataset, batch_size=8, collate_fn=my_collate_fn)
  
    print("dataloader created.")
    
    main(loader)
