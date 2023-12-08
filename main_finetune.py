import ast
import torch
import os

from PIL import Image
import numpy as np

import cv2
import numpy as np
import pandas as pd

import ast

import fire

from llama import Llama
from typing import List
from llama.main_model import med_mae, down_stream_model
from llama.mae import MaskedAutoencoderViTForClassification

from finetune_preprocess import *

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch import nn, optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pydicom
from torchvision import transforms
import zipfile
import pickle
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer, scheduler=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer'])
#    scheduler.load_state_dict(checkpoint['scheduler'])
   start_epoch = checkpoint['epoch']
    return start_epoch
    

def save_weights(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model weights saved to {filename}")

def load_weights(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    print(f"Model weights loaded from {filename}")

def multi_label_accuracy(outputs, labels, threshold=0.5):
    outputs = (outputs > threshold).float()
    correct_predictions = (outputs == labels).float()
    accuracy = correct_predictions.sum() / correct_predictions.numel()
    return accuracy.item()    
    
def main(
    pretrained_checkpoint_path = './llama/mae_pretrain_vit_base.pth',
    learning_rate: float = 1e-5,
    weight_decay: float = 0.001,
    epochs: int = 100
):
    start_time = time.time()
    print('checkpoint1')    
    # Load your dataset
    with open('dataset.pkl', 'rb') as file:
        dataset = pickle.load(file)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Define DataLoader
    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn) 
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)
    
    
    num_batches = len(train_loader)
    print(f"Number of batches in train_loader: {num_batches}")

    # Shape of the first element in train_loader
    for batch in train_loader:
        inputs, targets = batch
        print(f"Shape of the first batch of inputs: {inputs.shape}")
        print(f"Shape of the first batch of targets: {targets.shape}")
        break
    
    # main_model = MaskedAutoencoderViTForClassification(num_labels = 14, chkpt_dir = pretrained_checkpoint_path)
    
    main_model = down_stream_model('./med_mae_model.pth')
    
    
#     # model_checkpoint_path = './best_mae_model.pth'
#     model_checkpoint_path = './med_mae_model.pth'
    
#     if os.path.exists(model_checkpoint_path):
#         print('Loading checkpoint')
#         load_weights(main_model, model_checkpoint_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_model.to(device)
    
    optimizer = optim.Adam(
        main_model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader) * epochs, eta_min=0, last_epoch=-1)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.57 # Initialize best accuracy
    
    # training_checkpoint = 'main_finetune_checkpoint/checkpoint_epoch_10.pth.tar'
    training_checkpoint = 'main_finetune_checkpoint/med_checkpoint_epoch_19.pth.tar'
    
    if os.path.exists(training_checkpoint):
        start_epoch = load_checkpoint(training_checkpoint, main_model, optimizer, scheduler)
    else:
        start_epoch = 0

    print('Start Training...')
    
    for epoch in range(start_epoch, epochs):
        main_model.train()
        train_loader_progress = tqdm(train_loader)
        
        for images, labels in train_loader_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = main_model.forward(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

        scheduler.step()
        
        # Validation phase
        main_model.eval()
        total_accuracy = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            total_accuracy = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = main_model.forward(images)
                accuracy = multi_label_accuracy(outputs, labels)
                total_accuracy += accuracy
                
                # Store predictions and labels
                all_predictions.append(outputs.sigmoid().cpu().numpy())  # Use sigmoid to get probabilities
                all_labels.append(labels.cpu().numpy())
                
            # Concatenate all predictions and labels
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            # Compute AUC for each label and average
            auc_list = []
            for i in range(all_labels.shape[1]):
                try:
                    auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                    auc_list.append(auc)
                except ValueError:
                    # Handle labels with single class present in the dataset
                    print('Errors')
                    pass

            avg_auc = np.mean(auc_list) if auc_list else 0.0

            avg_accuracy = total_accuracy / len(val_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {avg_accuracy:.4f}, Validation mAUC: {avg_auc:.4f}')

            # Save the model if it has the best accuracy so far
            if avg_auc > best_auc:
                best_auc = avg_auc
                #save_weights(main_model, 'best_mae_model.pth')
                save_weights(main_model, 'best_med_mae_model.pth')
                print(f"New best model saved with auc: {best_auc:.4f}")
        
        # Switch back to training mode
        main_model.train()
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': main_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),  # Save scheduler state if you are using a scheduler
    }, filename=f"main_finetune_checkpoint/med_checkpoint_epoch_{epoch+1}.pth.tar")
        
    print('Training completed.')
    
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

    
if __name__ == "__main__":
    main()
