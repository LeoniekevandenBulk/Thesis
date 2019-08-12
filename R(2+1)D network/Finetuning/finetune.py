# -*- coding: utf-8 -*-
"""
Adapted code based on https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
by Leonieke van den Bulk in order to finetune the R2Plus1D network
"""

import os
import time
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VideoDataset
from Adapted_R2Plus1D_model import R2Plus1DClassifier


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

# Model to finetune
model = R2Plus1DClassifier(400, (2, 2, 2, 2), finetuned=True)
model_name = "R2Plus1D"

# Data directory. Expects the directory structure to be directory->[train/val/test]->[class labels]->[videos].
data_dir = os.path.join(os.getcwd(),"kinetics")

# Save directory
save_dir = os.path.join(os.getcwd())

# Batch size for training (change depending on how much memory you have)
batch_size = 10

# Number of epochs to train for
num_epochs = 6

# Learning rate
lr = 0.000001

# Momentum
momentum = 0.9

# Weight decay
weight_decay = 1e-4


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """Code to finetune the model further """
    since = time.time()
    
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    criterion.to(device)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device) #inputs = Variable(inputs, requires_grad=True).to(device)
                labels = labels.to(device) #Variable(labels).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    probs = nn.Softmax(dim=1)(outputs)
                    _, preds = torch.max(probs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_name = model_name + '4-' + str(epoch) + '.pt'
                torch.save(model.state_dict(), os.path.join(save_dir, save_name))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


print("Initializing Datasets and Dataloaders...")
# Create training and validation datasets
train_dataloader = DataLoader(VideoDataset(dataset_dir=data_dir, split='train',clip_len=16), batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(VideoDataset(dataset_dir=data_dir, split='val',  clip_len=16), batch_size=batch_size, num_workers=4)

# Create training and validation dataloaders
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

# Gather the parameters to be optimized/updated in this run.
params_to_update = model.parameters()

# Observe that all parameters are being optimized
optimizer = optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Use a scheduler to control the learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1) 

# Send the model to GPU
model = model.to(device)

# Train and evaluate
finetuned_model, hist = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=num_epochs)

# Save the model
print("Saving model")
save_name = model_name + ".pt"
torch.save(finetuned_model.state_dict(), os.path.join(save_dir, save_name))