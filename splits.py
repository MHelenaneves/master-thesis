# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:23:24 2022

@author: mhele
"""


validation_split = .05
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = 1203
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split] #train_indices is the 95%, and val_indices is 5%