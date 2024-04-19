#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:55:41 2024

@author: msaqib
"""

from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import random
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

# Custom MSE loss function for distillation
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pruned_embeddings, base_embeddings):
        # Ensuring that gradients are enabled for the inputs
        pruned_embeddings.requires_grad_(True)
        base_embeddings.requires_grad_(True)

        # Computing the squared differences between pruned_embeddings and base_embeddings
        squared_diff = (pruned_embeddings - base_embeddings) ** 2
        
        # Computing the mean of the squared differences
        loss = torch.mean(squared_diff)
        
        return loss

class pruning:
    def __init__(self, num_layers, base_model):
        self.num_layers = num_layers
        
        self.base_model = base_model
        self.pruned_model = base_model
        
        # Initializing student model with teacher model
        self.optimizer = optim.Adam(self.pruned_model.parameters(), lr=0.001)
        
        self.loss_function = MSELoss()

    def pruning_model(self, num_epochs=5):
        
        # Remove layers from the student model
        auto_model = self.pruned_model._first_module().auto_model
        
        # Keep specified number of layers from the base model
        layers_to_keep = self.num_layers  
        new_layers = torch.nn.ModuleList(
            [layer_module for i, layer_module in enumerate(auto_model.encoder.layer) if i in layers_to_keep]
        )
        auto_model.encoder.layer = new_layers
        auto_model.config.num_hidden_layers = len(layers_to_keep)
        
        # Read preprocessed data
        df = pd.read_csv("processed_data.csv")
        list_data = df['tokenized_docstring'].tolist()
        training_data = list_data
        random.shuffle(training_data)
        training_data = training_data[:1000]  # Select a subset of data for training
        
        # Training loop
        for epoch in range(num_epochs):
            # Iterate over batches of training data
            for inputs in training_data:
                pruned_embeddings = self.pruned_model.encode(inputs)
                base_embeddings = self.base_model.encode(inputs)
                
                
                pruned_embeddings = torch.tensor(pruned_embeddings, dtype=torch.float32)
                base_embeddings = torch.tensor(base_embeddings, dtype=torch.float32)

                # Compute the loss
                loss = self.loss_function(pruned_embeddings, base_embeddings)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Save the distilled model
        layers = len(layers_to_keep)
        
        output_path = f"..//models/pruned_model_with_{layers}_layer"  
        
        self.pruned_model.save(output_path)
