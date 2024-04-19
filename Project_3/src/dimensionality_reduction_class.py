#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:13:37 2024

@author: msaqib
"""

# Importing all the necessary libraries
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, models
import random
import numpy as np
import pandas as pd
import torch

class dimensionality_reduction:
    def __init__(self, model_name, new_dimension):
        """
        Constructor for dimensionality_reduction class.
        
        Parameters:
        - model_name: Name of the SentenceTransformer model to be used.
        - new_dimension: New dimensionality after dimensionality reduction.
        """
        self.model_name = model_name
        
        self.new_dimension = new_dimension
        
        # Initializing SentenceTransformer model
        self.model = SentenceTransformer(model_name)
        
        # Reading preprocessed data
        self.df = pd.read_csv("processed_data.csv")   
    
    def reduce_dimension(self):
        """
        Reducing the dimensionality of the SentenceTransformer model using PCA.
        """
        # Extract tokenized docstrings
        list_data = self.df['tokenized_docstring'].tolist() 
        
         # Shuffle the data
        random.shuffle(list_data) 
        
        # Selecting a subset of data for PCA training
        pca_train = list_data[:2000]  
        
        # Encoding the training data using the SentenceTransformer model
        train_embeddings = self.model.encode(pca_train, convert_to_numpy=True)
        
        # Performing the PCA for dimensionality reduction
        pca = PCA(n_components=self.new_dimension)
        pca.fit(train_embeddings)
        pca_comp = np.asarray(pca.components_)
        
        # Creating a Dense layer for mapping to the new dimension
        dense = models.Dense(
            in_features=self.model.get_sentence_embedding_dimension(),
            out_features=self.new_dimension,
            bias=False,
            activation_function=torch.nn.Identity(),
        )
        
        # Set weights of the Dense layer with PCA components
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
        
        # Adding the Dense layer to the model
        self.model.add_module("dense", dense)
        
        # Saving the modified model
        self.model.save(f'..//models/new_{self.new_dimension}_dim_model')
