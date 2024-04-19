#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:05:27 2024

@author: msaqib
"""

import os
import pandas as pd
import numpy as np
import time

class functions:
    def __init__(self):
        pass 
    
    # Function to get the size of a model
    def get_model_size(self, model_path):
        total_size = 0
       
        # Traversing through all files in the model directory
        for dirpath, _, filenames in os.walk(model_path):
            
            for filename in filenames:
                
                file_path = os.path.join(dirpath, filename)
                
                # Adding the size of each file to the total size
                total_size += os.path.getsize(file_path)
                
        # Convert total size to MB and return
        size = f"{total_size / (1024 * 1024):.2f} MB"
        
        return size
    
    # Function to get the size of an embedding file
    def get_embedding_size(self, emb_path):
        
        # Get the size of the embedding file
        emb_size = os.path.getsize(emb_path)
        
        # Convert size to MB and return
        size = f"{emb_size / (1024 * 1024):.2f} MB"
        
        return size
    
    # Function to generate embeddings using a model and save them
    def get_model_embedding(self, model, name):
        
        # Read the preprocessed data
        df = pd.read_csv("processed_data.csv")
        
        # Get tokenized docstrings from the DataFrame
        list_data = df['tokenized_docstring'].tolist()

        # Record starting time
        starting_time = time.time()
        print(f"Starting time: {starting_time:.2f}")

        # Generate embeddings for the data
        embeddings = model.encode(list_data)

        # Record ending time
        ending_time = time.time()
        print(f"Ending time: {ending_time:.2f}")

        # Calculate time taken for embedding generation
        time_taken = ending_time - starting_time
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Total time taken for embedding generation: {time_taken : .2f} seconds ")

        # Convert embeddings to numpy array and save
        embeddings_array = np.array(embeddings)
        
        np.save(f'..//embeddings/embeddings_{name}.npy', embeddings_array)
