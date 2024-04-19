# importing all the necessary libraries
import random
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import models, SentenceTransformer
from sklearn.decomposition import PCA

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, student_embeddings, teacher_embeddings):
        # Ensuring that gradients are enabled for the inputs
        student_embeddings.requires_grad_(True)
        teacher_embeddings.requires_grad_(True)

        # Computing the squared differences between student_embeddings and teacher_embeddings
        squared_diff = (student_embeddings - teacher_embeddings) ** 2
        
        # Computing the mean of the squared differences
        loss = torch.mean(squared_diff)
        
        return loss

class KnowledgeDistillation:
    def __init__(self, teacher_model_name, student_model_name):
        """
        Constructor for KnowledgeDistillation class.

        Parameters:
        - teacher_model_name: Name of the teacher SentenceTransformer model.
        - student_model_name: Name of the student SentenceTransformer model.
        """
        
        # Initialize teacher model
        self.teacher_model = SentenceTransformer(teacher_model_name)
        
        # Initialize student model
        self.student_model = SentenceTransformer(student_model_name)
        
        # Initialize MSE loss function
        self.loss_function = MSELoss()  
        
         # Adam optimizer for student model
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=0.001) 

    def train_student_model(self, num_epochs=5):
        """
        Train the student model using knowledge distillation.

        Parameters:
        - num_epochs: Number of training epochs.
        """
        
        # Reading preprocessed data
        df = pd.read_csv("processed_data.csv") 
        
         # Extracting tokenized docstrings
        list_data = df['tokenized_docstring'].tolist() 
        
        training_data = list_data
        
        random.shuffle(training_data)
        
         # Selecting a subset of data for training
        training_data = training_data[:2000] 
        
        # Performing dimensionality reduction if student dimension < teacher dimension
        if self.student_model.get_sentence_embedding_dimension() < self.teacher_model.get_sentence_embedding_dimension():
            pca_sentences = training_data
            pca_embeddings = self.teacher_model.encode(pca_sentences, convert_to_numpy=True)
            new_dimension = self.student_model.get_sentence_embedding_dimension()
            pca = PCA(n_components=new_dimension)
            pca.fit(pca_embeddings)
            pca_comp = np.asarray(pca.components_)
            
            # Adding Dense layer to teacher model to project embeddings to student dimension
            dense = models.Dense(
                in_features=self.teacher_model.get_sentence_embedding_dimension(),
                out_features=self.student_model.get_sentence_embedding_dimension(),
                bias=False,
                activation_function=torch.nn.Identity(),
            )
            
            dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
            self.teacher_model.add_module("dense", dense)
        
        for epoch in range(num_epochs):
            
            
            for inputs in training_data:
                student_embeddings = self.student_model.encode(inputs)
                teacher_embeddings = self.teacher_model.encode(inputs)
                student_embeddings = torch.tensor(student_embeddings, dtype=torch.float32)
                teacher_embeddings = torch.tensor(teacher_embeddings, dtype=torch.float32)

                # Computing the loss
                loss = self.loss_function(student_embeddings, teacher_embeddings)

                # Backpropagation
                self.optimizer.zero_grad()
                
                loss.backward()
                
                self.optimizer.step()
                
         # Pathwhere to save the model 
        output_path = "..//models/distiiled_model" 
        
        # saving the distillated model
        self.student_model.save(output_path)
