import pandas as pd
import numpy as np
import torch


class evaluation:
    
    def __init__(self):
        pass
    
    def cosine_similarity(self , vec1, vec2):
    
        # Calculating the dot product of the two vectors
        dot_product = np.dot(vec1, vec2)
    
        # Calculating the Euclidean norm of the first vector
        norm1 = np.linalg.norm(vec1)
    
        # Calculating the Euclidean norm of the second vector
        norm2 = np.linalg.norm(vec2)
    
        # Compute the cosine similarity by dividing the dot product by the product of the norms
        return dot_product / (norm1 * norm2)
    
    def get_top_code_and_docstring(self , question, embedding, model,list_data,df):
    
        # getting the vector embedding of question
        questions_embeddings = model.encode(question)
    
        # Calculate cosine similarities between the question embedding and all embeddings in the dataset
        cosine_similarities = np.array([self.cosine_similarity(questions_embeddings, emb) for emb in embedding])
    
        # Get the indices of the top 10 cosine similarities
        top_10_indices = np.argsort(cosine_similarities)[-10:][::-1]
    
        # Initialize a list to store the top 10 docstrings
        top_10_docstring = []
        
        # Initialize a list to store the top 10 code snippets
        top_code = []
    
        # Retrieve the docstrings corresponding to the top 10 indices
        for i in top_10_indices:
            top_10_docstring.append(list_data[i])
            top_code.append(df.loc[i].code)
            
        # Create a DataFrame to represent the table
        table_df = pd.DataFrame({'Docstrings': top_10_docstring, 'Code': top_code})

        return table_df
        
        
        
