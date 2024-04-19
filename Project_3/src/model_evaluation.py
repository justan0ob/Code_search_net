#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:53:36 2024

@author: msaqib
"""
import pandas as pd
import numpy as np
import torch
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


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
    
    def get_top_code_and_docstring(self , question, embedding, model,df):
    
        # getting the vector embedding of question
        questions_embeddings = model.encode(question)
    
        # Calculate cosine similarities between the question embedding and all embeddings in the dataset
        cosine_similarities = np.array([self.cosine_similarity(questions_embeddings, emb) for emb in embedding if self.cosine_similarity(questions_embeddings, emb)>=0.5])
        
        # Get the indices of the top 10 cosine similarities
        top_10_indices = np.argsort(cosine_similarities)[-10:][::-1]
    
        # Initialize a list to store the top 10 docstrings
        top_10_docstring = []
        
        # Initialize a list to store the top 10 code snippets
        top_code = []
    
        # Retrieve the docstrings corresponding to the top 10 indices
        for i in top_10_indices:
            top_10_docstring.append(df.loc[i].docstring)
            top_code.append(df.loc[i].code)
            
        # Create a DataFrame to represent the table
        table_df = pd.DataFrame({'Docstrings': top_10_docstring, 'Code': top_code})

        return table_df
    
    def add_column(self,df):
    
    # Add a new column named 'Query' with the same value for all rows
        df['Match']=None
    
        return df
    
    
    def check_response(self,Questions, top_match_code):
    
        # converting the top_match_code in the dataframe
        data=top_match_code
    
    
        # Questions is the response from the human for the claude
        human =Questions
    
        # Initialize the ChatAnthropic object
        chat = ChatAnthropic(anthropic_api_key="Your APi key " ,temperature=0, model_name="claude-3-haiku-20240307")

        # Defining system message with task description and data
        system = (
            """ Your task is to provide a response of only 'YES' if there is a 75 percentage matching of code for
            the human input and the data,
            or only 'No' if there isn't,
            when comparing the data to human input. Do not generate unnecessary response give 'YES' or 'NO' only.
        
        data: {data}
        human: {human}
        """
        )
    
        # Creating the ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human",human)])

        # Creating the chain combining prompt and chat
        chain = prompt | chat
    
        # Invoking the chain with data and human input
        response=chain.invoke(
            {
                "data": data,
                "human": human,
                }
            )
        return response
    
        
        
        
