#importing all necessary library

import pandas as pd
import nltk
import re

from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# creating class for the data_preprocessor
class data_preprocessor:
    
    def __init__(self):
        pass
       
    def preprocessing_text(self , text):
        text = text
        
        # Make Lower case all character in text
        text = text.lower()   
        
        # Removes non alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)                             
        
        # Tokenizing words
        tokens = word_tokenize(text)
        
        # Obtaining stop words like ‘a’, ‘the’ etc.
        stop_words = set(stopwords.words('english')) 
        
        # Removing stop words lie (a, the, is)
        tokens = [word for word in tokens if word not in stop_words]    
        
        # Obtaining base of words like ‘happy’ and not ‘happiest’
        lemmatizer = WordNetLemmatizer()
        
        # lemmatizing each word.
        tokens = [lemmatizer.lemmatize(word) for word in tokens]        
        
        # Joining and returning
        preprocessed_text = ' '.join(tokens)                            
        
        return preprocessed_text
    
    
    def detect_language(self, text):
        text = text
        
        # Function to detect the language of the given text
    
        try:
            # Attempt to detect the language
            
            lang = detect(text)
            
            return lang
        
        except:
            
            # Return None if an exception occurs (e.g., unsupported language)
            return None
    
            
    
