import os

class loading_data:
    def __init__(self):
        pass


    def get_file_path(self, filename, directory,directory_1=None) :
        
        current_directory = os.getcwd()
        
        # getting the parent directory of the current working directory
        parent_directory = os.path.dirname(current_directory)
        
        # Navigate to the desired directory
        if directory_1==None:
             target_directory = os.path.join(parent_directory, directory)
             
        else :     
            target_directory = os.path.join(parent_directory, directory, directory_1)
        
        file_path=os.path.join(target_directory,filename)
        
        return file_path
    
    def get_directory(self,directory,directory_1=None):
        
        current_directory = os.getcwd()
        
        # getting the parent directory of the current working directory
        parent_directory = os.path.dirname(current_directory)
        
        # Navigate to the desired directory
        if directory_1==None :
            target_directory = os.path.join(parent_directory, directory)
        else :    
            target_directory = os.path.join(parent_directory, directory,directory_1)
        
        return target_directory
    
    def save_file(self,filename, directory) :
        current_directory = os.getcwd()
        
        # getting the parent directory of the current working directory
        parent_directory = os.path.dirname(current_directory)
        
        data_directory = os.path.join(parent_directory, directory)
        
        file_name=filename
        
        saving_path=os.path.join(data_directory,file_name)
        
        return  saving_path