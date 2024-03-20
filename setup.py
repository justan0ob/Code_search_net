from setuptools import setup


with open('requirements.txt') as req:
    
    requirements = req.read().splitlines()
    
setup(
   name='Code_Search_Net',
   version='1.0',
   python_requires='>=3.11',
   packages=requirements,
   author='Mohd saqib',
   description = 'For a particular query searcheing whether the code for this query is present or not in our dataset')
