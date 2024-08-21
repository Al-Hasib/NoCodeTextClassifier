from setuptools import find_packages, setup
from typing import List


# get the requirements into list

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements



'''
def get_requirements_chatgpt(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    
    # Strip whitespace and newline characters from each line
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('-e')]
    
    return requirements

'''


setup(
    name='NoCodeTextClassifier',
    version='0.0.4',
    author='abdullah',
    author_email='alhasib.iu.cse@gmail.com',
    description="This package is for Text Classification of NLP Task",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Al-Hasib/NoCodeTextClassifier",
    install_requires=["pandas","scikit-learn","matplotlib","seaborn","pathlib","nltk","xgboost"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  # Additional metadata
    python_requires='>=3.8',  # Minimum Python version required 
)

