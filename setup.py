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
    version='0.0.1',
    author='abdullah',
    author_email='alhasib.iu.cse@gmail.com',
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages()
)