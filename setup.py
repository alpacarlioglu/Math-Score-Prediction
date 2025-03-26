from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads the requirements file and returns a list of package requirements.
    """
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [x.replace('\n', '') for x in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements
    
setup(
    name='Math Score Prediction',
    version='1.0.0',
    author='Alp Acarlioglu',
    author_email='alpacarlioglu@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
