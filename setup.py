from setuptools import find_packages, setup
from typing import List


'''
This function will read all the packages from requirements.txt and return it as a list
'''

HYPE_E_DOT = '-e .'
def getRequirements(fileName: str)->List[str]:
    requirements = []
    with open(fileName) as fileObject:
        requirements = fileObject.readlines()
        [req.replace('\n','') for  req in requirements]

        if HYPE_E_DOT in requirements:
            requirements.remove(HYPE_E_DOT)
    print(requirements)
    return requirements

setup(
    name = 'mlproject', 
    version = '0.0.1',
    author = 'Lishel Aquinas',
    author_email = 'lishelaquinas@gmail.com',
    packages = find_packages(),
    #install_requires = ['pandas','numpy','seaborn']
    install_requires = getRequirements('requirements.txt')

)