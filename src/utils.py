import sys
import os
import dill
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from src.exception import CustomException

def saveobject(filePath, object):
    try:
        directoryPath = os.path.dirname(filePath)
        os.makedirs(directoryPath, exist_ok=True)

        with open (filePath ,"wb") as fileObject:
            dill.dump(object, fileObject)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(Xtrain,ytrain,Xtest,ytest,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(Xtrain, ytrain)
            ytrainPredict = model.predict(Xtrain)
            ytestPredict = model.predict(Xtest)
            trainModelScore = r2_score(ytrain,ytrainPredict)
            testModelScore = r2_score(ytest,ytestPredict)
            report[list(models.keys())[i]] = testModelScore
        return report

    except Exception as e:
        raise CustomException(e, sys)
   

    