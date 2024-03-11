import sys
import os
import dill
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.exception import CustomException

def saveobject(filePath, object):
    try:
        directoryPath = os.path.dirname(filePath)
        os.makedirs(directoryPath, exist_ok=True)

        with open (filePath ,"wb") as fileObject:
            dill.dump(object, fileObject)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(Xtrain,ytrain,Xtest,ytest,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            gs = GridSearchCV(model, para , cv=3)
            gs.fit(Xtrain, ytrain)

            model.set_params(**gs.best_params_)

            model.fit(Xtrain, ytrain)
            ytrainPredict = model.predict(Xtrain)
            ytestPredict = model.predict(Xtest)
            trainModelScore = r2_score(ytrain,ytrainPredict)
            testModelScore = r2_score(ytest,ytestPredict)
            report[list(models.keys())[i]] = testModelScore
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        print("Inside load object function")
        with open (file_path, "rb") as fileObject:
            return dill.load(fileObject)
    
    except Exception as e:
        raise CustomException(e, sys)


   

    