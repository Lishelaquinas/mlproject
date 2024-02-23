import os
import sys
from dataclasses import dataclass
from src.utils import evaluate_model,saveobject
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import saveobject

@dataclass
class ModelTrainerConfig:
    #create a pickle file for the model.
    trainedModelFilePath = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modelTrainerConfig = ModelTrainerConfig()
    
    def initiateModelTrainer(self, train_array, test_array, preprocessorPath):
        try:
            logging.info("Splitting the train and test data into Input Features and target")

            Xtrain, ytrain, Xtest, ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            #create  a dictionary of Models that we will be using the project.

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            #create a report that has the various model.
            modelReport:dict = evaluate_model(Xtrain = Xtrain, ytrain = ytrain,Xtest=Xtest, ytest=ytest,models = models)

            #Get the best model score.
            bestModelScore = max(sorted(modelReport.values()))

            # Get the best model Name.
            bestModelName = list(modelReport.keys())[list(modelReport.values()).index(bestModelScore)]

            bestModel = models[bestModelName]

            if bestModelScore < 0.6:
                raise CustomException ("No best Model found", sys)
            logging.info("Best model found.")

            saveobject(
                self.modelTrainerConfig.trainedModelFilePath,
                bestModel
                )
            
            predicted = bestModel.predict(Xtest)
            score = r2_score(ytest,predicted)
            return score
        
        except Exception as e:
            raise CustomException(e, sys)