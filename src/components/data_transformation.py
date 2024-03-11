import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer  #used to create pipeline.
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import saveobject

@dataclass

class DataTransformationConfig:
    preprocessorObjectFile = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        '''
        This function is responsible for data transformation.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )


            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)        
    def initiateDataTransformation(self, train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read test and train data.")
            logging.info("Obtaining pre processing info.")

            preprocessingObject = self.get_data_transformation_object()

            targetColumn = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            #Create input feature and target featire for test and train.
            input_feature_train_df = train_df.drop(columns=[targetColumn],axis=1)
            target_feature_train_df = train_df[targetColumn]

            input_feature_test_df = test_df.drop(columns=[targetColumn],axis=1)
            target_feature_test_df = test_df[targetColumn]

            logging.info("Applying Preprocessing object on test and train data.")

            #Apply preprocessing object on train and test input feature.

            inputFeatureTrain = preprocessingObject.fit_transform(input_feature_train_df)
            inputFeatureTest = preprocessingObject.transform(input_feature_test_df)

            train_arr = np.c_[inputFeatureTrain, np.array(target_feature_train_df)]
            test_arr = np.c_[inputFeatureTest, np.array(target_feature_test_df)]

            logging.info("Save object file")
            
            saveobject(
                filePath = self.datatransformationconfig.preprocessorObjectFile,
                object = preprocessingObject
            )

            return(
                train_arr,
                test_arr,
                self.datatransformationconfig.preprocessorObjectFile
            )
        except Exception as e:
            raise CustomException(e, sys)
            