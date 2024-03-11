import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipleline:
    def __init__(self):
        pass
    def predict(self, features):
        # load the preprocessor object and model object
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model = load_object(file_path = model_path)
            print("Model loaded sucessfully")
            preprocessor  = load_object(file_path = preprocessor_path)
            print("preprocessor loaded sucessfully")
            dataScaled  = preprocessor.transform(features)
            predictions = model.predict(dataScaled) 
            return predictions
        except Exception as e:
            raise CustomException(e,sys)



#This class will be responsible for mapping the fields to the test data.
class CustomData:
    def __init__(self, 
                 gender: str,
                 race_ethnicity:str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score,
                 writing_score):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    #Function returns the data in the form of a dataFrame.
        
    def dataAsDataFrame(self):
        try:
            customDatainput = {
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score]
                }
            return pd.DataFrame(customDatainput)
        except Exception as e:
            raise CustomException(e,sys)
        



