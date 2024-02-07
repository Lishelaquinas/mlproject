import os
import sys
# import our exception from src
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#This class will contain all the configuration related to saving retriving the data.
@dataclass
class DataIngestionConfig:
    #no giving init. it will be automatically called because of data class
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")

class DataIngestion:
    # when i call this class the three variables from dataingestionConfig will be saved in these three variables.
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()
    
    def initiate_DataIngestion(self):
        logging.info("Entered data ingestion component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Reading from stud.csv in data frame.')

            os.makedirs(os.path.dirname(self.ingestionConfig.test_data_path), exist_ok=True)

            #saving the data in raw data path 
            df.to_csv(self.ingestionConfig.raw_data_path, index=False, header=True)

            logging.info("Train test split initated")
            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestionConfig.train_data_path, index=False, header = True)
            test_set.to_csv(self.ingestionConfig.test_data_path, index=False, header = True)

            logging.info("Ingestion of data completed")
            return (
                self.ingestionConfig.test_data_path,
                self.ingestionConfig.train_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    object = DataIngestion()
    object.initiate_DataIngestion()



                                        
    