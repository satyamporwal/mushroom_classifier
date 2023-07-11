import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from data_ingestion import DataIngestion
from data_transformation import DataTransformation, DataTransformationInitiated
from sklearn.tree import DecisionTreeClassifier


from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            

            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet(),
                'RandomForestRegressor': RandomForestRegressor(),
                'XGBoostRegressor': XGBRegressor(),
                'decisiontree' : DecisionTreeClassifier()

            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    train_data_path = 'artifacts/train.csv'
    test_data_path = 'artifacts/test.csv'

    obj = DataTransformationInitiated()
    train_arr, test_arr, preprocessor_file_path = obj.initiate_data_transformation(train_data_path, test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)
    print("Data transformation completed successfully.")
    # print("Preprocessor file saved at:", preprocessor_file_path)
