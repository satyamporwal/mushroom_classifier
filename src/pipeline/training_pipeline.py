import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformationInitiated
from src.components.model_trainer import ModelTrainer


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
   