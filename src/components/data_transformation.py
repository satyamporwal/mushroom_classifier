import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object
from data_ingestion import DataIngestion

@dataclass
class DataTransformation:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformationInitiated:
    def __init__(self):
        self.datatransformation = DataTransformation()

    def get_datatransformation_obj(self) -> ColumnTransformer:
        try:
            logging.info('get_datatransformation_obj initiated')
            categorical_cols = ['cap-surface', 'bruises', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'veil-type', 'ring-type', 'spore-print-color', 'population', 'habitat', 'stalk-root']

            logging.info('pipeline setup')

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([('cat_pipeline', cat_pipeline, categorical_cols)])

            return preprocessor

        except Exception as e:
            logging.info('there is an error in the pipeline')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data, test_data):
        try:
            logging.info('data transformation initiated')
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            logging.info('reading data successful')
            logging.info('handling null values started')

            train_df['stalk-root'] = train_df['stalk-root'].replace('?', np.nan)
            test_df['stalk-root'] = test_df['stalk-root'].replace('?', np.nan)

            logging.info(f'Train Dataframe Null Values:\n{train_df.isnull().sum()}')
            logging.info(f'Test Dataframe Null Values:\n{test_df.isnull().sum()}')

            train_df['stalk-root_na'] = train_df['stalk-root']
            logging.info(f'Train Dataframe Null Values:\n{train_df.isnull().sum()}')
            logging.info(f'Test Dataframe Null Values:\n{test_df.isnull().sum()}')

            sampled_values = train_df['stalk-root'].dropna().sample(train_df['stalk-root_na'].isnull().sum()).values
            train_df.loc[train_df['stalk-root_na'].isnull(), 'stalk-root_na'] = sampled_values
            train_df['stalk-root'] = train_df['stalk-root_na']

            test_df['stalk-root_na'] = test_df['stalk-root']

            
            sampled_values = test_df['stalk-root'].dropna().sample(test_df['stalk-root_na'].isnull().sum()).values
            test_df.loc[test_df['stalk-root_na'].isnull(), 'stalk-root_na'] = sampled_values

            test_df['stalk-root'] = test_df['stalk-root_na']

            logging.info(f'Train Dataframe Null Values:\n{train_df.isnull().sum()}')
            logging.info(f'Test Dataframe Null Values:\n{test_df.isnull().sum()}')
            logging.info(f'Train Dataframe Head In Logging:\n{train_df.head().to_string()}')
            logging.info(f'Train Dataframe Head In Logging:\n{test_df.head().to_string()}')

            
            
            train_df = train_df.drop(columns='stalk-root_na', axis=1)
            test_df = test_df.drop(columns='stalk-root_na', axis=1)
            logging.info('handling missing data successful')
            logging.info(f'Train Dataframe Head In Logging:\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head In Logging:\n{test_df.head().to_string()}')
            train_df['class'] = train_df['class'].map({'p': 0, 'e': 1})
            test_df['class'] = test_df['class'].map({'p': 0, 'e': 1})

            

            logging.info('getting preprocessor object')
            preprocessing_obj = self.get_datatransformation_obj()

            target_column_name = 'class'
            drop_columns = [target_column_name, 'cap-shape', 'cap-color', 'odor', 'gill-attachment', 'stalk-shape', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 'ring-number']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            print(input_feature_train_df.head())
            

            le = LabelEncoder()
            input_feature_train_df = input_feature_train_df.apply(le.fit_transform)

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            logging.info(f'Test Dataframe Head In Logging:\n{test_df.info()}')

            input_feature_test_df = input_feature_test_df.apply(le.fit_transform)
            logging.info(f'Test Dataframe Head In Logging:\n{input_feature_test_df.head()}')

            ## Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.datatransformation.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (train_arr, test_arr, self.datatransformation.preprocessor_obj_file_path)

        except Exception as e:
            logging.info("Exception occurred in the initiate_data_transformation")
            raise CustomException(e, sys)

# Running the data transformation
"""if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    train_data_path = 'artifacts/train.csv'
    test_data_path = 'artifacts/test.csv'

    obj = DataTransformationInitiated()
    train_arr, test_arr, preprocessor_file_path = obj.initiate_data_transformation(train_data_path, test_data_path)
    print("Data transformation completed successfully.")
    # print("Preprocessor file saved at:", preprocessor_file_path)"""


