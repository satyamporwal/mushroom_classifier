from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = r"artifacts\\preprocessor.pkl"
            model_path = r"artifacts\\model.pkl"
            #label_data_path = r"artifacts\\label_encoder.pkl"
            #print("This is path @@\n",model_path)
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            #label_data= load_object(label_data_path)
            #le = LabelEncoder()
            
            #l#ogging.info(f'Train Dataframe Head In Logging:\n{label_data}')

            #data_labelled = label_data.fit_transform(features)
            #data= label_data.transform(features)
            data_scaled = preprocessor.transform(features)
            for i in range(data_scaled.shape[1]):
                 unique_values = np.unique(data_scaled[:, i])
                 print(f"Unique values in column {i}: {unique_values}")


            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info('There is an error in the prediction pipeline predict')
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, cap_surface, bruises, gill_spacing, gill_size, gill_color, stalk_surface_above_ring, stalk_root,
                ring_type, spore_print_color, population, habitat, ):
        self.cap_surface = cap_surface
        self.bruises = bruises
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_root = stalk_root
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat
    

    def get_data_as_dataframe(self):
        try:
            logging.info('Converting into dataframe as started')
            custom_data_input_dict = {
                'cap-surface': [self.cap_surface],
                'bruises': [self.bruises],
                'gill-spacing': [self.gill_spacing],  # Convert to float
                'gill-size': [self.gill_size],
                'gill-color': [self.gill_color],
                'stalk-root': [self.stalk_root],
                'stalk-surface-above-ring': [self.stalk_surface_above_ring],
                'ring-type': [self.ring_type],
                'spore-print-color': [self.spore_print_color],
                'population': [self.population],
                'habitat': [self.habitat],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)


