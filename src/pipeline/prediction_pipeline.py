from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            #le = LabelEncoder()
            #label_data = features.apply(le.fit_transform)
            logging.info(f'Train Dataframe Head In Logging:\n{features}')


            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info('There is an error in the prediction pipeline predict')
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, cap_surface, bruises, gill_spacing, gill_size, gill_color, stalk_surface_above_ring,
                 stalk_surface_below_ring, veil_type, ring_type, spore_print_color, population, habitat, stalk_root):
        self.cap_surface = cap_surface
        self.bruises = bruises
        self.gill_spacing = gill_spacing
        self.gill_size = gill_size
        self.gill_color = gill_color
        self.stalk_surface_above_ring = stalk_surface_above_ring
        self.stalk_surface_below_ring = stalk_surface_below_ring
        self.veil_type = veil_type
        self.ring_type = ring_type
        self.spore_print_color = spore_print_color
        self.population = population
        self.habitat = habitat
        self.stalk_root = stalk_root

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
                'stalk-surface-below-ring' : [self.stalk_surface_below_ring],
                'veil-type': [self.veil_type],
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


