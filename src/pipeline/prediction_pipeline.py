from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd


class PredictionPipeline:
    def predict(self):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(preprocessor)
            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info('There is an error in the prediction pipeline predict')
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        cap_surface,
        bruises,
        gill_spacing,
        gill_size,
        gill_color,
        stalk_surface_above_ring,
        stalk_surface_below_ring,
        veil_type,
        ring_type,
        spore_print_color,
        population,
        habitat,
        stalk_root,
    ):
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
                'cap_surface': [self.cap_surface],
                'bruises': [self.bruises],
                'gill_spacing': [self.gill_spacing],
                'gill_size': [self.gill_size],
                'gill_color': [self.gill_color],
                'stalk_surface_above_ring': [self.stalk_surface_above_ring],
                'stalk_surface_below_ring': [self.stalk_surface_below_ring],
                'ring_type': [self.ring_type],
                'spore_print_color': [self.spore_print_color],
                'population': [self.population],
                'habitat': [self.habitat],
                'stalk_root': [self.stalk_root],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)
