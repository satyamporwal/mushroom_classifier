from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            cap_surface=request.form.get('cap_surface'),
            bruises=request.form.get('bruises'),
            gill_spacing=request.form.get('gill_spacing'),
            gill_size=request.form.get('gill_size'),
            gill_color=request.form.get('gill_color'),
            stalk_surface_above_ring=request.form.get('stalk_surface_above_ring'),
            stalk_surface_below_ring=request.form.get('stalk_surface_below_ring'),
            ring_type=request.form.get('ring_type'),
            spore_print_color=request.form.get('spore_print_color'),
            population=request.form.get('population'),
            habitat=request.form.get('habitat'),
            stalk_root=request.form.get('stalk_root')
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()

        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(final_new_data)
            pred = predict_pipeline.predict(data_scaled, model)

            results = round(pred[0], 2)

            return render_template('results.html', final_result=results)
        except Exception as e:
            logging.info('An error occurred during prediction')
            raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
