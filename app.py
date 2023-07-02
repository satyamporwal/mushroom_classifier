from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
            cap_surface=request.form.get('Cap-Surface'),
            bruises=request.form.get('Bruises'),
            gill_spacing=request.form.get('Gill-Spacing'),
            gill_size=request.form.get('Gill-Size'),
            gill_color=request.form.get('Gill-Color'),
            stalk_root=request.form.get('Stalk-Root'),
            stalk_surface_above_ring=request.form.get('Stalk-Surface-Above-Ring'),
            stalk_surface_below_ring=request.form.get('Stalk-Surface-below-Ring'),
            veil_type=request.form.get('Veil-Type'),
            ring_type=request.form.get('Ring-Type'),
            spore_print_color=request.form.get('Spore-Print-Color'),
            population=request.form.get('Population'),
            habitat=request.form.get('Habitat')
        )

        final_new_data = data.get_data_as_dataframe()
        
        
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)
        results = round(pred[0], 2)

        return render_template('result.html', final_result=results)
      

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
