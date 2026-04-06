from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.pred_pipeline import input_data, Pred_Pipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = input_data(
                Age=int(request.form.get('Age', 0)),
                Income=float(request.form.get('Income', 0.0)),
                Home=request.form.get('Home', ''),
                Emp_length=float(request.form.get('Emp_length', 0.0)),
                Intent=request.form.get('Intent', ''),
                Amount=float(request.form.get('Amount', 0.0)),
                Rate=float(request.form.get('Rate', 0.0)),
                Status=int(request.form.get('Status', 0)),
                Percent_income=float(request.form.get('Percent_income', 0.0)),
                Cred_length=int(request.form.get('Cred_length', 0))
            )
            pred_data = data.transfrom_data_as_dataframe()
            print(pred_data)
            print("Before Prediction")

            predict_pipeline = Pred_Pipeline()
            print("During Prediction")
            results, probability = predict_pipeline.predict(pred_data)
            print("After Prediction")
             
            if results == 1:
                message = f"There are high chances of risk (probability: {probability:.2f}). Immediate attention required."
            else:
                message = f"There are low chances of risk (probability: {1 - probability:.2f})."
            
            return render_template('home.html', results=message)
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"<h1>Error: {str(e)}</h1><p>{traceback.format_exc()}</p>", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8083, debug=False)
