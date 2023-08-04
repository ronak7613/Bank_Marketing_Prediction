import sklearn

print("Scikit-Learn Version:", sklearn.__version__)

import pandas as pd
import numpy as np
from flask import Flask, request, Response, render_template
#import pickle
import json
import joblib
app = Flask(__name__)

#model = pickle.load(open("G:/pu sallybus and releted stuff/coe/week_9/Bank marketing respons/ens_model.sav", "rb"))
sup_model = joblib.load('random_search_model.sav')
unsup_model = joblib.load('agglo_clust_model.sav')


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Int64Dtype):
            return int(obj)
        return super().default(obj)
    
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    json_data = request.json
    print(json_data)
    query_df = pd.DataFrame([json_data])
    query_df.set_index('ID')
    prediction = sup_model.predict(query_df)
    prediction_list = prediction.tolist()
    response_data = {"Prediction": prediction_list}
    #response = Response(json.dumps(response_data, cls=CustomJSONEncoder), content_type="application/json")
    return render_template("result.html", prediction_result=response_data)

@app.route("/cluster", methods=["POST"])
def cluster():
    json_data = request.json
    cquery_df = pd.DataFrame(json_data)
    
    cluster_labels = unsup_model.fit_predict(cquery_df)
    response_data = {"ClusterLabels": cluster_labels.tolist()}
    
    return Response(json.dumps(response_data), content_type="application/json")


if __name__ == "__main__":
    app.run(debug=True)




# Column: job
# Categories: ['admin.' 'blue-collar' 'entrepreneur' 'housemaid' 'management' 'retired'
#  'self-employed' 'services' 'student' 'technician' 'unemployed' 'unknown']
# Encoded labels: [ 0  1  2  3  4  5  6  7  8  9 10 11]
# -----------------------------------
# Column: marital
# Categories: ['divorced' 'married' 'single']
# Encoded labels: [0 1 2]
# -----------------------------------
# Column: education
# Categories: ['primary' 'secondary' 'tertiary' 'unknown']
# Encoded labels: [0 1 2 3]
# -----------------------------------
# Column: default
# Categories: ['no' 'yes']
# Encoded labels: [0 1]
# -----------------------------------
# Column: housing
# Categories: ['no' 'yes']
# Encoded labels: [0 1]
# -----------------------------------
# Column: loan
# Categories: ['no' 'yes']
# Encoded labels: [0 1]
# -----------------------------------
# Column: contact
# Categories: ['cellular' 'telephone' 'unknown']
# Encoded labels: [0 1 2]
# -----------------------------------
# Column: month
# Categories: ['apr' 'aug' 'dec' 'feb' 'jan' 'jul' 'jun' 'mar' 'may' 'nov' 'oct' 'sep']
# Encoded labels: [ 0  1  2  3  4  5  6  7  8  9 10 11]
# -----------------------------------
# Column: poutcome
# Categories: ['failure' 'other' 'success' 'unknown']
# Encoded labels: [0 1 2 3]
# -----------------------------------

# ID           13689
# age             41
# job              9
# marital          1
# education        3
# default          0
# balance         30
# housing          1
# loan             0
# contact          0
# day             10
# month            5
# campaign         1
# pdays           -1
# previous         0
# poutcome         3
# Name: 3, dtype: int64



