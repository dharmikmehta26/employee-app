from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json
import csv


rfc = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():

    return render_template('index.html')


def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input =np.zeros(18)
    # set the numerical input as they are
    enc_input[0] = data['avg_monthly_hrs']
    enc_input[1] = data['last_evaluation']
    enc_input[2] = data['n_projects']
    enc_input[3] = data['salary']
    enc_input[4] = data['satisfaction']
    enc_input[5] = data['tenure']

    departments = ['IT', 'admin', 'engineering', 'finance',
    'information_technology', 'management', 'marketing', 'procurement',
    'product', 'sales', 'support', 'temp']

    cols = ['avg_monthly_hrs', 'last_evaluation', 'n_projects', 'salary',
       'satisfaction', 'tenure', 'IT', 'admin', 'engineering', 'finance',
       'information_technology', 'management', 'marketing', 'procurement',
       'product', 'sales', 'support', 'temp']

    # redefine the the user inout to match the column name
    redefinded_user_input = data['department']
    # search for the index in columns name list
    department_column_index = cols.index(redefinded_user_input)
    #print(mark_column_index)
    # fullfill the found index with 1
    enc_input[department_column_index] = 1


    return enc_input

@app.route('/api',methods=['POST'])
def get_delay():
    result=request.form
    avg_monthly_hrs = result['avg_monthly_hrs']
    department = result['department']
    last_evaluation = result['last_evaluation']
    #mark = result['mark']
    n_projects = result['n_projects']
    salary = result['salary']
    satisfaction = result['satisfaction']
    tenure = result['tenure']
    user_input = {'avg_monthly_hrs':avg_monthly_hrs,'last_evaluation':last_evaluation, 'n_projects':n_projects, 'salary':salary,'satisfaction':satisfaction,'tenure':tenure,'department':department}


    #print(user_input)
    a = input_to_one_hot(user_input)
    pred =rfc.predict([a])[0]
    #pred = round(pred, 2)
    """
    def default(o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError
        """
    prob_pred = rfc.predict_proba([a]).max()
    stats = 'Leave the company' if pred == 0 else 'stay in the company'


    fieldnames = ['avg_monthly_hrs','department','last_evaluation', 'n_projects', 'salary','satisfaction','tenure','status']


    with open('namelist.csv','a') as inFile:
            writer = csv.DictWriter(inFile, fieldnames=fieldnames)
            writer.writerow({'avg_monthly_hrs': avg_monthly_hrs, 'department': department,'last_evaluation':last_evaluation, 'n_projects':n_projects, 'salary':salary,'satisfaction':satisfaction,'tenure':tenure,'status':stats})


    return json.dumps({'status':stats,'prob_pred':prob_pred});
    # return render_template('result.html',prediction=price_pred)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
