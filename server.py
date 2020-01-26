from flask import Flask, request, send_from_directory
import io
import pandas as pd
app = Flask(__name__)
# from LSTM import get_prediction, get_multi_predict
tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]
# get_prediction()
# get_multi_predict()
p = io.StringIO()
prediction = pd.read_csv('generated_data/prediction.csv', header = 0)
prediction.to_csv(p, index=False)

mp = io.StringIO()
prediction = pd.read_csv('generated_data/prediction_multi.csv', header = 0)
prediction.to_csv(mp, index=False)

data = pd.read_csv('merged_data.csv', header=0)
s = io.StringIO()
data.to_csv(s, index=False)

@app.route('/mergedData', methods=['GET'])
def get_mergedData():
    return s.getvalue()

@app.route('/predict', methods=['GET'])
def get_prediction():
    return p.getvalue()

@app.route('/predict-multi', methods=['GET'])
def get_multi_prediction():
    return mp.getvalue()

if __name__ == '__main__':
    app.run(debug=True)
