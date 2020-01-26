from flask import Flask, request, send_from_directory
import io
import pandas as pd
app = Flask(__name__)
from LSTM import get_prediction
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
p = io.StringIO()
prediction = pd.DataFrame(get_prediction())
prediction.to_csv(p, index=False, header = 0)
data = pd.read_csv('merged_data.csv', header=0)
s = io.StringIO()
data.to_csv(s, index=False)

@app.route('/mergedData', methods=['GET'])
def get_mergedData():
    return s.getvalue()

@app.route('/predictionData', methods=['GET'])
def get_prediction():
    return p.getvalue()

if __name__ == '__main__':
    app.run(debug=True)
