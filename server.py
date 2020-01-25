from flask import Flask, request, send_from_directory
import io
import pandas as pd
app = Flask(__name__)

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

data = pd.read_csv('merged_data.csv', header=0)
s = io.StringIO()
data.to_csv(s, index=False)
# print(s.getvalue())

@app.route('/mergedData', methods=['GET'])
def get_tasks():
    return s.getvalue()


if __name__ == '__main__':
    app.run(debug=True)
