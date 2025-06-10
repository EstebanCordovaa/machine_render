from .utils import db_connect
engine = db_connect()

# your code here
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('models/iris_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['sepal length (cm)']),
                float(request.form['sepal width (cm)']),
                float(request.form['petal length (cm)']),
                float(request.form['petal width (cm)'])
            ]
            pred = model.predict([features])[0]
            clases = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
            prediction = clases[pred]
        except Exception as e:
            prediction = f"Error en la predicción: {e}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)