from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('KNN.sav', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    BMI = request['BMI']
    HighChol = request['HighChol']
    Smoker = request['Smoker']
    result = model.predict([[BMI, HighChol, Smoker]])[0]
    return render_template('index.html', **locals)


if __name__ == '__main__':
    app.run(debug=True)
