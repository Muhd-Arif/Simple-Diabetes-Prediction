from flask import Flask, render_template, request, redirect
import pickle

app = Flask(__name__)

model = pickle.load(open('model.sav', 'rb'))
modelKNN = pickle.load(open('KNN.sav', 'rb'))
modelDT = pickle.load(open('NB.sav', 'rb'))
modelNB = pickle.load(open('DT.sav', 'rb'))

model_results = pickle.load(open('model_results.sav', 'rb'))

formVal = []


@app.route('/')
def home():
    formVal = []
    return render_template('index.html', **locals())


@app.route('/predict', methods=['POST'])
def predict():

    global formVal

    HighBP = float(request.form['HighBP'])
    HighChol = float(request.form['HighChol'])
    CholCheck = float(request.form['CholCheck'])
    BMI = float(request.form['BMI'])
    Smoker = float(request.form['Smoker'])
    Stroke = float(request.form['Stroke'])
    HeartDiseaseorAttack = float(request.form['HeartDiseaseorAttack'])
    # PhysActivity = float(request.form['PhysActivity'])
    # Fruits = float(request.form['Fruits'])
    # Veggies = float(request.form['Veggies'])
    # HvyAlcoholConsump = float(request.form['HvyAlcoholConsump'])
    # AnyHealthcare = float(request.form['AnyHealthcare'])
    # NoDocbcCost = float(request.form['NoDocbcCost'])
    GenHlth = float(request.form['GenHlth'])
    MentHlth = float(request.form['MentHlth'])
    PhysHlth = float(request.form['PhysHlth'])
    DiffWalk = float(request.form['DiffWalk'])
    # Sex = float(request.form['Sex'])
    Age = float(request.form['Age'])
    # Education = float(request.form['Education'])
    # Income = float(request.form['Income'])

    # Determine which button was clicked
    # if 'knn_button' in request.form:
    #     model = modelKNN
    #     prediction_type = 'KNN Prediction'
    # elif 'dt_button' in request.form:
    #     model = modelDT
    #     prediction_type = 'DT Prediction'
    # elif 'nb_button' in request.form:
    #     model = modelNB
    #     prediction_type = 'Naive Bayes Prediction'

    # result = model.predict([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
    #                          HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]])[0]

    formVal = [HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
               HeartDiseaseorAttack, GenHlth, MentHlth, PhysHlth, DiffWalk, Age]

    result = model.predict([formVal])[0]

    # return render_template('index.html', result=result, prediction_type=prediction_type)
    return render_template('result.html', result=result, model_results=model_results)


@app.route('/model-prediction', methods=['POST'])
def display_result():
    # Determine which button was clicked
    if 'knn_button' in request.form:
        model_type = 'knn'
        model_name = 'KNN'
    elif 'nb_button' in request.form:
        model_type = 'nb'
        model_name = 'Naive Bayes'
    elif 'dt_button' in request.form:
        model_type = 'dt'
        model_name = 'Decision Tree'

    print(formVal)
    result = model.predict([formVal])[0]

    # return render_template('index.html', result=result, prediction_type=prediction_type)
    return render_template('result_model.html', result=result, model_results=model_results, model_type=model_type, model_name=model_name)


if __name__ == '__main__':
    app.run(debug=True)
