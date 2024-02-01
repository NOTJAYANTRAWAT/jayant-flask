from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from flask_cors import CORS
label_encoder = LabelEncoder()
app = Flask(__name__)
CORS(app)

model = joblib.load('xgb_model.sav')
scaler = joblib.load('min_max_scaler.sav')
label_encoder_smoking_alcohol = joblib.load('label_encoder_smoking_alcohol.sav')
label_encoder_stroke_heart = joblib.load('label_encoder_stroke_heart.sav')
label_encoder_bp_chol = joblib.load('label_encoder_bp_chol.sav')
def transform_values(age,income,education,smk,alc,strk,heart,bp,chol,sex,walk):

    if smk == "Yes":
        smk = 1
    else:
        smk = 0
    
    if alc == "Yes":
        alc = 1
    else:
        alc = 0

    if strk == "Yes":
        strk = 1
    else:
        strk = 0

    if heart == "Yes":
        heart = 1
    else:
        heart = 0
    
    if bp == "Yes":
        bp = 1
    else:
        bp = 0

    if chol == "Yes":
        chol = 1
    else:
        chol = 0

    if sex == "Male":
        sex = 1
    else:
        sex = 0

    if walk == "Yes":
        walk = 1
    else:
        walk = 0
    
    if 18 <= age <= 24:
        age_category = 1
    elif 25 <= age <= 29:
        age_category = 2
    elif 30 <= age <= 34:
        age_category = 3
    elif 35 <= age <= 39:
        age_category = 4
    elif 40 <= age <= 44:
        age_category = 5
    elif 45 <= age <= 49:
        age_category = 6
    elif 50 <= age <= 54:
        age_category = 7
    elif 55 <= age <= 59:
        age_category = 8
    elif 60 <= age <= 64:
        age_category = 9
    elif 65 <= age <= 69:
        age_category = 10
    elif 70 <= age <= 74:
        age_category = 11
    elif 75 <= age <= 79:
        age_category = 12
    else:
        age_category = 13

    if income == 0:
        income_category = 0 
    elif income <= 100000:
        income_category = 1 
    elif income <= 250000:
        income_category = 2 
    elif income <= 500000:
        income_category = 3 
    elif income <= 700000:
        income_category = 4 
    elif income <= 1000000:
        income_category = 5 
    elif income <= 1800000:
        income_category = 6  
    elif income <= 3000000:
        income_category = 7
    else:
        income_category = 8

    education_map = {
        "Uneducated": 1,
        "Elementary": 2,
        "Middle School": 3,
        "High School": 4,
        "Graduate": 5,
        "Post Graduate": 6
    }
    education_category = education_map.get(education, 1)

    return age_category, income_category, education_category ,smk,alc,strk,heart,bp,chol,sex,walk

def preprocess_and_predict(education, income, gen, men, phy, bp, chol, strk, heart, smk, alc, bmi, age, walk, sex):
    age, income, education,smk,alc,strk,heart,bp,chol,sex,walk = transform_values(age, income, education,smk,alc,strk,heart,bp,chol,sex,walk)
    bp,chol,strk,heart,smk,alc = float(bp),float(chol),float(strk),float(heart),float(smk),float(alc)
    
    education_and_income = education * income
    gen_men_phy = gen + men + phy

    # Ensure the format matches the training data of the label encoder
    bp_chol = f'{bp}_{chol}'
    bp_chol_encoded = label_encoder_bp_chol.transform([bp_chol])[0]
    strk_heart = f'{strk}_{heart}'
    strk_heart_encoded = label_encoder_stroke_heart.transform([strk_heart])[0]
    smk_alc = f'{smk}_{alc}'
    smk_alc_encoded = label_encoder_smoking_alcohol.transform([smk_alc])[0]

    features = np.array([[bmi, age, education_and_income, gen_men_phy]], dtype=float)
    features_scaled = scaler.transform(features)
    final_input = np.hstack((features_scaled, [[bp_chol_encoded, strk_heart_encoded, smk_alc_encoded, walk, sex]]))
    prediction = model.predict(final_input)
    
    return prediction[0]


@app.route('/')
def hello_world():
    return 'Hello World'
    
@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Extracting data from the request
    education = data['Education']
    income = data['Income']
    gen = data['GenHlth']
    men = data['MentHlth']
    phy = data['PhysHlth']
    bp = data['HighBP']
    chol = data['HighChol']
    strk = data['Stroke']
    heart = data['HeartDiseaseorAttack']
    smk = data['Smoking']
    alc = data['HvyAlcoholConsump']
    bmi = data['BMI']
    age = data['Age']
    walk = data['DiffWalk']
    sex = data['Sex']
    prediction = preprocess_and_predict(education, income, gen, men, phy, bp, chol, strk, heart, smk, alc, bmi, age, walk, sex)
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
