import joblib
label_encoder_smoking_alcohol = joblib.load('label_encoder_smoking_alcohol.sav')
st = 1_0
lb = label_encoder_smoking_alcohol.transform([st])
print(lb)