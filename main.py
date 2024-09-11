from fastapi import FastAPI
import pickle
import pandas as pd
import joblib  # To load the label encoders
from data_model import ObesityData

app = FastAPI(
    title='Obesity Level Prediction',
    description='Classification problem'
)

# Load the trained model
with open(r'C:\Users\carva\OneDrive\Desktop\Obesity_level_MLOps\model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the pre-trained label encoders
caec_encoder = joblib.load('../CAEC_label_encoder.pkl')
calc_encoder = joblib.load('../CALC_label_encoder.pkl')
mtrans_encoder = joblib.load('../MTRANS_label_encoder.pkl')

@app.get("/")
def index():
    return 'Welcome to the obesity prediction app'

@app.post('/predict')
def model_predict(data: ObesityData):
    # Convert the input Pydantic model data into a dictionary and then into a pandas DataFrame
    sample = pd.DataFrame({
        'Gender': [data.Gender],
        'Age': [data.Age],
        'Height': [data.Height],
        #'Weight': [data.Weight],
        'family_history_with_overweight': [data.family_history_with_overweight],
        'FAVC': [data.FAVC],
        'FCVC': [data.FCVC],
        'NCP': [data.NCP],
        'CAEC': [data.CAEC],
        'SMOKE': [data.SMOKE],
        'CH2O': [data.CH2O],
        'SCC': [data.SCC],
        'FAF': [data.FAF],
        'TUE': [data.TUE],
        'CALC': [data.CALC],
        'MTRANS': [data.MTRANS]
    })

    # Preprocess binary features manually
    sample['Gender'] = sample['Gender'].map({'Male': 0, 'Female': 1})
    sample['family_history_with_overweight'] = sample['family_history_with_overweight'].map({'yes': 1, 'no': 0})
    sample['FAVC'] = sample['FAVC'].map({'yes': 1, 'no': 0})
    sample['SMOKE'] = sample['SMOKE'].map({'yes': 1, 'no': 0})
    sample['SCC'] = sample['SCC'].map({'yes': 1, 'no': 0})

    # Use the saved label encoders to encode the categorical features
    sample['CAEC'] = caec_encoder.transform(sample['CAEC'])
    sample['CALC'] = calc_encoder.transform(sample['CALC'])
    sample['MTRANS'] = mtrans_encoder.transform(sample['MTRANS'])

    # Debug: print the processed sample to ensure it's correct
    print(sample)

    # Perform prediction with the pre-trained model
    predicted_value = model.predict(sample)

    # Return the prediction result
    return {"prediction": predicted_value[0]}
