from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
import joblib  # To load the label encoders
from data_model import ObesityData

app = FastAPI(
    title='Obesity Level Prediction',
    description='Classification problem'
)

# Set up the templates directory
templates = Jinja2Templates(directory="templates")

# Load the trained model
with open('/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the pre-trained label encoders
caec_encoder = joblib.load('../CAEC_label_encoder.pkl')
calc_encoder = joblib.load('../CALC_label_encoder.pkl')
mtrans_encoder = joblib.load('../MTRANS_label_encoder.pkl')


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Render the HTML form (form.html)
    return templates.TemplateResponse('form.html', {"request": request})


@app.post('/predict', response_class=HTMLResponse)
async def model_predict(request: Request,
                        Gender: str = Form(...),
                        Age: float = Form(...),
                        Height: float = Form(...),
                        family_history_with_overweight: str = Form(...),
                        FAVC: str = Form(...),
                        FCVC: int = Form(...),
                        NCP: float = Form(...),
                        CAEC: str = Form(...),
                        SMOKE: str = Form(...),
                        CH2O: float = Form(...),
                        SCC: str = Form(...),
                        FAF: float = Form(...),
                        TUE: float = Form(...),
                        CALC: str = Form(...),
                        MTRANS: str = Form(...)):
    
    # Collect the data into a DataFrame
    sample = pd.DataFrame({
        'Gender': [Gender],
        'Age': [Age],
        'Height': [Height],
        'family_history_with_overweight': [family_history_with_overweight],
        'FAVC': [FAVC],
        'FCVC': [FCVC],
        'NCP': [NCP],
        'CAEC': [CAEC],
        'SMOKE': [SMOKE],
        'CH2O': [CH2O],
        'SCC': [SCC],
        'FAF': [FAF],
        'TUE': [TUE],
        'CALC': [CALC],
        'MTRANS': [MTRANS]
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

    # Perform prediction with the pre-trained model
    predicted_value = model.predict(sample)

    # Return the prediction result to be shown on the web page
    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": predicted_value[0]
    })
