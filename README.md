
# Obesity Level Prediction App

This is a machine learning web application that predicts obesity levels based on user input using a pre-trained model. The app uses FastAPI for the backend and is designed to take multiple user inputs related to health and lifestyle to classify obesity levels.

## Features

- Predicts obesity level based on input such as gender, age, height, family history of obesity, eating habits, physical activity, and more.
- Interactive web form built with HTML.
- FastAPI framework for backend logic and serving the machine learning model.
- Random Forest machine learning model for prediction.
- Deployed on a cloud platform (Render, Heroku, etc.).

## Project Structure

```bash
├── src
│   ├── main.py            # Main FastAPI app
│   ├── data_model.py      # Data model definition
│   ├── model.pkl          # Pre-trained model
│   ├── CAEC_label_encoder.pkl  # Label encoders for categorical features
│   ├── CALC_label_encoder.pkl  # Label encoders for categorical features
│   ├── MTRANS_label_encoder.pkl  # Label encoders for categorical features
│   ├── templates
│       └── form.html      # HTML template for the web form
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
```

## How It Works

1. The user inputs their personal and lifestyle data (such as age, gender, physical activity, etc.) into a form on the web app.
2. The backend receives the data, processes it using a pre-trained machine learning model, and returns a prediction of their obesity level.
3. The app provides the result back to the user, displaying it on the same page.

## Installation and Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/YourUsername/Obesity_Prediction.git
   cd Obesity_Prediction
   ```

2. **Set up a virtual environment**:

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:

   ```bash
   uvicorn src.main:app --reload
   ```

5. **Open the app in your browser**:

   Go to `http://127.0.0.1:8000` to see the form and predict obesity levels.

## Usage

- Open the application in your browser.
- Fill in the form with the required information.
- Submit the form, and the app will display your predicted obesity level.

## Deployment

### Deploying on Render/Heroku

To deploy the application on Render, Heroku, or any cloud service, follow these steps:

1. **Add your `Procfile` (for Heroku)**:

   ```
   web: uvicorn src.main:app --host 0.0.0.0 --port $PORT
   ```

2. **Push to GitHub** and link your repository to Render/Heroku.

3. **Deploy your app**, and the platform will automatically detect the `requirements.txt` and build the app.

## Technologies Used

- **FastAPI**: A modern, fast (high-performance) web framework for building APIs.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: For the machine learning model.
- **Joblib**: To save and load label encoders.
- **HTML**: Frontend for user interaction.
- **Jinja2**: Templating engine for HTML in FastAPI.

## Model Information

- The model used is a **Random Forest Classifier**, trained on a dataset containing lifestyle and personal health information.
- Pre-processing was done to encode categorical variables like `CAEC`, `CALC`, `MTRANS` using label encoders.

## Screenshots

1. **Main Form**:

   ![Obesity Level Predictor Form](images/Obesity.npg)

2. **Prediction Result**:

   ![Prediction Result](screenshot2.jpg)

## Future Improvements

- Add validation for form inputs.
- Improve UI/UX for better usability.
- Explore deploying on additional cloud platforms.
- Add more classification models for comparison.
