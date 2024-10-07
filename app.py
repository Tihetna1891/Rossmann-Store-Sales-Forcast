# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the serialized model
model = joblib.load("sales_prediction_model_2023-10-07-16-30-00.pkl")  # Use your actual saved model path

# Define a data model to structure the input data
class SalesPredictionInput(BaseModel):
    store_id: int
    promo: int
    state_holiday: str
    school_holiday: int
    assortment: str
    store_type: str
    competition_distance: float
    promo2: int
    # Add other relevant fields as per your model input features

# API endpoint for home
@app.get("/")
def home():
    return {"message": "Welcome to the Sales Prediction API!"}

# API endpoint for making predictions
@app.post("/predict/")
def predict_sales(data: SalesPredictionInput):
    # Preprocess the incoming data
    input_data = np.array([[data.store_id, data.promo, data.state_holiday, data.school_holiday,
                            data.assortment, data.store_type, data.competition_distance, data.promo2]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction result
    return {"predicted_sales": prediction[0]}
