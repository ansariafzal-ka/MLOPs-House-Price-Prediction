from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import sys
from src.exception import CustomException

app = FastAPI()

class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post('/predict')
def predict(house: HouseData):
    try:
        # loading the preprocessor
        with open('artifacts/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        # loading the model
        with open('artifacts/models/model.pkl', 'rb') as f:
            model = pickle.load(f)

            house_dict = house.dict()
            df = pd.DataFrame([house_dict])
            preprocessed_df = preprocessor.transform(df)
            log_prediction = model.predict(preprocessed_df)[0]
            prediction = np.exp(log_prediction)
            return {
                'prediction': float(prediction)
            }
    except Exception as e:
        raise CustomException(e, sys)