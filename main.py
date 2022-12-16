import pickle
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from pydantic import BaseModel

model = pickle.load(open('model.sav', 'rb'))
print(type(model))
from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3000/",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {
        "test": "testing"
    }

class Item(BaseModel):
    language: float
    memory: float
    speed: float
    visual:  float
    audio: float
    survey: float

def get_result(lang_vocab, memory, speed, visual, audio, survey):
    #2D numpy array created with the values input by the user.
    array = np.array([[lang_vocab, memory, speed, visual, audio, survey]])
    #The output given by model is converted into an int and stored in label.
    label = int(model.predict(array))
    #Giving final output to user depending upon the model prediction.
    if(label == 0):
        output = "There is a high chance of the applicant to have dyslexia."
    elif(label == 1):
        output = "There is a moderate chance of the applicant to have dyslexia."
    else:
        output = "There is a low chance of the applicant to have dyslexia."
    return output

@app.post("/result")
def result(scores:Item):

    return get_result(scores.language, scores.memory, scores.speed, scores.visual, scores.audio, scores.survey)
    


