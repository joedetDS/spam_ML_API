from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

vectorizer = joblib.load("spam_vect.sav")
spam_model=joblib.load("spam-detector.sav")

app = FastAPI()

#set up the CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"]
    )

class ModelInput(BaseModel):
    text: str


#Create the end point
@app.post("/spam_detection")
def spam_detection(input_parameters: ModelInput):
    
    input_mail = input_parameters.text
    #Preprocess the entered mail
    v_mail=vectorizer.transform([input_mail])
    
    #Predict using the loaded model
    prediction= spam_model.predict(v_mail)
    
    #Check if mail is spam or non-spam
    if prediction[0]==0:
        result = "Non-Spam"
    else:
        result = "Spam"
    return result.strip('"')
