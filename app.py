from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI()

# Trening modelu
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Model danych do predykcji
class PredictionRequest(BaseModel):
    features: list

@app.get("/")
def read_root():
    return {"message": "API działa poprawnie!"}

@app.post("/predict")
def predict(data: PredictionRequest):
    if len(data.features) != 4:
        return {"error": "Dane muszą zawierać dokładnie 4 wartości cech."}
    
    prediction = model.predict(np.array([data.features]))
    return {"prediction": prediction.tolist()}

@app.get("/info")
def model_info():
    return {
        "model_type": "DecisionTreeClassifier",
        "features_count": iris.data.shape[1]
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

