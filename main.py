from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 1. Charger le modèle UNE SEULE FOIS au démarrage
model = joblib.load("regression.joblib")

# 2. Créer l'app FastAPI
app = FastAPI()

# 3. Définir le schéma des données d'entrée
class HouseFeatures(BaseModel):
    size: float
    nb_rooms: int
    garden: int

# 4. Endpoint GET (on le garde pour tester rapidement)
@app.get("/predict")
def predict_get():
    return {"y_pred": 2}

# 5. Endpoint POST qui utilise le modèle
@app.post("/predict")
def predict_post(features: HouseFeatures):
    # Mettre les features dans le bon format (liste de listes)
    X = [[features.size, features.nb_rooms, features.garden]]
    # Faire la prédiction
    y_pred = model.predict(X)
    # Retourner le résultat en JSON
    return {"y_pred": float(y_pred[0])}