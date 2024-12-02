from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from typing import List

# Inicializando a aplicação FastAPI
app = FastAPI()

# Carregar o modelo e o scaler previamente treinados
model = load_model('/modelo/lstm_model.h5')
scaler = joblib.load('/modelo/scaler.pkl')

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

# Função para fazer previsões
def predict_prices(data, time_step=60):
    # Normalizar os dados de entrada
    scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
    # Criar sequência para a previsão
    X_input = scaled_data[-time_step:].reshape(1, time_step, 1)
    predicted_scaled = model.predict(X_input)
    # Desnormalizar a previsão
    predicted_price = scaler.inverse_transform(predicted_scaled)
    return predicted_price[0][0]

# Modelo para receber os dados
class PriceData(BaseModel):
    data: List[float]

# Endpoint para receber dados e fazer previsões
@app.post("/predict")
async def predict(price_data: PriceData):
    try:
        # Fazer a previsão
        predicted_price = predict_prices(price_data.data)

        # Retornar a previsão
        return {"predicted_price": predicted_price}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
