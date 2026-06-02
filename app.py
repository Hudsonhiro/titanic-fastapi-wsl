from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
from typing import Literal

app = FastAPI(
    title="API de Predição do Titanic",
    version="2.0.0"
)

# Carrega o modelo atualizado
try:
    modelo = joblib.load('modelo.pkl')
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")

# Define o novo formato dos dados
class Passageiro(BaseModel):
    Pclass: int
    Sex: Literal["male", "female"]  # Agora a API só aceita por extenso!
    Age: float
    Fare: float                     # Nova coluna adicionada na API

@app.get("/")
def home():
    return {"status": "Online", "versao": "2.0.0"}

@app.post("/predict")
def prever_sobrevivencia(passageiro: Passageiro):
    # Converte o texto recebido ("male"/"female") para o número que a IA entende (0/1)
    sexo_numerico = 1 if passageiro.Sex == "female" else 0
    
    # Organiza os dados na ordem exata do main.py: Pclass, Sex, Age, Fare
    dados_entrada = [[
        passageiro.Pclass, 
        sexo_numerico, 
        passageiro.Age, 
        passageiro.Fare
    ]]
    
    # Faz a predição e calcula a probabilidade
    predicao = modelo.predict(dados_entrada)
    probabilidade = modelo.predict_proba(dados_entrada)[predicao] * 100
    
    resultado = "Sobreviveu" if predicao == 1 else "Não Sobreviveu"
    
    return {
        "passageiro_enviado": {
            "classe": passageiro.Pclass,
            "genero": passageiro.Sex,
            "idade": passageiro.Age,
            "tarifa": passageiro.Fare
        },
        "resultado_predicao": resultado,
        "confianca_do_modelo": f"{probabilidade:.2f}%"
    }
