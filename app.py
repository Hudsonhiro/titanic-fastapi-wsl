from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="API de Predição do Titanic",
    description="Interface profissional para prever sobrevivência de passageiros usando IA.",
    version="1.0.0"
)

# Carrega o modelo que você treinou no Passo 1
try:
    modelo = joblib.load('modelo.pkl')
except Exception as e:
    print(f"Erro ao carregar o modelo. Rode o main.py primeiro. Erro: {e}")

# Define o formato exato dos dados que a API aceita receber
class Passageiro(BaseModel):
    Pclass: int       # Classe do bilhete (1, 2 ou 3)
    Sex: int          # 0 para Homem, 1 para Mulher
    Age: float        # Idade do passageiro

@app.get("/")
def home():
    return {"status": "Online", "mensagem": "API do Titanic rodando com sucesso no WSL!"}

@app.post("/predict")
def prever_sobrevivencia(passageiro: Passageiro):
    # Organiza os dados recebidos no formato de matriz exigido pelo Scikit-Learn
    dados_entrada = [[passageiro.Pclass, passageiro.Sex, passageiro.Age]]
    
    # Faz a predição (0 ou 1)
    predicao = modelo.predict(dados_entrada)[0]
    
    # Descobre a probabilidade percentual (opcional, deixa mais profissional!)
    probabilidade = modelo.predict_proba(dados_entrada)[0][predicao] * 100
    
    resultado = "Sobreviveu" if predicao == 1 else "Não Sobreviveu"
    
    return {
        "predicao": int(predicao),
        "resultado": resultado,
        "confianca_do_modelo": f"{probabilidade:.2f}%"
    }
