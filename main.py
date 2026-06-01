import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Carregando os dados do Titanic...")
treino = pd.read_csv("train.csv")

# Tratamento básico de dados (preenche idades vazias e converte texto em número)
treino['Age'] = treino['Age'].fillna(treino['Age'].median())
treino['Sex'] = treino['Sex'].map({'male': 0, 'female': 1})

# Seleciona as colunas de entrada (X) e o resultado esperado (y)
X = treino[['Pclass', 'Sex', 'Age']]
y = treino['Survived']

print("Treinando o modelo de Machine Learning...")
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X, y)

# Salva o "cérebro" da IA em um arquivo
joblib.dump(modelo, 'modelo.pkl')
print("Sucesso! Arquivo 'modelo.pkl' gerado na pasta.")
