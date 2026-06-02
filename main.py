import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Carregando os dados do Titanic...")
treino = pd.read_csv("train.csv")

# 1. Tratamento de dados (Idade e Tarifa)
treino['Age'] = treino['Age'].fillna(treino['Age'].median())
treino['Fare'] = treino['Fare'].fillna(treino['Fare'].median()) # Nova coluna tratada

# 2. Converte texto para número (Homem = 0, Mulher = 1)
treino['Sex'] = treino['Sex'].map({'male': 0, 'female': 1})

# 3. Seleciona as colunas de entrada (Adicionamos 'Fare')
X = treino[['Pclass', 'Sex', 'Age', 'Fare']]
y = treino['Survived']

print("Treinando o modelo de Machine Learning...")
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X, y)

# 4. Salva o modelo atualizado
joblib.dump(modelo, 'modelo.pkl')
print("Sucesso! Novo 'modelo.pkl' gerado com a variável Fare.")
