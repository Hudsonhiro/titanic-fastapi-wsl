# 🚢 Titanic Machine Learning API

Uma solução profissional de engenharia de dados para o clássico desafio do Titanic no Kaggle. Este projeto treina um modelo de Machine Learning (`Random Forest`) para prever a sobrevivência de passageiros e o disponibiliza através de uma API web de alta performance.

## 🛠️ Tecnologias Utilizadas

* **WSL 2 (Ubuntu):** Ambiente de desenvolvimento Linux nativo.
* **uv:** Gerenciador de pacotes extremamente rápido da Astral.
* **FastAPI:** Framework moderno e rápido para a criação da API.
* **Scikit-Learn & Pandas:** Ferramentas para manipulação de dados e Machine Learning.
* **Uvicorn:** Servidor ASGI para rodar a aplicação.

## 🚀 Como Rodar o Projeto Localmente

### 1. Clonar o repositório
```bash
git clone https://github.com
cd NOME_DO_REPOSITORIO
```

### 2. Criar e Ativar o Ambiente Virtual (com uv)
```bash
uv venv
source .venv/bin/activate
```

### 3. Instalar as Dependências
```bash
uv pip install -r requirements.txt
```

### 4. Treinar o Modelo de IA
Execute o script para processar os dados e gerar o arquivo `modelo.pkl`:
```bash
python main.py
```

### 5. Iniciar a API FastAPI
```bash
uvicorn app:app --reload
```

Acesse a documentação interativa direto no seu navegador em: `http://127.0.0`
