# Desafio Técnico – Alocação Inteligente de Pontos de Marketing

Este projeto implementa uma solução inteligente para alocação de pontos de marketing entre clientes e produtos, com o objetivo de **maximizar a receita futura** com base em dados históricos de pedidos, pontos e preços.

## 🔍 Objetivo

Desenvolver um pipeline que:

- Agrupe clientes e produtos com base em similaridade.
- Modele o impacto dos investimentos em pontos de marketing na receita futura.
- Faça recomendações diárias otimizadas para alocação de pontos.

## 🧠 Abordagem

A abordagem segmenta os dados em grupos de clientes-produtos para modelar suas dinâmicas específicas. Um modelo MLP (Perceptron Multicamadas) é treinado para cada grupo, com escalonamento personalizado e rastreamento com MLflow.

As previsões são feitas para janelas diárias, permitindo estratégias de alocação adaptativas ao longo do tempo.

> 💡 *Esta é uma versão inicial. O pipeline pode ser significativamente melhorado com uma análise mais profunda das features e com a geração de novas variáveis que capturem o comportamento e a sazonalidade dos clientes.*

## 📂 Estrutura do Projeto
'''
desafio_mt2/
├── data/ # Dados de entrada e saída
├── mlruns/ # Rastreamento MLflow, Modelos treinados (.keras) e scalers
├── functions.py # Funções principais e pipeline
├── main.ipynb # Notebook de execução e análise
├── solution.csv # Alocação final recomendada
├── requirements.txt
└── README.md
'''

## 🚀 Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/GGBF1991/desafio_mt2.git
   cd desafio_mt2
2. Crie um ambiente virtual:
    python -m venv .venv
    source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
3. Instale as dependências:
    pip install -r requirements.txt
