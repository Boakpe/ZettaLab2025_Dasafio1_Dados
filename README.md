# Desafio I: Ciência e Governança de Dados - Zetta Lab 2025

Este repositório contém a solução para o Desafio I do Zetta Lab 2025.

## Estrutura do Projeto

```
desafio1_zetta_lab/
├── data/             # Dados utilizados no desafio (IBGE, INPE, etc.)
├── src/              # Notebooks de processamento de dados
│   ├── integracao.ipynb  # Integra e consolida todos os dados (desmatamento, PIB, queimadas, Bolsa Família, plantações e rebanho bovino) em um único dataset final
│   ├── pib.ipynb         # Processa e filtra os dados de PIB dos municípios do Pará
│   ├── plantacoes.ipynb  # Processa dados de área plantada de soja e milho por município e ano
│   ├── queimadas.ipynb   # Processa dados de focos de queimadas por município e ano
│   ├── rebanho.ipynb     # Processa dados do rebanho bovino por município e ano
├── dashboard.py      # Código do dashboard em Streamlit
├── requirements.txt  # Dependências do projeto
├── RELATÓRIO.pdf     # Relatório final em PDF
├── RELATÓRIO.md      # Relatório em formato Markdown
└── README.md         # Este arquivo
```

## Como executar

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

2. Execute o dashboard:
   ```
   streamlit run dashboard.py
   ```

OBS: Testado com Python 3.12.9

## Sobre

- Dados: disponíveis na pasta `data/`
- Processamento: notebooks em `src/`