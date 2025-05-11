# Desafio I: Ciência e Governança de Dados - Zetta Lab 2025

Este repositório contém a solução para o Desafio I do Zetta Lab 2025.

## Acesso ao Dashboard Interativo

Você pode acessar o dashboard interativo diretamente no Streamlit Cloud através do seguinte link:

➡️ **[Acessar Dashboard](https://boakpe-zettalab2025-dasafio1-dados-dashboard.streamlit.app/)** ⬅️

Alternativamente, siga as instruções abaixo para executá-lo localmente.

## Estrutura do Projeto

```
desafio1_zetta_lab/
├── data/             # Dados brutos utilizados no desafio (IBGE, INPE, etc.)
├── src/              # Notebooks de processamento e ETL dos dados
│   ├── integracao.ipynb  # Integra e consolida todos os dados processados em um dataset final
│   ├── pib.ipynb         # Processa e filtra os dados de PIB dos municípios do Pará
│   ├── plantacoes.ipynb  # Processa dados de área plantada de soja e milho
│   ├── queimadas.ipynb   # Processa dados de focos de queimadas
│   ├── rebanho.ipynb     # Processa dados do rebanho bovino
│   └── # (Outros notebooks de processamento, ex: desmatamento, Bolsa Família)
├── dashboard.py      # Código do dashboard interativo em Streamlit
├── requirements.txt  # Dependências do projeto Python
├── RELATÓRIO.pdf     # Relatório final detalhado em PDF
├── RELATÓRIO.md      # Relatório final em formato Markdown
└── README.md         # Este arquivo
```

## Como Executar Localmente

Para executar o projeto em sua máquina local, siga os passos:

1.  **Clone o repositório (se ainda não o fez):**
    ```bash
    git clone https://github.com/Boakpe/ZettaLab2025_Dasafio1_Dados.git
    cd ZettaLab2025_Dasafio1_Dados
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows:
    # venv\Scripts\activate
    # No macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o dashboard:**
    ```bash
    streamlit run dashboard.py
    ```

**Observação:** O projeto foi testado com Python 3.12.9.

## Sobre o Processamento de Dados

*   **Dados Fonte:** Os dados brutos utilizados neste desafio, provenientes de diversas fontes como IBGE, INPE, entre outros, estão localizados na pasta `data/`.
*   **Processamento e ETL:** A etapa de processamento dos dados foi realizada por meio de Jupyter Notebooks, que se encontram na pasta `src/`. Cada notebook é responsável por tratar uma fonte de dados específica (PIB, plantações, queimadas, rebanho, etc.), culminando no notebook `integracao.ipynb`, que consolida todas as informações em um dataset final utilizado pelo dashboard.

## Relatórios

A análise detalhada, as metodologias empregadas, os insights obtidos e as conclusões do projeto estão documentados nos seguintes arquivos:

*   `RELATÓRIO.pdf`: Relatório final em formato PDF.
*   `RELATÓRIO.md`: Relatório final em formato Markdown.