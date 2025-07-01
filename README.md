# Desafio II: Ciência e Governança de Dados - Zetta Lab 2025

Este repositório contém a solução completa para os **Desafios I e II** do Zetta Lab 2025, abrangendo desde a aquisição e exploração de dados até a modelagem preditiva e a criação de um dashboard interativo para análise dos impactos do desmatamento no estado do Pará.

## Acesso ao Dashboard Interativo

Você pode acessar o dashboard interativo, que inclui as previsões dos modelos, diretamente no Streamlit Cloud através do seguinte link:

➡️ **[Acessar Dashboard](https://boakpe-zettalab2025-dasafio1-dados-dashboard2-09cv3g.streamlit.app/)** ⬅️ 
*(Link atualizado para o novo dashboard, se aplicável, ou mantenha o antigo se foi atualizado no mesmo link)*

Alternativamente, siga as instruções abaixo para executá-lo localmente.

## Principais Adições do Desafio II

Esta nova fase do projeto avançou da análise exploratória para a modelagem preditiva, com as seguintes adições:
- **Novos Dados:** Inclusão de dados de **Internações por Doenças Respiratórias** do DATASUS para medir o impacto na saúde pública.
- **Modelagem Preditiva:** Treinamento de **4 modelos de Machine Learning (LightGBM)** para prever a evolução de:
  1. PIB per Capita
  2. VAB Agropecuária
  3. Benefícios Sociais (Bolsa Família)
  4. Internações por Doenças Respiratórias
- **Novo Dashboard:** Um novo dashboard foi criado (`dashboard2.py`) para incluir as previsões dos modelos, permitir simulações de cenários futuros e apresentar análises de importância das variáveis (SHAP e LightGBM).
- **Relatório Final:** Documentação completa da metodologia de modelagem, resultados e recomendações estratégicas (`RELATÓRIO2.MD`).

## Estrutura do Projeto

```
desafio_zetta_lab_final/
├── data/                       # Dados brutos utilizados no desafio (IBGE, INPE, DATASUS, etc.)
│   └── RESULTADOS/             # Dados processados e consolidados
├── models/                     # Modelos de Machine Learning treinados e salvos (.joblib)
│   ├── modelo_pib_pred.joblib
│   ├── ...
├── notebooks/                  # Notebooks Jupyter para cada etapa do projeto
│   ├   # Notebooks da fase de ETL (Desafio I)
│   │   ├── pib.ipynb
│   │   ├── plantacoes.ipynb
│   │   ├── queimadas.ipynb
│   │   ├── ...
│   │   └── integracao.ipynb
│   │   # Notebooks da fase de modelagem (Desafio II)
│   └── 2_Modelagem/            
│       ├── doencas_respiratorias.ipynb  # ETL para os novos dados de saúde
│       └── train_models.ipynb         # Treinamento, validação e salvamento dos 4 modelos
├── dashboard2.py               # Código do dashboard interativo final (Streamlit)
├── requirements.txt            # Dependências do projeto Python
├── relatorio_fase_1.md         # Relatório da primeira fase (Análise Exploratória)
├── relatorio_final.md          # Relatório conclusivo do projeto (Desafio II)
└── README.md                   # Este arquivo
```

## Como Executar Localmente

Para executar o projeto e o dashboard interativo em sua máquina local, siga os passos abaixo:

1.  **Clone o repositório:**
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
    streamlit run dashboard2.py
    ```
    O dashboard será aberto em seu navegador padrão.

**Observação:** O projeto foi desenvolvido e testado com Python 3.10+.

## Sobre as Etapas do Projeto

1.  **Processamento de Dados (ETL):** Os dados brutos, localizados na pasta `data/`, foram processados e limpos usando Jupyter Notebooks (`notebooks/1_ETL_Dados/`). O notebook `integracao.ipynb` consolida todas as fontes em um dataset final.
2.  **Modelagem Preditiva:** Na pasta `notebooks/2_Modelagem/`, o notebook `doencas_respiratorias.ipynb` trata os novos dados de saúde. Em seguida, `train_models.ipynb` executa a engenharia de features, a otimização de hiperparâmetros (com Optuna) e o treinamento dos quatro modelos LightGBM, salvando os artefatos finais na pasta `models/`.
3.  **Visualização e Análise:** O `dashboard2.py` carrega os modelos salvos e os dados processados para fornecer uma plataforma interativa de visualização de previsões, análise de cenários e interpretação dos fatores de influência (SHAP).

## Relatórios

A análise detalhada, as metodologias empregadas, os insights obtidos e as conclusões do projeto estão documentados nos seguintes arquivos:

Parte I:
*   `RELATÓRIO.pdf`: Relatório final em formato PDF.
*   `RELATÓRIO.md`: Relatório final em formato Markdown.

Parte II:
*   `RELATÓRIO.pdf`: Relatório final em formato PDF.
*   `RELATÓRIO.md`: Relatório final em formato Markdown.