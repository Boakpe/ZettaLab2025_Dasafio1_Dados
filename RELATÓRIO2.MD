Com certeza! Com base no excelente trabalho da primeira fase e nos scripts de modelagem e do dashboard fornecidos, elaborei o relatório final para o Desafio II. O relatório sintetiza a metodologia, os resultados dos modelos, a análise de importância das features e as recomendações estratégicas, seguindo estritamente as diretrizes do desafio.

---

## Relatório Final: Modelagem Preditiva e Recomendações Estratégicas para os Impactos do Desmatamento no Pará

**Zetta Lab 2025 - Desafio II: Ciência e Governança de Dados** \
**Autor(a):** Breno Carvalho Pedroso \
<img src='https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png' alt="GitHub Logo" width="25" style="vertical-align: middle; margin-right: 5px;"> Repositório do Projeto (contém Notebooks, Dados, Modelos e Dashboard): [https://github.com/Boakpe/ZettaLab2025_Dasafio1_Dados](https://github.com/Boakpe/ZettaLab2025_Dasafio1_Dados)

### 1. Introdução

Este relatório documenta a segunda e conclusiva fase do projeto, dando continuidade à análise exploratória sobre os impactos socioeconômicos do desmatamento no estado do Pará. Na primeira etapa, foi realizado um extenso trabalho de aquisição, limpeza e exploração de dados, culminando em um dashboard interativo que revelou correlações complexas entre variáveis ambientais, econômicas e sociais.

A presente fase avança da exploração para a predição. O objetivo é desenvolver e validar modelos de Machine Learning capazes de prever a evolução de indicadores-chave, quantificar a influência de diferentes fatores (como o desmatamento) nessas previsões e, com base nesses insights, formular recomendações estratégicas para políticas públicas. Respondendo à pergunta central — "Como poderíamos avaliar e prever os impactos socioeconômicos do desmatamento no estado do Pará?" — este trabalho apresenta uma abordagem quantitativa para antecipar tendências e subsidiar a tomada de decisão com base em evidências.

### 2. Metodologia de Modelagem Preditiva

Para construir um sistema de previsão robusto, foi adotada uma metodologia estruturada, desde a engenharia de features até a validação e otimização dos modelos.

#### 2.1. Escolha da Abordagem

A abordagem adotada foi treinar modelos de previsão de séries temporais para quatro variáveis-alvo distintas, representando diferentes dimensões do impacto socioeconômico e de saúde:
1.  **PIB per capita (R$):** Indicador de bem-estar econômico.
2.  **VAB Agropecuária (R$ 1.000):** Medida da atividade do setor primário, principal vetor de desmatamento.
3.  **Total de Benefícios Básicos (Bolsa Família):** Proxy para vulnerabilidade social.
4.  **Internações por Doenças Respiratórias:** Indicador de impacto na saúde pública, potencialmente ligado a queimadas.

O problema foi enquadrado como uma tarefa de previsão para o próximo passo no tempo (ano `t+1`), utilizando dados históricos (ano `t` e anteriores) como preditores. Esta abordagem permite que cada modelo capture as dinâmicas específicas que influenciam sua variável-alvo.

#### 2.2. Engenharia de Features

Para enriquecer os modelos com informações temporais, foi implementada uma extensa engenharia de features. A função `criar_features` gerou variáveis cruciais a partir dos dados brutos:
*   **Lags Temporais (Shifts):** Valores de features de anos anteriores (e.g., `Desmatamento_lag1`, `PIB_lag1`) foram criados para capturar a dependência temporal e os efeitos de autorregressão. Foram utilizados até 3 lags.
*   **Taxas de Crescimento (Growth):** A variação percentual de uma feature em relação ao ano anterior (`pct_change`) foi calculada para modelar a dinâmica de aceleração ou desaceleração.
*   **Médias Móveis (Rolling Means):** Médias de janelas de 2 e 3 anos foram calculadas para suavizar flutuações de curto prazo e capturar tendências mais estáveis.
*   **Tendência Linear (Trend):** Foi calculada uma tendência linear simples sobre os últimos 3 anos de dados para cada feature, aproximando a direção de longo prazo da variável.

Essa engenharia transforma uma série de "fotografias" anuais em um conjunto de dados dinâmico, permitindo que os modelos aprendam não apenas com valores absolutos, mas também com suas trajetórias e momentum.

#### 2.3. Escolha do Modelo: LightGBM

O algoritmo **LightGBM (Light Gradient Boosting Machine)** foi selecionado para todos os modelos. A escolha se justifica por suas vantagens:
*   **Alto Desempenho:** Conhecido por sua precisão em dados tabulares.
*   **Eficiência Computacional:** É significativamente mais rápido que outros algoritmos de boosting, o que é crucial para a otimização de hiperparâmetros.
*   **Flexibilidade:** Lida nativamente com valores ausentes e variáveis categóricas (como `Município`).
*   **Regularização:** Possui parâmetros de regularização (L1/L2) que ajudam a prevenir o overfitting.

#### 2.4. Validação e Otimização de Hiperparâmetros

Para garantir a robustez e a capacidade de generalização dos modelos, foi implementado um rigoroso processo de validação e otimização:
*   **Divisão Temporal:** Os dados foram divididos em três conjuntos cronológicos para simular um cenário real de previsão:
    *   **Treino:** Dados até o ano de 2020.
    *   **Validação:** Dados do ano de 2021 (usado para otimização).
    *   **Teste:** Dados do ano de 2022 (usado para avaliação final do modelo).
    Essa divisão evita o vazamento de dados (*data leakage*), garantindo que o modelo seja avaliado em dados que ele nunca viu.
*   **Otimização com Optuna:** A biblioteca `Optuna` foi utilizada para a otimização automática de hiperparâmetros. A Optuna explorou de forma inteligente diferentes combinações de parâmetros (como `learning_rate`, `num_leaves`, etc.), buscando minimizar o erro (RMSE) no conjunto de validação.
*   **Early Stopping:** Durante o treinamento final do modelo com os melhores parâmetros, a técnica de *early stopping* foi empregada. Ela monitora o desempenho no conjunto de validação e interrompe o treinamento se o erro não melhorar por 300 rodadas consecutivas, evitando o overfitting.

O desempenho final dos quatro modelos no conjunto de teste (ano de 2022) foi o seguinte:

| Modelo | Métrica | Valor | Descrição |
| :--- | :--- | :--- | :--- |
| **PIB per Capita** | RMSE | 31.979.620.320,60 | Erro Quadrático Médio |
| | R² | 0.941 | Coeficiente de Determinação |
| **VAB Agropecuária** | RMSE | 13.929.571.298,85 | Erro Quadrático Médio |
| | R² | 0.985 | Coeficiente de Determinação |
| **Benefícios Sociais** | RMSE | 389.049.888,29 | Erro Quadrático Médio |
| | R² | 0.989 | Coeficiente de Determinação |
| **Saúde Respiratória** | RMSE | 1.838,62 | Erro Quadrático Médio |
| | R² | 0.887 | Coeficiente de Determinação |

Os altíssimos valores de R² (acima de 0.94 para a maioria dos modelos, exceto saúde) indicam que os modelos foram capazes de explicar uma grande parte da variância nos dados de teste, demonstrando excelente poder preditivo.

### 3. Análise da Importância das Variáveis

Entender *quais* fatores mais influenciam as previsões é tão importante quanto a precisão dos modelos. Foram utilizadas duas técnicas complementares para interpretar os resultados, ambas implementadas na aba "Importância das Features" do dashboard.

1.  **Importância Nativa do LightGBM (Gain):** Esta abordagem mede o ganho total em precisão que uma feature proporciona ao ser utilizada para dividir os dados nas árvores de decisão do modelo. É uma visão de alto nível das features mais úteis. Consistentemente, variáveis de *lag* (como `PIB per capita_lag1`) e a própria categoria `Município` apareceram como muito importantes, indicando forte dependência temporal e características locais específicas.

2.  **Análise SHAP (SHapley Additive exPlanations):** O SHAP oferece uma visão mais granular e profunda, mostrando não apenas a importância de uma feature, mas também a **direção e a magnitude** de seu impacto em cada previsão individual.
    *   **Modelo de PIB per Capita:** A análise SHAP revelou que o `VAB Agropecuária_lag1` tem um impacto positivo significativo. O `Desmatamento (km²)_lag1` também aparece como um fator relevante, muitas vezes com impacto positivo a curto prazo, sugerindo que a expansão econômica em muitas áreas está, de fato, ligada à conversão da floresta.
    *   **Modelo de Saúde Respiratória:** O `Focos de Queimada_lag1` emergiu como a variável mais impactante, com uma clara relação positiva: valores mais altos de focos de queimada levam a um aumento previsto nas internações respiratórias. A `População` também é um fator importante, como esperado.

Essas análises confirmam hipóteses e quantificam relações, fornecendo uma base sólida para a formulação de políticas direcionadas.

### 4. Dashboard Interativo de Resultados

O produto final desta análise é um dashboard interativo desenvolvido com Streamlit. Ele serve como uma ponte entre a complexidade dos modelos e a necessidade de insights acionáveis para gestores públicos e analistas. Suas principais funcionalidades são:
*   **Previsões Personalizadas:** O usuário pode selecionar um município específico ou uma visão agregada para o estado e definir o horizonte de previsão (1 a 10 anos).
*   **Visualização Integrada:** Gráficos interativos mostram a série histórica e as previsões futuras em uma única visualização, permitindo a fácil identificação de tendências.
*   **Análise de Impacto do Desmatamento:** Uma funcionalidade permite sobrepor a série histórica de desmatamento no mesmo gráfico, facilitando a análise visual da correlação entre a perda de floresta e as variáveis socioeconômicas e de saúde.
*   **Transparência do Modelo:** A aba "Importância das Features" permite que os usuários explorem os fatores que impulsionam as previsões de cada um dos quatro modelos, aumentando a confiança e a interpretabilidade dos resultados.
*   **Acesso aos Dados:** Tabelas com os valores previstos e os dados históricos completos estão disponíveis para consulta e download.

O dashboard não é apenas uma ferramenta de visualização, mas uma plataforma de simulação e análise de cenários, permitindo que os tomadores de decisão explorem os possíveis futuros de diferentes municípios sob a influência de fatores-chave.

### 5. Recomendações Estratégicas

Com base nos resultados da modelagem e na análise de importância das variáveis, são propostas as seguintes recomendações estratégicas:

1.  **Políticas de Desenvolvimento Econômico Desacopladas do Desmatamento:**
    *   **Insight:** O `VAB Agropecuária` e o `Desmatamento` são preditores importantes e positivamente correlacionados com o `PIB per capita` a curto prazo.
    *   **Recomendação:** Fomentar cadeias produtivas sustentáveis e de maior valor agregado (bioeconomia, turismo ecológico, sistemas agroflorestais) para que o crescimento econômico não dependa da expansão da fronteira agrícola. Utilizar o dashboard para monitorar municípios onde o PIB cresce junto com o desmatamento e direcionar programas de transição econômica para essas áreas.

2.  **Sistema de Alerta e Prevenção de Saúde Pública:**
    *   **Insight:** `Focos de Queimada` é o principal preditor do aumento de `Internações por Doenças Respiratórias`.
    *   **Recomendação:** Integrar os dados de focos de queimada do INPE com a Secretaria de Saúde do Estado. Utilizar o modelo preditivo para antecipar picos de internações em municípios específicos, permitindo a alocação proativa de recursos (medicamentos, leitos, profissionais de saúde) e a emissão de alertas de saúde para a população em áreas de risco.

3.  **Fiscalização Ambiental Inteligente e Preditiva:**
    *   **Insight:** A variável `Município` é consistentemente uma das mais importantes, indicando que as dinâmicas são altamente localizadas.
    *   **Recomendação:** Utilizar o dashboard para identificar os municípios com as maiores taxas de desmatamento previstas para o ano seguinte. Cruzar essa informação com a importância local de vetores como `Área plantada soja` e `Total Rebanho (Bovino)` para direcionar as equipes de fiscalização do IBAMA e da SEMAS/PA para os "hotspots" mais prováveis, otimizando o uso de recursos limitados.

4.  **Focalização de Políticas Sociais:**
    *   **Insight:** O modelo de `Benefícios Sociais` é influenciado por indicadores econômicos como o `PIB per capita`.
    *   **Recomendação:** Monitorar as previsões de PIB per capita. Em municípios onde se prevê estagnação ou queda econômica (potencialmente devido ao esgotamento de recursos naturais pós-desmatamento), o governo pode antecipar um aumento na demanda por programas de assistência social e fortalecer a rede de proteção social preventivamente.

### 6. Conclusão

Este projeto demonstrou com sucesso a aplicação de técnicas de ciência de dados para avaliar e prever os complexos impactos socioeconômicos do desmatamento no Pará. A partir de dados públicos, foram desenvolvidos modelos preditivos de alta performance e um dashboard interativo que traduz os resultados em insights acionáveis.

A análise confirmou quantitativamente a forte ligação entre a expansão agropecuária, o desmatamento, o crescimento econômico de curto prazo e os impactos negativos na saúde pública. As ferramentas desenvolvidas oferecem ao poder público a capacidade de antecipar tendências, entender os fatores determinantes por trás delas e formular políticas mais eficazes e direcionadas. O trabalho estabelece uma base sólida para uma governança de dados mais proativa e baseada em evidências, crucial para conciliar o desenvolvimento com a conservação ambiental na Amazônia.