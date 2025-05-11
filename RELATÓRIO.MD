## Relatório Preliminar: Análise dos Impactos Socioeconômicos do Desmatamento no Estado do Pará

**Zetta Lab 2025 - Desafio I: Ciência e Governança de Dados**
**Autor(a):** Breno Carvalho Pedroso

### 1. Introdução

Este relatório preliminar apresenta os resultados iniciais da primeira etapa do Desafio I do Zetta Lab 2025, focado em "Ciência e Governança de Dados". O objetivo central é responder à pergunta: "Como poderíamos avaliar e prever os impactos socioeconômicos do desmatamento no estado do Pará?". Esta fase concentrou-se na aquisição, organização, exploração e visualização de dados relevantes.

### 2. Escolha dos Dados

Para abordar a complexa relação entre desmatamento e seus impactos socioeconômicos, foram selecionadas diversas bases de dados brasileiras, abrangendo dimensões ambientais, econômicas e sociais em nível municipal para o estado do Pará, no período de 2008 a 2024. A escolha visou capturar tanto as manifestações do desmatamento quanto seus potenciais vetores e consequências.

As principais variáveis e suas respectivas fontes incluem:

*   **Desmatamento (km²):**
    *   **Fonte:** INPE - Projeto PRODES/TerraBrasilis (monitoramento anual do desmatamento na Amazônia Legal).
    *   **Justificativa:** Principal indicador da pressão sobre a cobertura florestal, essencial para a análise.
*   **Indicadores Econômicos Municipais:**
    *   **PIB Municipal e PIB per capita (R$):** IBGE
    *   **Valor Adicionado Bruto (VAB) por Setor (Agropecuária, Indústria, Serviços) (R$ 1.000):** IBGE.
    *   **Justificativa:** Medem a atividade econômica e o bem-estar material, permitindo analisar como o desmatamento se correlaciona com o desenvolvimento econômico local e a estrutura produtiva.
*   **Indicadores Demográficos e Sociais:**
    *   **População Estimada:** IBGE.
    *   **Beneficiários de Programas de Transferência de Renda (ex: Bolsa Família/Auxílio Brasil):** Ministério do Desenvolvimento Social (MDS) - Dados agregados por município, obtidos via VIS DATA 3.
    *   **Justificativa:** A população indica a escala humana afetada e a pressão demográfica. Programas sociais podem indicar vulnerabilidade e a resposta do Estado.
*   **Indicadores Agropecuários:**
    *   **Área Plantada/Colhida de Culturas Relevantes (ex: Soja, Milho) (ha):** IBGE - Produção Agrícola Municipal (PAM).
    *   **Efetivo de Rebanhos (ex: Bovino) (cabeças):** IBGE - Pesquisa da Pecuária Municipal (PPM).
    *   **Justificativa:** A expansão agropecuária é um dos principais vetores de desmatamento. Essas variáveis ajudam a quantificar essa relação.
*   **Indicadores Ambientais Complementares:**
    *   **Focos de Queimada:** INPE - Programa Queimadas.
    *   **Justificativa:** Queimadas estão frequentemente associadas ao processo de desmatamento e preparo de terra para uso agropecuário.

A granularidade municipal e a série temporal permitem análises de evolução e comparações espaciais, cruciais para entender a heterogeneidade dos impactos no Pará.

### 3. Metodologia de Aquisição e Preparação dos Dados

A aquisição dos dados seguiu as seguintes etapas:

1.  **Identificação e Acesso às Fontes:** Utilização dos portais oficiais do INPE (TerraBrasilis, BDQueimadas) e IBGE (SIDRA, Contas Regionais), além de dados do MDS.
2.  **Extração:** Download dos arquivos em formatos como CSV, XLS e XLSX. Filtros geográficos (Estado do Pará) e temporais (2008 em diante) foram aplicados durante a extração ou logo após.
3.  **Limpeza e Tratamento:**
    *   Padronização de nomes de colunas.
    *   Conversão de tipos de dados (ex: strings para numéricos, tratamento de separadores decimais).
    *   Selecão de colunas relevantes para a análise.	
    *   Verificação e tratamento preliminar de valores ausentes (NaNs). (Os valores foram extrapolados usando uma regressão linear ou os valores foram preenchidos de acordo com o contexto do problema).
4.  **Consolidação:** Criação de um DataFrame único em Python, unindo as diversas fontes de dados por município e ano.
5.  **Ferramentas:** Python, Pandas, Scikit-learn, Plotly e Streamlit.

### 4. Análise Exploratória de Dados (EDA) e Principais Insights Preliminares

A EDA foi conduzida utilizando o DataFrame consolidado, com foco em visualizações e estatísticas descritivas para identificar padrões, tendências e correlações. Um dashboard interativo foi desenvolvido em Streamlit com gráficos gerados pela biblioteca Plotly para facilitar a exploração.

**Principais Insights Preliminares:**

1.  **Panorama do Desmatamento:**
    *   **Tendência Temporal Estadual:** A evolução do desmatamento total no Pará ao longo dos anos apresenta flutuações, com períodos de aumento e redução, indicando a complexidade dos fatores envolvidos e a eficácia variável das políticas de contenção. Por exemplo, o desmatamento apresentou um grande aumento entre 2018 e 2021 em razão da mudança na conjuntura política e do enfraquecimento de políticas ambientais, o que impactou diretamente a fiscalização e o controle do uso da terra.
    *   **Concentração Geográfica:** O desmatamento não é uniforme. Alguns municípios consistentemente figuram entre os mais desmatados, sugerindo "hotspots" de pressão ambiental. O mapa do desmatamento anual ilustra essa distribuição espacial dinâmica.
    *   **Evolução Municipal:** Municípios específicos demonstram trajetórias distintas de desmatamento, alguns com picos agudos e outros com aumento mais gradual, como visualizado nos gráficos de linha por município.

2.  **Relações Socioeconômicas:**
    *   **Desmatamento e PIB per capita:** A relação não é linear. Em alguns municípios, o aumento do desmatamento pode coincidir com o crescimento do PIB per capita (potencialmente impulsionado pela expansão de atividades primárias), enquanto em outros, essa relação pode ser menos clara ou até inversa a longo prazo, especialmente se os recursos naturais se esgotarem ou os impactos ambientais negativos se acumularem. Os gráficos de dispersão anuais oferecem uma visão pontual dessa dinâmica.
    *   **Desmatamento e VAB Agropecuária:** Observa-se uma correlação positiva entre o aumento do VAB da agropecuária e as taxas de desmatamento em muitos municípios, reforçando o papel da expansão agrícola e pecuária como um dos principais vetores.
    *   **Desmatamento e Programas Sociais (Bolsa Família):** Embora o programa tenha ampla cobertura na região, não foi possível identificar, com base nos dados analisados, uma relação direta ou consistente entre o número de beneficiários e as variações no desmatamento. A presença do programa pode refletir mais as condições socioeconômicas locais do que influenciar diretamente as dinâmicas ambientais.

3.  **Vetores Agropecuários:**
    *   **Expansão de Culturas e Rebanho:** O aumento da área plantada de commodities como soja e milho, e a expansão do rebanho bovino, mostram correlação espacial e temporal com as áreas de maior desmatamento em diversos municípios. Os gráficos de linha por município para essas variáveis, quando comparados com o desmatamento, evidenciam essas conexões.
    *   **Focos de Queimada:** Há uma forte correlação entre o número de focos de queimada e as áreas desmatadas, indicando o uso do fogo como ferramenta para limpeza de terra e expansão de atividades.

4.  **Estrutura Econômica Municipal (VAB):**
    *   Municípios com maior participação do VAB Agropecuário em sua economia tendem a apresentar taxas de desmatamento mais elevadas. A análise da evolução da composição do VAB ao longo do tempo pode indicar se o desmatamento está associado a uma primarização da economia local.

**Exemplo de Insight Específico:**
No município de Altamira, observa-se um pico de desmatamento em 2018, coincidindo com um aumento expressivo na área plantada de soja e milho (750% para a soja e 500% para o milho). Simultaneamente, o PIB per capita apresentou um crescimento de menos de 1%, sugerindo que os benefícios econômicos da expansão podem não estar sendo amplamente distribuídos ou que há um aumento da vulnerabilidade em outros setores.

### 5. Limitações do Estudo Preliminar

*   **Causalidade vs. Correlação:** A EDA identifica correlações, mas não estabelece relações de causa e efeito. Análises estatísticas mais robustas são necessárias para inferir causalidade.
*   **Disponibilidade e Defasagem de Dados:** Alguns dados socioeconômicos, como o IDH-M, não são calculados anualmente, limitando análises temporais de alta frequência para certos indicadores. Dados mais recentes (ex: 2024) podem ser preliminares ou parciais para algumas fontes.
*   **Heterogeneidade Intra-municipal:** A análise em nível municipal mascara variações internas significativas.
*   **Fatores Externos:** Políticas públicas (nacionais e estaduais), preços de commodities, eventos climáticos extremos e outros fatores exógenos não foram explicitamente modelados nesta fase, mas influenciam as dinâmicas observadas.

### 6. Conclusão Preliminar

A aquisição e exploração inicial dos dados revelaram complexas interações entre o desmatamento e diversos indicadores socioeconômicos no estado do Pará. O dashboard interativo desenvolvido permite uma visualização dinâmica dessas relações, fornecendo uma base sólida para investigações futuras. As descobertas preliminares confirmam a forte ligação do desmatamento com a expansão agropecuária e evidenciam a necessidade de políticas que considerem as particularidades econômicas e sociais de cada município para promover um desenvolvimento mais sustentável e mitigar os impactos negativos da perda florestal. A continuidade deste trabalho, com foco em modelagem preditiva e análises estatísticas mais aprofundadas, será crucial para subsidiar a tomada de decisão e a governança de dados no enfrentamento deste desafio.