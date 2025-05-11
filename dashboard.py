import pandas as pd
import streamlit as st
import plotly.express as px
import json
import plotly.graph_objects as go

# --- 0. Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Impactos do Desmatamento no Pará")

# --- 1. Data Loading and Preprocessing ---
@st.cache_data
def load_and_prepare_data(csv_path='data/RESULTADOS/df_final.csv'): 
    try:
        df_raw = pd.read_csv(csv_path)
        df = df_raw.copy()

        # Padronizar nomes das colunas para facilitar (opcional, mas recomendado)
        column_mapping = {
            'Município': 'municipio',
            'Ano': 'ano',
            'Código IBGE': 'codigo_ibge',
            'Desmatamento (km²)': 'desmatamento_km2',
            'PIB per capita (R$)': 'pib_per_capita',
            'PIB (R$ 1.000)': 'pib_total_mil_reais',
            'VAB Agropecuária (R$ 1.000)': 'vab_agro_mil_reais',
            'VAB Indústria (R$ 1.000)': 'vab_industria_mil_reais',
            'VAB Serviços (R$ 1.000)': 'vab_servicos_mil_reais',
            'População': 'populacao',
            'Focos de Queimada': 'focos_queimada',
            'Total de Benefícios Básicos (Bolsa Família)': 'bolsa_familia_beneficios', # Ou um nome mais curto
            'Área plantada soja (ha)': 'area_soja_ha',
            'Área plantada milho (ha)': 'area_milho_ha',
            'Total Rebanho (Bovino)': 'rebanho_bovino_cabecas'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Converter colunas numéricas que podem ter sido lidas como string (se houver vírgulas como decimal)
        # O seu exemplo de CSV parece já estar com pontos decimais, então isso pode não ser necessário.
        # Mas é uma boa prática verificar. Ex:
        # cols_to_numeric = ['desmatamento_km2', 'pib_per_capita', ...]
        # for col in cols_to_numeric:
        #     if df[col].dtype == 'object':
        #         df[col] = df[col].str.replace(',', '.', regex=False).astype(float)

        # Certifica que 'codigo_ibge' é string para o mapa
        df['codigo_ibge'] = df['codigo_ibge'].astype(str)
        # Certifica que 'ano' é int
        df['ano'] = df['ano'].astype(int)

        return df
    except FileNotFoundError:
        st.error(f"ERRO: Arquivo '{csv_path}' não encontrado. Verifique o caminho.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"ERRO ao carregar ou processar os dados: {e}")
        return pd.DataFrame()

df_data = load_and_prepare_data() # Carrega seus dados consolidados

# --- Load GeoJSON ---
geojson_path = 'data/GEOJSON/municipios_pa.json' 
feature_id_key_geojson = 'properties.id'      

@st.cache_data
def load_geojson(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"ERRO: Arquivo GeoJSON '{path}' não encontrado. O mapa não funcionará.")
        return None
    except json.JSONDecodeError:
        st.error(f"ERRO: Arquivo GeoJSON '{path}' não é um JSON válido.")
        return None
    except Exception as e:
        st.error(f"ERRO ao carregar GeoJSON '{path}': {e}")
        return None

geojson_para = load_geojson(geojson_path)

# --- Main Dashboard Title ---
st.title("Análise dos Impactos Socioeconômicos do Desmatamento no Estado do Pará")
st.markdown("Explore dados de desmatamento, indicadores socioeconômicos e atividades agropecuárias nos municípios paraenses.")
st.markdown("---")

# --- Sidebar Filters (Global) ---
st.sidebar.header("Filtros Globais")
if not df_data.empty:
    all_municipalities_sorted = sorted(df_data['municipio'].unique())
    selected_municipalities_global = st.sidebar.multiselect(
        "Selecione Municípios (para algumas análises):",
        options=all_municipalities_sorted,
        default=all_municipalities_sorted[:5] # Default para os primeiros 5 ou os mais relevantes
    )

    min_year_data, max_year_data = int(df_data['ano'].min()), int(df_data['ano'].max())
    selected_year_global = st.sidebar.slider(
        "Selecione um Ano (para mapas e algumas análises):",
        min_value=min_year_data,
        max_value=max_year_data,
        value=max_year_data,
        step=1
    )
else:
    st.sidebar.warning("Dados não carregados. Filtros indisponíveis.")
    selected_municipalities_global = []
    selected_year_global = 2023 # Um default qualquer


# --- Create Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Panorama do Desmatamento",
    "💰 Indicadores Socioeconômicos",
    "🔗 Correlações e Relações",
    "🌾 Análise Setorial Agropecuária"
])


# --- Tab 1: Panorama do Desmatamento ---
with tab1:
    st.header("Panorama Geral do Desmatamento no Pará")

    if df_data.empty:
        st.warning("Não foi possível carregar os dados. As visualizações não podem ser geradas.")
    else:
        # Data prep for this tab
        df_desmatamento_estado_para = df_data.groupby('ano')['desmatamento_km2'].sum().reset_index()

        # Top 10 - Acumulado no período disponível
        df_top_10_mun_acumulado_total = df_data.groupby('municipio')['desmatamento_km2'].sum().nlargest(10).reset_index()

        # --- Map Section ---
        st.subheader(f"Mapa Interativo: Desmatamento Municipal em {selected_year_global}")
        if geojson_para:
            df_map_year = df_data[df_data['ano'] == selected_year_global]
            if not df_map_year.empty:
                max_desmatamento_ano = df_map_year['desmatamento_km2'].max()
                fig_map = px.choropleth_mapbox(
                    df_map_year,
                    geojson=geojson_para,
                    locations='codigo_ibge',
                    featureidkey=feature_id_key_geojson,
                    color='desmatamento_km2',
                    color_continuous_scale="Reds",
                    range_color=(0, max_desmatamento_ano if max_desmatamento_ano > 0 else 1), # Evita erro se max for 0
                    mapbox_style="carto-positron",
                    zoom=4.2,
                    center={"lat": -3.7, "lon": -52.5}, # Ajustado ligeiramente
                    opacity=0.7,
                    labels={'desmatamento_km2': 'Desmatamento (km²)'},
                    hover_name='municipio',
                    hover_data={'desmatamento_km2': ':.2f km²', 'codigo_ibge': False, 'ano': False}
                )
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info(f"Não há dados de desmatamento para o ano {selected_year_global} para exibir no mapa.")
        else:
            st.error("GeoJSON não carregado. Mapa não pode ser exibido.")

        st.markdown("---")
        # --- Static Plots Section (Side-by-Side) ---
        st.subheader("Análises Agregadas do Desmatamento")
        col1_t1, col2_t1 = st.columns(2)

        with col1_t1:
            fig_total_para_evolucao = px.line(
                df_desmatamento_estado_para, x='ano', y='desmatamento_km2',
                title='Evolução do Desmatamento Total no Pará (km²)',
                labels={'ano': 'Ano', 'desmatamento_km2': 'Área Desmatada (km²)'}, markers=True
            )
            fig_total_para_evolucao.update_layout(title_x=0.5)
            st.plotly_chart(fig_total_para_evolucao, use_container_width=True)

        with col2_t1:
            fig_top_10_mun = px.bar(
                df_top_10_mun_acumulado_total, x='municipio', y='desmatamento_km2',
                title='Top 10 Municípios: Maior Desmatamento Acumulado (km²)',
                labels={'municipio': 'Município', 'desmatamento_km2': 'Desmatamento Total (km²)'}
            )
            fig_top_10_mun.update_layout(xaxis_tickangle=-45, title_x=0.5)
            st.plotly_chart(fig_top_10_mun, use_container_width=True)

        st.markdown("---")
        st.subheader("Evolução do Desmatamento por Município")
        if selected_municipalities_global:
            df_filtered_mun_evol = df_data[df_data['municipio'].isin(selected_municipalities_global)]
            fig_evol_mun_filtered = px.line(
                df_filtered_mun_evol, x='ano', y='desmatamento_km2', color='municipio',
                title='Evolução do Desmatamento (km²) nos Municípios Selecionados',
                labels={'ano': 'Ano', 'desmatamento_km2': 'Desmatamento (km²)', 'municipio': 'Município'}
            )
            fig_evol_mun_filtered.update_layout(title_x=0.5)
            st.plotly_chart(fig_evol_mun_filtered, use_container_width=True)
        else:
            st.info("Selecione municípios na barra lateral para visualizar a evolução individual.")

# --- Tab 2: Indicadores Socioeconômicos ---
with tab2:
    st.header("Indicadores Socioeconômicos e sua Relação com o Desmatamento")

    if df_data.empty or not selected_municipalities_global:
        st.warning("Dados não carregados ou nenhum município selecionado na barra lateral.")
    else:
        df_filtered_socio = df_data[df_data['municipio'].isin(selected_municipalities_global)]

        indicador_opts = {
            'PIB per capita (R$)': 'pib_per_capita',
            'População': 'populacao',
            'Benefícios Bolsa Família': 'bolsa_familia_beneficios',
            'Focos de Queimada': 'focos_queimada'
        }
        selected_indicador_label = st.selectbox(
            "Selecione um Indicador Socioeconômico:",
            options=list(indicador_opts.keys())
        )
        selected_indicador_col = indicador_opts[selected_indicador_label]

        col1_t2, col2_t2 = st.columns(2)

        with col1_t2:
            st.subheader(f"Evolução: {selected_indicador_label}")
            fig_evol_indicador = px.line(
                df_filtered_socio, x='ano', y=selected_indicador_col, color='municipio',
                title=f'Evolução de {selected_indicador_label} nos Municípios Selecionados',
                labels={'ano': 'Ano', selected_indicador_col: selected_indicador_label, 'municipio': 'Município'}
            )
            fig_evol_indicador.update_layout(title_x=0.5)
            st.plotly_chart(fig_evol_indicador, use_container_width=True)

        with col2_t2:
            st.subheader(f"Desmatamento vs. {selected_indicador_label} ({selected_year_global})")
            df_scatter_socio = df_data[df_data['ano'] == selected_year_global]
            if not df_scatter_socio.empty:
                 fig_scatter_socio = px.scatter(
                    df_scatter_socio,
                    x=selected_indicador_col,
                    y='desmatamento_km2',
                    color='municipio', # Pode ficar poluído, considere remover ou usar 'size'
                    size='populacao',  # Exemplo de uso de 'size'
                    hover_name='municipio',
                    title=f'Desmatamento vs. {selected_indicador_label} em {selected_year_global}',
                    labels={selected_indicador_col: selected_indicador_label, 'desmatamento_km2': 'Desmatamento (km²)'}
                )
                 fig_scatter_socio.update_layout(title_x=0.5, showlegend=False)
                 st.plotly_chart(fig_scatter_socio, use_container_width=True)
            else:
                st.info(f"Não há dados para o ano {selected_year_global} para este gráfico de dispersão.")

        st.markdown("---")
        st.subheader(f"Mapa: {selected_indicador_label} por Município ({selected_year_global})")
        if geojson_para:
            df_map_socio_year = df_data[df_data['ano'] == selected_year_global]
            if not df_map_socio_year.empty and selected_indicador_col in df_map_socio_year.columns:
                max_val_indicador = df_map_socio_year[selected_indicador_col].max()
                fig_map_socio = px.choropleth_mapbox(
                    df_map_socio_year,
                    geojson=geojson_para,
                    locations='codigo_ibge',
                    featureidkey=feature_id_key_geojson,
                    color=selected_indicador_col,
                    color_continuous_scale="Viridis", # Outra escala de cor
                    range_color=(0, max_val_indicador if max_val_indicador > 0 else 1),
                    mapbox_style="carto-positron",
                    zoom=4.2, center={"lat": -3.7, "lon": -52.5}, opacity=0.7,
                    labels={selected_indicador_col: selected_indicador_label},
                    hover_name='municipio'
                )
                fig_map_socio.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=500)
                st.plotly_chart(fig_map_socio, use_container_width=True)
            else:
                st.info(f"Não há dados de '{selected_indicador_label}' para o ano {selected_year_global} para exibir no mapa.")


# --- Tab 3: Correlações e Relações ---
# --- Tab 3: Correlações e Relações ---
with tab3:
    st.header("Correlações e Relações entre Variáveis")

    if df_data.empty:
        st.warning("Dados não carregados.")
    else:
        # Definição inicial das colunas para correlação
        base_cols_for_corr = [
            'desmatamento_km2', 'pib_per_capita', 'vab_agro_mil_reais',
            'vab_industria_mil_reais', 'vab_servicos_mil_reais', 'populacao',
            'focos_queimada', 'bolsa_familia_beneficios', 'area_soja_ha',
            'area_milho_ha', 'rebanho_bovino_cabecas'
        ]
        # Filtra para garantir que só usamos colunas que realmente existem no df_data
        cols_for_corr = [col for col in base_cols_for_corr if col in df_data.columns]

        if not cols_for_corr:
            st.warning("Nenhuma das colunas de correlação especificadas foi encontrada nos dados carregados.")
        else:
            analysis_level = st.radio(
                "Nível de Análise para Correlação:",
                ("Estado do Pará (Agregado por Ano)", "Municípios Selecionados (Dados Anuais)"),
                key="corr_level"
            )

            df_corr_analysis = pd.DataFrame() # Inicializa

            if analysis_level == "Estado do Pará (Agregado por Ano)":
                # Definir funções de agregação para o nível estadual
                agg_rules_estado = {}
                # Colunas que serão somadas. 'pib_total_mil_reais' é necessário para calcular 'pib_per_capita' estadual.
                # Adicionamos 'pib_total_mil_reais' aqui se existir no df_data original
                sum_cols_candidates = ['desmatamento_km2', 'vab_agro_mil_reais',
                                   'vab_industria_mil_reais', 'vab_servicos_mil_reais', 'populacao',
                                   'focos_queimada', 'bolsa_familia_beneficios', 'area_soja_ha',
                                   'area_milho_ha', 'rebanho_bovino_cabecas']
                if 'pib_total_mil_reais' in df_data.columns: # Verifica se pib_total_mil_reais existe
                    sum_cols_candidates.append('pib_total_mil_reais')

                for col in sum_cols_candidates:
                    if col in df_data.columns: # Só adiciona se a coluna existir nos dados originais
                        agg_rules_estado[col] = 'sum'
                
                if not agg_rules_estado:
                    st.warning("Nenhuma coluna válida para agregação estadual encontrada em df_data.")
                else:
                    # Agrega os dados por ano
                    df_estado_agg = df_data.groupby('ano', as_index=False).agg(agg_rules_estado)

                    # Calcula PIB per capita estadual se as colunas necessárias estiverem presentes
                    if 'pib_total_mil_reais' in df_estado_agg.columns and \
                       'populacao' in df_estado_agg.columns and \
                       not df_estado_agg['populacao'].eq(0).any(): # Evita divisão por zero
                        df_estado_agg['pib_per_capita'] = (df_estado_agg['pib_total_mil_reais'] * 1000) / df_estado_agg['populacao']
                    elif 'pib_per_capita' in cols_for_corr: # Se não puder calcular, mas é esperado
                        df_estado_agg['pib_per_capita'] = pd.NA # Adiciona a coluna com NA

                    # Seleciona apenas as colunas que estão em cols_for_corr E existem em df_estado_agg
                    final_cols_for_state_corr = [col for col in cols_for_corr if col in df_estado_agg.columns]
                    if final_cols_for_state_corr:
                        df_corr_analysis = df_estado_agg[final_cols_for_state_corr].copy()
                    else:
                        st.warning("Nenhuma coluna de 'cols_for_corr' pôde ser preparada para a análise de correlação estadual.")

            elif selected_municipalities_global: # Nível Municipal
                df_municipal_filtered = df_data[df_data['municipio'].isin(selected_municipalities_global)]
                # Para correlação municipal, não agrupamos por ano, pois queremos a variação ao longo dos anos por município.
                # Selecionamos apenas as colunas relevantes.
                final_cols_for_mun_corr = [col for col in cols_for_corr if col in df_municipal_filtered.columns]
                if final_cols_for_mun_corr:
                    df_corr_analysis = df_municipal_filtered[final_cols_for_mun_corr].copy()
                else:
                    st.info("Nenhuma coluna de 'cols_for_corr' encontrada para os municípios selecionados.")
            else:
                st.info("Selecione municípios na barra lateral para análise de correlação em nível municipal, ou escolha 'Estado do Pará'.")


            # Prossegue com a matriz de correlação e scatter plot se df_corr_analysis for válido
            if not df_corr_analysis.empty and len(df_corr_analysis) > 1:
                st.subheader("Matriz de Correlação")
                # Remove colunas que são inteiramente NA antes de calcular a correlação
                df_corr_analysis_cleaned = df_corr_analysis.dropna(axis=1, how='all')
                
                # Recalcula cols_for_corr com base nas colunas que restaram após limpeza
                valid_cols_for_heatmap = [col for col in cols_for_corr if col in df_corr_analysis_cleaned.columns]

                if not valid_cols_for_heatmap or len(valid_cols_for_heatmap) < 2:
                    st.warning("Não há colunas suficientes com dados válidos para gerar a matriz de correlação.")
                else:
                    corr_matrix = df_corr_analysis_cleaned[valid_cols_for_heatmap].corr()
                    fig_corr_heatmap = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        title="Matriz de Correlação entre Variáveis Selecionadas"
                    )
                    fig_corr_heatmap.update_layout(title_x=0.5, height=max(400, 50 * len(valid_cols_for_heatmap))) # Ajusta altura
                    st.plotly_chart(fig_corr_heatmap, use_container_width=True)

                    st.markdown("---")
                    st.subheader("Gráfico de Dispersão entre Duas Variáveis")
                    
                    # Atualiza opções para selectbox com base nas colunas válidas
                    scatter_options = valid_cols_for_heatmap 

                    if len(scatter_options) >=2:
                        col1_t3, col2_t3, col3_t3 = st.columns(3)
                        with col1_t3:
                            # Tenta encontrar um default sensível ou usa o primeiro
                            default_x_index = scatter_options.index('vab_agro_mil_reais') if 'vab_agro_mil_reais' in scatter_options else 0
                            x_axis_val = st.selectbox('Variável Eixo X:', options=scatter_options, index=default_x_index, key="scatter_x")
                        with col2_t3:
                            default_y_index = scatter_options.index('desmatamento_km2') if 'desmatamento_km2' in scatter_options else (1 if len(scatter_options) > 1 else 0)
                            y_axis_val = st.selectbox('Variável Eixo Y:', options=scatter_options, index=default_y_index, key="scatter_y")
                        
                        color_options_scatter = [None]
                        if analysis_level == "Estado do Pará (Agregado por Ano)" and 'ano' in df_estado_agg.columns: # 'ano' não estará em df_corr_analysis diretamente
                            color_options_scatter.append('ano') # Precisaria adicionar 'ano' ao df_corr_analysis para o estado
                        elif analysis_level != "Estado do Pará (Agregado por Ano)" and 'municipio' in df_data.columns and selected_municipalities_global:
                             # Para municipal, se quisermos colorir por ano ou município, o df_corr_analysis precisaria dessas colunas
                             # Atualmente df_corr_analysis só tem as colunas numéricas para correlação.
                             # Para colorir, precisaríamos de um dataframe mais completo para o scatter.
                             pass # Simplificando: sem coloração dinâmica por enquanto para manter df_corr_analysis focado.

                        # Para colorir o scatter plot de forma útil, especialmente no nível municipal,
                        # pode ser melhor usar o df_municipal_filtered diretamente, que ainda tem 'ano' e 'municipio'.
                        # E para o estado, df_estado_agg que tem 'ano'.
                        
                        df_for_scatter = df_corr_analysis # Use o df_corr_analysis que já tem as colunas numéricas
                        hover_name_scatter = None
                        color_val_scatter_final = None # Por enquanto, sem cor dinâmica no scatter se não for trivial

                        if analysis_level == "Estado do Pará (Agregado por Ano)":
                            # Adicionar 'ano' de volta para colorir/hover, se foi removido
                            if 'ano' not in df_for_scatter.columns and 'ano' in df_estado_agg.columns:
                                df_for_scatter = pd.merge(df_for_scatter, df_estado_agg[['ano']], left_index=True, right_index=True, how='left')
                            color_val_scatter_final = 'ano' if 'ano' in df_for_scatter.columns else None
                        
                        elif selected_municipalities_global: # Nível Municipal
                            # O df_corr_analysis atual para municipal não tem 'ano' ou 'municipio'
                            # Para adicionar, precisaríamos reconstruí-lo ou usar o df_municipal_filtered
                            # Vamos usar o df_municipal_filtered que é mais completo
                            df_for_scatter = df_data[df_data['municipio'].isin(selected_municipalities_global)]
                            color_val_scatter_final = 'municipio' # Ou 'ano'
                            hover_name_scatter = 'municipio'

                        with col3_t3:
                             # Manter a opção de cor, mas ela pode não funcionar perfeitamente sem ajustar df_for_scatter
                             color_val_temp = st.selectbox('Colorir por (opcional):', options=[None, 'ano', 'municipio'], index=0, key="scatter_color")
                             # Usaremos color_val_scatter_final que é mais seguro
                        
                        if x_axis_val and y_axis_val and x_axis_val != y_axis_val:
                            # Remove NaNs apenas para as colunas X e Y do scatter plot
                            df_scatter_plot_data = df_for_scatter.dropna(subset=[x_axis_val, y_axis_val])
                            if not df_scatter_plot_data.empty:
                                fig_scatter_corr = px.scatter(
                                    df_scatter_plot_data,
                                    x=x_axis_val,
                                    y=y_axis_val,
                                    color=color_val_scatter_final if color_val_scatter_final in df_scatter_plot_data.columns else None,
                                    trendline="ols" if df_scatter_plot_data[x_axis_val].nunique() > 1 else None,
                                    title=f'{y_axis_val.replace("_"," ").title()} vs. {x_axis_val.replace("_"," ").title()}',
                                    labels={x_axis_val:x_axis_val.replace('_',' ').title(), y_axis_val:y_axis_val.replace('_',' ').title()},
                                    hover_name=hover_name_scatter if hover_name_scatter in df_scatter_plot_data.columns else None
                                )
                                fig_scatter_corr.update_layout(title_x=0.5)
                                st.plotly_chart(fig_scatter_corr, use_container_width=True)
                            else:
                                st.info(f"Não há dados suficientes após remover valores ausentes para as variáveis '{x_axis_val}' e '{y_axis_val}'.")
                        elif x_axis_val == y_axis_val:
                            st.info("Selecione variáveis diferentes para os eixos X e Y.")
                        else:
                            st.info("Selecione variáveis para os eixos X e Y.")
                    else:
                        st.info("Não há variáveis suficientes para o gráfico de dispersão após a limpeza dos dados.")
            else:
                st.info("Não há dados suficientes para calcular a correlação para a seleção atual (necessário mais de 1 linha de dados).")

# --- Tab 4: Análise Setorial Agropecuária ---
with tab4:
    st.header("Análise Setorial: Agropecuária e Desmatamento")

    if df_data.empty or not selected_municipalities_global:
        st.warning("Dados não carregados ou nenhum município selecionado na barra lateral.")
    else:
        df_filtered_agro = df_data[df_data['municipio'].isin(selected_municipalities_global)]

        col1_t4, col2_t4 = st.columns(2)

        with col1_t4:
            st.subheader("Desmatamento vs. VAB Agropecuária")
            fig_desmat_vab = px.line(
                df_filtered_agro, x='ano', y='vab_agro_mil_reais', color='municipio',
                title='Evolução do VAB Agropecuário (R$ 1.000)',
                labels={'ano':'Ano', 'vab_agro_mil_reais':'VAB Agro (R$ 1.000)'}
            )
            # Adicionar desmatamento como segunda linha (ou área) para comparação visual
            # Poderia ser um gráfico com dois eixos Y, mas o Plotly Express simplifica com cores.
            # Ou, um gráfico de área para o desmatamento total dos municípios selecionados
            df_desmat_total_selecionados = df_filtered_agro.groupby('ano')['desmatamento_km2'].sum().reset_index()
            fig_desmat_total_sel_line = px.line(df_desmat_total_selecionados, x='ano', y='desmatamento_km2', markers=True)
            fig_desmat_total_sel_line.update_traces(yaxis="y2", name="Desmatamento Total Selecionado (km²)")


            from plotly.subplots import make_subplots
            fig_combined_vab_desmat = make_subplots(specs=[[{"secondary_y": True}]])
            for i, mun in enumerate(df_filtered_agro['municipio'].unique()):
                df_mun = df_filtered_agro[df_filtered_agro['municipio'] == mun]
                fig_combined_vab_desmat.add_trace(
                    go.Scatter(x=df_mun['ano'], y=df_mun['vab_agro_mil_reais'], name=f'VAB Agro - {mun}',
                               line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)])) ,
                    secondary_y=False,
                )
            fig_combined_vab_desmat.add_trace(
                go.Scatter(x=df_desmat_total_selecionados['ano'], y=df_desmat_total_selecionados['desmatamento_km2'], name='Desmatamento Total (km²)',
                           line=dict(color='red', dash='dash')),
                secondary_y=True,
            )
            fig_combined_vab_desmat.update_layout(title_text="VAB Agropecuário vs. Desmatamento", title_x=0.5)
            fig_combined_vab_desmat.update_yaxes(title_text="VAB Agropecuário (R$ 1.000)", secondary_y=False)
            fig_combined_vab_desmat.update_yaxes(title_text="Desmatamento Total (km²)", secondary_y=True)
            st.plotly_chart(fig_combined_vab_desmat, use_container_width=True)


        with col2_t4:
            st.subheader("Área Plantada (Soja e Milho)")
            df_melted_areas = df_filtered_agro.melt(
                id_vars=['ano', 'municipio'],
                value_vars=['area_soja_ha', 'area_milho_ha'],
                var_name='cultura', value_name='area_ha'
            )
            df_melted_areas['cultura'] = df_melted_areas['cultura'].map({'area_soja_ha': 'Soja', 'area_milho_ha': 'Milho'})

            fig_areas_plantadas = px.line(
                df_melted_areas, x='ano', y='area_ha', color='municipio', line_dash='cultura',
                title='Evolução da Área Plantada (Soja e Milho) em ha',
                labels={'ano':'Ano', 'area_ha':'Área Plantada (ha)', 'cultura':'Cultura'}
            )
            fig_areas_plantadas.update_layout(title_x=0.5)
            st.plotly_chart(fig_areas_plantadas, use_container_width=True)

        st.markdown("---")
        col3_t4, col4_t4 = st.columns(2)
        with col3_t4:
            st.subheader("Evolução do Rebanho Bovino")
            fig_rebanho = px.line(
                df_filtered_agro, x='ano', y='rebanho_bovino_cabecas', color='municipio',
                title='Evolução do Rebanho Bovino (Cabeças)',
                labels={'ano':'Ano', 'rebanho_bovino_cabecas':'Rebanho Bovino (Cabeças)'}
            )
            fig_rebanho.update_layout(title_x=0.5)
            st.plotly_chart(fig_rebanho, use_container_width=True)

        with col4_t4:
            st.subheader("Desmatamento vs. Rebanho Bovino (Ano Selecionado Globalmente)")
            df_scatter_rebanho = df_data[df_data['ano'] == selected_year_global]
            if not df_scatter_rebanho.empty:
                fig_scatter_desmat_rebanho = px.scatter(
                    df_scatter_rebanho, x='rebanho_bovino_cabecas', y='desmatamento_km2',
                    color='municipio', # Pode poluir, mas útil para identificar
                    size='populacao',
                    hover_name='municipio',
                    title=f'Desmatamento vs. Rebanho Bovino em {selected_year_global}',
                    labels={'rebanho_bovino_cabecas':'Rebanho Bovino (Cabeças)', 'desmatamento_km2':'Desmatamento (km²)'}
                )
                fig_scatter_desmat_rebanho.update_layout(title_x=0.5, showlegend=False)
                st.plotly_chart(fig_scatter_desmat_rebanho, use_container_width=True)
            else:
                st.info(f"Não há dados para o ano {selected_year_global} para este gráfico de dispersão.")

        st.markdown("---")
        st.subheader("Proporção do VAB por Setor (Média dos Municípios Selecionados)")
        if not df_filtered_agro.empty:
            df_vab_share = df_filtered_agro.groupby('ano')[['vab_agro_mil_reais', 'vab_industria_mil_reais', 'vab_servicos_mil_reais']].mean().reset_index()
            df_vab_share_melted = df_vab_share.melt(id_vars='ano', var_name='setor_vab', value_name='vab_medio_mil_reais')
            df_vab_share_melted['setor_vab'] = df_vab_share_melted['setor_vab'].map({
                'vab_agro_mil_reais': 'Agropecuária',
                'vab_industria_mil_reais': 'Indústria',
                'vab_servicos_mil_reais': 'Serviços'
            })
            fig_vab_pie_over_time = px.bar(
                 df_vab_share_melted, x='ano', y='vab_medio_mil_reais', color='setor_vab',
                 title='Evolução da Composição Média do VAB (R$ 1.000) nos Municípios Selecionados',
                 labels={'ano':'Ano', 'vab_medio_mil_reais':'VAB Médio (R$ 1.000)'},
                 barmode='stack' # ou 'group'
            )
            fig_vab_pie_over_time.update_layout(title_x=0.5)
            st.plotly_chart(fig_vab_pie_over_time, use_container_width=True)


# --- Footer (Opcional) ---
st.markdown("---")
st.markdown("Desenvolvido como parte do Desafio Zetta Lab 2025.")
st.markdown("Fontes de Dados: IBGE (SIDRA), INPE (TerraBrasilis/Queimadas), MDS (Bolsa Família/Auxílio Brasil - via Portal da Transparência ou dados compilados).")

