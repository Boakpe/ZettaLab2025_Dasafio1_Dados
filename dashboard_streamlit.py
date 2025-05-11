import pandas as pd
import streamlit as st
import plotly.express as px
import json

# --- 0. Streamlit Page Configuration (Optional, but good practice) ---
st.set_page_config(layout="wide", page_title="Desmatamento no Pará")

# Set max width to 1200px
st.html("""
    <style>
        .stMainBlockContainer {
            max-width:1400px;
        }
    </style>
    """
)

# --- 1. Data Loading and Preprocessing ---
# Use st.cache_data for functions that load data to improve performance
@st.cache_data
def load_data():
    # Ensure your CSV file is in the 'data/DESMATAMENTO/' subdirectory relative to your script
    # or provide an absolute path.
    try:
        df_desmatamento_raw = pd.read_csv('data/DESMATAMENTO/terrabrasilis_legal_amazon_4_30_2025,_1_15_32 PM.csv', sep=';')
        df_desmatamento = df_desmatamento_raw.copy()
        df_desmatamento['areakm'] = df_desmatamento['areakm'].str.replace(',', '.').astype(float)
        df_desmatamento = df_desmatamento[df_desmatamento['year'] != 2007] # Filter out year 2007
        df_desmatamento['geocode_ibge'] = df_desmatamento['geocode_ibge'].astype(str)
        return df_desmatamento
    except FileNotFoundError:
        st.error("ERRO: Arquivo 'terrabrasilis_legal_amazon_4_30_2025,_1_15_32 PM.csv' não encontrado na pasta 'data/DESMATAMENTO/'. Verifique o caminho.")
        return pd.DataFrame() # Return empty DataFrame to prevent further errors
    except Exception as e:
        st.error(f"ERRO ao carregar ou processar dados de desmatamento: {e}")
        return pd.DataFrame()

df_desmatamento = load_data()

# --- Load GeoJSON ---
geojson_path = 'data/geojson/municipios_pa.json'
feature_id_key_geojson = 'properties.id' # As per your original script

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
st.title("Dashboard de Desmatamento no Estado do Pará")

# --- Create Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Panorama do Desmatamento",
    "Indicadores Socioeconômicos",
    "Correlações e Relações",
    "Análise Setorial"
])

# --- Tab 1: Panorama do Desmatamento ---
with tab1:
    st.header("Panorama Geral do Desmatamento no Pará")

    if df_desmatamento.empty:
        st.warning("Não foi possível carregar os dados de desmatamento. As visualizações não podem ser geradas.")
    else:
        # --- 2. Prepare Data for Plots (specific to this tab) ---
        df_desmatamento_estado_para = df_desmatamento.groupby('year')['areakm'].sum().reset_index()
        df_top_10_municipios_acumulado = df_desmatamento.groupby('municipality')['areakm'].sum().reset_index()
        df_top_10_municipios_acumulado = df_top_10_municipios_acumulado.nlargest(10, 'areakm')

        # --- 3. Create Figures for Static Plots (specific to this tab) ---
        fig_total_para_evolucao = px.line(
            df_desmatamento_estado_para,
            x='year',
            y='areakm',
            title='Evolução do Desmatamento Total no Pará (km²)',
            labels={'year': 'Ano', 'areakm': 'Área Desmatada (km²)'},
            markers=True
        )
        fig_total_para_evolucao.update_layout(title_x=0.5)

        fig_top_10_municipios = px.bar(
            df_top_10_municipios_acumulado,
            x='municipality',
            y='areakm',
            title='Top 10 Municípios com Maior Desmatamento Acumulado no Pará (km²)',
            labels={'municipality': 'Município', 'areakm': 'Área Desmatada Total (km²)'}
        )
        fig_top_10_municipios.update_layout(xaxis_tickangle=-45, title_x=0.5)

        fig_evolucao_municipios = px.line(
            df_desmatamento,
            x='year',
            y='areakm',
            color='municipality',
            title='Evolução do Desmatamento por Município no Pará (km²)',
            labels={'year': 'Ano', 'areakm': 'Área Desmatada (km²)', 'municipality': 'Município'}
        )
        fig_evolucao_municipios.update_layout(title_x=0.5, showlegend=False) # Legend might be too cluttered

        # --- Map Section ---
        st.subheader("Mapa Interativo: Desmatamento Municipal por Ano")

        available_years = sorted(df_desmatamento['year'].unique())
        if available_years: # Check if there are available years
            selected_year = st.slider(
                "Selecione o Ano:",
                min_value=int(min(available_years)),
                max_value=int(max(available_years)),
                value=int(min(available_years)), # Default to the most recent year
                step=1,
                format="%d",
            )

            if geojson_para is None or feature_id_key_geojson is None:
                st.error(f"Não foi possível carregar o GeoJSON dos municípios ('{geojson_path}') ou a chave de ID não foi identificada. O mapa não pode ser exibido.")
            else:
                filtered_df_map = df_desmatamento[df_desmatamento['year'] == selected_year]
                max_areakm_global = df_desmatamento['areakm'].max() # Use global max for consistent color scale across years

                fig_map = px.choropleth_mapbox(
                    filtered_df_map,
                    geojson=geojson_para,
                    locations='geocode_ibge',
                    featureidkey=feature_id_key_geojson,
                    color='areakm',
                    color_continuous_scale="Reds",
                    range_color=(0, max_areakm_global),
                    mapbox_style="carto-positron",
                    zoom=4.5,
                    center={"lat": -3.5, "lon": -52.5},
                    opacity=0.7,
                    labels={'areakm': 'Área Desmatada (km²)'},
                    hover_name='municipality',
                    hover_data={'areakm': ':.2f km²', 'geocode_ibge': False, 'year': True}
                )
                fig_map.update_layout(
                    margin={"r":0,"t":0,"l":0,"b":0},
                )
                st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Não há anos disponíveis nos dados para exibir o mapa.")


        # --- Static Plots Section (Side-by-Side) ---
        st.markdown("---")
        st.subheader("Análises Agregadas do Desmatamento")
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig_total_para_evolucao, use_container_width=True)

        with col2:
            st.plotly_chart(fig_top_10_municipios, use_container_width=True)

        # --- Full-Width Static Plot ---
        st.markdown("---")
        st.subheader("Evolução Detalhada por Município")

        # Add filter bar for municipalities
        # Get the top 10 municipalities by total deforestation
        top_10_municipalities = (df_desmatamento.groupby('municipality')['areakm']
                       .sum()
                       .sort_values(ascending=False)
                       .head(10)
                       .index
                       .tolist())
        
        # Create multiselect with all municipalities but default to top 10
        all_municipalities = sorted(df_desmatamento['municipality'].unique())
        selected_municipalities = st.multiselect(
            "Selecione os municípios para visualizar:",
            options=all_municipalities,
            default=top_10_municipalities  # Default to top 10 municipalities
        )

        if selected_municipalities:
            df_filtered_municipios = df_desmatamento[df_desmatamento['municipality'].isin(selected_municipalities)]
            fig_evolucao_municipios_filtered = px.line(
                df_filtered_municipios,
                x='year',
                y='areakm',
                color='municipality',
                title='Evolução do Desmatamento por Município no Pará (km²)',
                labels={'year': 'Ano', 'areakm': 'Área Desmatada (km²)', 'municipality': 'Município'}
            )
            fig_evolucao_municipios_filtered.update_layout(title_x=0.5, showlegend=True)
            st.plotly_chart(fig_evolucao_municipios_filtered, use_container_width=True)
        else:
            st.info("Selecione pelo menos um município para visualizar o gráfico.")

# --- Tab 2: Indicadores Socioeconômicos ---
with tab2:
    st.header("Indicadores Socioeconômicos e Desmatamento")
    st.write("Conteúdo para a análise de indicadores socioeconômicos e sua relação com o desmatamento será adicionado aqui.")
    st.info("Possíveis análises: Relação entre IDH, PIB per capita, densidade populacional e taxas de desmatamento.")
    # Placeholder for future content

# --- Tab 3: Correlações e Relações ---
with tab3:
    st.header("Correlações e Relações entre Variáveis")
    st.write("Conteúdo para explorar correlações entre desmatamento e outras variáveis (ex: atividades agropecuárias, focos de calor) será adicionado aqui.")
    st.info("Possíveis análises: Matriz de correlação, gráficos de dispersão interativos.")
    # Placeholder for future content

# --- Tab 4: Análise Setorial ---
with tab4:
    st.header("Análise Setorial do Desmatamento")
    st.write("Conteúdo para analisar o desmatamento por diferentes setores (ex: agricultura, pecuária, mineração, infraestrutura) será adicionado aqui.")
    st.info("Possíveis análises: Gráficos de pizza/barra mostrando a contribuição de cada setor para o desmatamento, se os dados permitirem.")
    # Placeholder for future content