# File: app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Dashboard de Previsões Econômicas")

# --- Generic Loading Functions (no change) ---
@st.cache_resource
def load_model(model_path, columns_path):
    try:
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        return model, model_columns
    except FileNotFoundError:
        st.error(f"Erro: Arquivo de modelo ou colunas não encontrado: {model_path} ou {columns_path}")
        st.info("Por favor, execute os scripts de treinamento primeiro para gerar os arquivos.")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/RESULTADOS/df_final2.csv')
        df = df.sort_values(by=['Município', 'Ano'])
        return df
    except FileNotFoundError:
        st.error("Erro: Arquivo de dados ('data/RESULTADOS/df_final2.csv') não encontrado.")
        return None

# --- Initial Loading (MODIFIED) ---
df_initial = load_data()
pib_model, pib_columns = load_model('models/modelo_pib_pred.joblib', 'models/model_columns.joblib')
vab_model, vab_columns = load_model('models/modelo_vab_pred.joblib', 'models/vab_model_columns.joblib')
beneficios_model, beneficios_columns = load_model('models/modelo_beneficios_pred.joblib', 'models/beneficios_model_columns.joblib')
# MODIFICATION: Load the new respiratory model
respiratorio_model, respiratorio_columns = load_model('models/modelo_respiratorio_pred.joblib', 'models/respiratorio_model_columns.joblib')

# --- Generic Prediction Functions (MODIFIED) ---
def generate_future_predictions(df_city, num_years_to_predict, model, model_cols, target_column_name):
    # MODIFICATION: Add the new feature to the lag list
    features_to_lag = [
        'Desmatamento (km²)', 'PIB per capita (R$)', 'VAB Agropecuária (R$ 1.000)',
        'Focos de Queimada', 'Área plantada soja (ha)', 'Total Rebanho (Bovino)',
        'Total de Benefícios Básicos (Bolsa Família)',
        'Internações por Doenças Respiratórias'
    ]
    df_history = df_city.copy()
    predictions = []
    last_known_year = df_history['Ano'].max()
    for i in range(num_years_to_predict):
        last_year_data = df_history.iloc[-1]
        current_features = {}
        for feature in features_to_lag:
            if feature in last_year_data:
                current_features[f'{feature}_lag1'] = last_year_data[feature]
                previous_year_data = df_history.iloc[-2]
                previous_value = previous_year_data[feature]
                if previous_value != 0:
                    growth = (last_year_data[feature] - previous_value) / previous_value
                else:
                    growth = 0.0
                current_features[f'{feature}_growth'] = growth
        for col in df_history.columns:
            if col not in ['Município', 'Ano'] and col not in current_features:
                 current_features[col] = last_year_data[col]
        input_df_dict = {col: current_features.get(col, 0) for col in model_cols}
        input_df = pd.DataFrame([input_df_dict])[model_cols]
        predicted_value = model.predict(input_df)[0]
        predictions.append({'Ano': last_known_year + 1 + i, 'Predicted Value': predicted_value})
        new_row = last_year_data.copy()
        new_row['Ano'] = last_known_year + 1 + i
        new_row[target_column_name] = predicted_value
        df_history = pd.concat([df_history, new_row.to_frame().T], ignore_index=True)
    return pd.DataFrame(predictions)

@st.cache_data(show_spinner=False)
def generate_state_level_predictions(_df_all_cities, num_years, _model, _model_cols, target_column_name):
    all_predictions = []
    cities_with_enough_data = [city for city, data in _df_all_cities.groupby('Município') if len(data) >= 2]
    progress_bar = st.progress(0, text=f"Calculando previsões para {target_column_name}...")
    for i, city_name in enumerate(cities_with_enough_data):
        city_data = _df_all_cities[_df_all_cities['Município'] == city_name]
        city_predictions = generate_future_predictions(city_data, num_years, _model, _model_cols, target_column_name)
        all_predictions.append(city_predictions)
        progress_bar.progress((i + 1) / len(cities_with_enough_data), text=f"Calculando para: {city_name}")
    progress_bar.empty()
    if not all_predictions: return pd.DataFrame(columns=['Ano', 'Predicted Value'])
    full_state_predictions = pd.concat(all_predictions)
    aggregated_predictions = full_state_predictions.groupby('Ano')['Predicted Value'].mean().reset_index()
    return aggregated_predictions

# --- Generic UI Function (MODIFIED) ---
def create_prediction_tab(df_all_data, selected_view, num_years, model, model_cols, target_column_name, target_friendly_name, y_axis_title, is_monetary=True):
    # ... (code for data preparation is the same)
    if selected_view == "Estado Completo (Média)":
        st.header(f"Análise Agregada: {target_friendly_name} Médio do Estado")
        with st.spinner(f'Gerando previsões agregadas...'):
            future_predictions_df = generate_state_level_predictions(df_all_data, num_years, model, model_cols, target_column_name)
        history_df = df_all_data.groupby('Ano')[target_column_name].mean().reset_index()
        desmatamento_df = df_all_data.groupby('Ano')['Desmatamento (km²)'].mean().reset_index()
        chart_title = f'{target_friendly_name} e Desmatamento Médio no Estado'
    else:
        # ... (rest of data prep is the same)
        city_data = df_all_data[df_all_data['Município'] == selected_view].copy()
        if len(city_data) < 2:
            st.warning(f"O município '{selected_view}' não possui dados históricos suficientes.")
            return
        st.header(f"Análise para {selected_view}: {target_friendly_name}")
        with st.spinner('Gerando previsões...'):
            future_predictions_df = generate_future_predictions(city_data, num_years, model, model_cols, target_column_name)
        history_df = city_data[['Ano', target_column_name]]
        desmatamento_df = city_data[['Ano', 'Desmatamento (km²)']]
        chart_title = f'{target_friendly_name} e Desmatamento para {selected_view}'

    # ... (code for checkbox and chart creation is the same)
    show_deforestation = st.checkbox("Incluir análise de Desmatamento no gráfico", value=True, key=f"defo_check_{target_friendly_name}")
    history_df = history_df.rename(columns={target_column_name: 'Historic Value'})
    future_predictions_df = future_predictions_df.rename(columns={'Predicted Value': 'Predicted Value'})
    last_hist_point = pd.DataFrame([{'Ano': history_df.iloc[-1]['Ano'], 'Predicted Value': history_df.iloc[-1]['Historic Value']}])
    connected_predictions_df = pd.concat([last_hist_point, future_predictions_df], ignore_index=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df['Ano'], y=history_df['Historic Value'], mode='lines+markers', name='Valor Histórico', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=connected_predictions_df['Ano'], y=connected_predictions_df['Predicted Value'], mode='lines+markers', name='Valor Previsto', line=dict(color='firebrick', dash='dash')))
    if show_deforestation:
        fig.add_trace(go.Scatter(x=desmatamento_df['Ano'], y=desmatamento_df['Desmatamento (km²)'], name='Desmatamento (km²)', yaxis='y2', mode='lines', line=dict(color='green', dash='dot', width=2)))
    fig.update_layout(title=chart_title, xaxis_title='Ano', template='plotly_white', hovermode="x unified", legend_title_text='Métricas', yaxis=dict(title=y_axis_title, color='royalblue'), yaxis2=dict(title='Desmatamento (km²)', overlaying='y', side='right', showgrid=False, color='green'))
    st.plotly_chart(fig, use_container_width=True, key=f"prediction_chart_{target_friendly_name}")
    
    st.subheader("Valores Previstos")
    display_preds = future_predictions_df.copy()
    
    # MODIFICATION: Format based on whether the value is monetary or a simple count
    table_column_name = 'Valor Previsto'
    if is_monetary:
        display_preds['Predicted Value'] = display_preds['Predicted Value'].apply(lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        table_column_name = 'Valor Previsto (R$)'
    else:
        display_preds['Predicted Value'] = display_preds['Predicted Value'].apply(lambda x: f"{x:,.0f}")
        
    st.dataframe(display_preds.rename(columns={'Predicted Value': table_column_name}).set_index('Ano'), use_container_width=True)

    with st.expander("Ver dados históricos completos"):
        if selected_view == "Estado Completo (Média)":
             merged_history = pd.merge(history_df.rename(columns={'Historic Value': target_friendly_name}), desmatamento_df, on='Ano')
             st.dataframe(merged_history.set_index('Ano'))
        else:
            st.dataframe(df_all_data[df_all_data['Município'] == selected_view].set_index('Ano'))

# --- SHAP Functions (MODIFIED) ---
@st.cache_data
def prepare_data_for_shap(_df):
    df_temp = _df.copy()
    # MODIFICATION: Add the new feature to the lag list
    features_to_lag = [
        'Desmatamento (km²)', 'PIB per capita (R$)', 'VAB Agropecuária (R$ 1.000)',
        'Focos de Queimada', 'Área plantada soja (ha)', 'Total Rebanho (Bovino)',
        'Total de Benefícios Básicos (Bolsa Família)',
        'Internações por Doenças Respiratórias'
    ]
    for feature in features_to_lag:
        if feature in df_temp.columns:
            df_temp[f'{feature}_lag1'] = df_temp.groupby('Município')[feature].shift(1)
            lag_col = df_temp[f'{feature}_lag1']
            growth = np.where(lag_col != 0, (df_temp[feature] - lag_col) / lag_col, 0)
            df_temp[f'{feature}_growth'] = growth
    df_model_ready = df_temp.dropna().copy()
    df_model_ready.replace([np.inf, -np.inf], 0, inplace=True)
    latest_year = df_model_ready['Ano'].max()
    df_shap = df_model_ready[df_model_ready['Ano'] == latest_year]
    return df_shap

def plot_shap_summary(model, data, model_columns):
    # ... (function content remains the same)
    st.info(f"Gerando análise de importância para o ano mais recente de dados ({data['Ano'].iloc[0]}).")
    X_shap = data.reindex(columns=model_columns, fill_value=0)
    with st.spinner("Calculando valores SHAP... Isso pode levar um momento."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_shap)
    st.subheader("Importância Média das Features")
    st.markdown("Este gráfico mostra o impacto médio de cada feature nas previsões do modelo. Features no topo são as mais importantes.")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)
    st.subheader("Impacto Detalhado das Features")
    st.markdown("""Este gráfico mostra como o valor de uma feature afeta a previsão...""")
    fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, show=False)
    st.pyplot(fig_beeswarm, use_container_width=True)
    plt.close(fig_beeswarm)

# --- Main App Layout (MODIFIED) ---
st.title("📊 Dashboard de Previsões Econômicas, Sociais e de Saúde")

# MODIFICATION: Check for all four models
if df_initial is not None and all([pib_model, vab_model, beneficios_model, respiratorio_model]):
    st.sidebar.header("Parâmetros de Simulação")
    list_of_cities = df_initial['Município'].unique().tolist()
    list_of_cities.sort()
    view_options = ["Estado Completo (Média)"] + list_of_cities
    selected_view = st.sidebar.selectbox('Selecione a Visualização', options=view_options, index=0)
    num_years = st.sidebar.slider('Quantos anos prever no futuro?', 1, 10, 3, key="pred_years")
    st.sidebar.info("Os controles nesta barra lateral se aplicam às abas de previsão.")

    # MODIFICATION: Add the new tab
    tab_pib, tab_vab, tab_beneficios, tab_respiratorio, tab_importance = st.tabs([
        "📈 PIB per Capita", 
        "🚜 VAB Agropecuária", 
        "💰 Benefícios Sociais",
        "🏥 Saúde Respiratória",
        "🔍 Importância das Features"
    ])

    with tab_pib:
        create_prediction_tab(df_initial, selected_view, num_years, pib_model, pib_columns, 'PIB per capita (R$)', 'PIB per Capita', 'PIB per Capita (R$)')
    with tab_vab:
        create_prediction_tab(df_initial, selected_view, num_years, vab_model, vab_columns, 'VAB Agropecuária (R$ 1.000)', 'VAB Agropecuária', 'VAB Agropecuária (R$ 1.000)')
    with tab_beneficios:
        create_prediction_tab(df_initial, selected_view, num_years, beneficios_model, beneficios_columns, 'Total de Benefícios Básicos (Bolsa Família)', 'Benefícios Sociais', 'Total de Benefícios (R$)')
    
    # ADDED: Content for the new respiratory health tab
    with tab_respiratorio:
        create_prediction_tab(
            df_all_data=df_initial,
            selected_view=selected_view,
            num_years=num_years,
            model=respiratorio_model,
            model_cols=respiratorio_columns,
            target_column_name='Internações por Doenças Respiratórias',
            target_friendly_name='Internações Respiratórias',
            y_axis_title='Número de Internações',
            is_monetary=False # Specify that this is not a currency value
        )

    with tab_importance:
        st.header("Análise de Importância das Features (SHAP)")
        st.markdown("Esta seção ajuda a entender quais fatores mais influenciam as previsões dos modelos.")
        shap_data = prepare_data_for_shap(df_initial)
        
        # MODIFICATION: Add new option to the radio button
        model_choice = st.radio(
            "Selecione o modelo para analisar:", 
            ('PIB per Capita', 'VAB Agropecuária', 'Benefícios Sociais', 'Saúde Respiratória'), 
            horizontal=True
        )
        
        if model_choice == 'PIB per Capita':
            plot_shap_summary(pib_model, shap_data, pib_columns)
        elif model_choice == 'VAB Agropecuária':
            plot_shap_summary(vab_model, shap_data, vab_columns)
        elif model_choice == 'Benefícios Sociais':
            plot_shap_summary(beneficios_model, shap_data, beneficios_columns)
        # ADDED: Logic for the new model
        elif model_choice == 'Saúde Respiratória':
            plot_shap_summary(respiratorio_model, shap_data, respiratorio_columns)