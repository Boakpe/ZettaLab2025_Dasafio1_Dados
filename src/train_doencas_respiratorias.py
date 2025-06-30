# Save this code as: modelo_respiratorio_pred.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import shap
import joblib

# --- 1. Carregar e Preparar Dados ---
try:
    df = pd.read_csv('data/RESULTADOS/df_final2.csv')
except FileNotFoundError:
    print("Arquivo 'data/RESULTADOS/df_final2.csv' não encontrado. Verifique o caminho.")
    exit()
    
df = df.sort_values(by=['Município', 'Ano'])

# --- 2. Feature Engineering ---
features_to_lag = [
    'Desmatamento (km²)', 'PIB per capita (R$)', 'VAB Agropecuária (R$ 1.000)',
    'Focos de Queimada', 'Área plantada soja (ha)', 'Total Rebanho (Bovino)',
    'Total de Benefícios Básicos (Bolsa Família)',
    'Internações por Doenças Respiratórias' # Adicionando a nova feature alvo
]

print("Criando features de lag e crescimento...")
for feature in features_to_lag:
    if feature in df.columns:
        df[f'{feature}_lag1'] = df.groupby('Município')[feature].shift(1)
        df[f'{feature}_growth'] = df.groupby('Município')[feature].pct_change()

# --- 3. Definir Alvo e Features ---
TARGET_COLUMN = 'Internações por Doenças Respiratórias'
df['respiratorio_target'] = df.groupby('Município')[TARGET_COLUMN].shift(-1)

df_model = df.dropna().copy()
df_model.replace([np.inf, -np.inf], 0, inplace=True)

y = df_model['respiratorio_target']
# Remover o alvo original e identificadores das features
X = df_model.drop(columns=['respiratorio_target', 'Município', TARGET_COLUMN]) 

# --- 4. Divisão Temporal ---
print("Dividindo os dados em treino e teste...")
X_train = X[X['Ano'] <= 2021]
y_train = y[X['Ano'] <= 2021]
X_test = X[X['Ano'] > 2021]
y_test = y[X['Ano'] > 2021]

X_train = X_train.drop(columns=['Ano'])
X_test = X_test.drop(columns=['Ano'])

# --- 5. Treinamento do Modelo ---
print("Treinando o modelo LightGBM para doenças respiratórias...")
lgb_params = {
    'objective': 'regression_l1', # MAE
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

model = lgb.LGBMRegressor(**lgb_params)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(100, verbose=False)])

# --- 6. Avaliação e Análise ---
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- Resultados do Modelo de Internações Respiratórias ---")
print(f"RMSE no conjunto de teste: {rmse:,.2f}")
print(f"R² no conjunto de teste: {r2:.2f}")

# Salvar o modelo e as colunas
MODEL_FILE = 'models/modelo_respiratorio_pred.joblib'
COLUMNS_FILE = 'models/respiratorio_model_columns.joblib'
joblib.dump(model, MODEL_FILE)
joblib.dump(X_train.columns.tolist(), COLUMNS_FILE)
print(f"\nModelo salvo como '{MODEL_FILE}'")
print(f"Colunas salvas como '{COLUMNS_FILE}'")

# --- Análise de Importância com SHAP (Opcional) ---
print("\nGerando gráficos SHAP...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True, plot_size=(10, 8))
shap.summary_plot(shap_values, X_test, show=True, plot_size=(10, 8))