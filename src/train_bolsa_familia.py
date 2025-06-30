# Salve este código como: modelo_beneficios_pred.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import joblib

# --- 1. Carregar e Preparar Dados ---
# Assumindo que o script está na raiz do projeto e os dados em data/RESULTADOS/
try:
    df = pd.read_csv('data/RESULTADOS/df_final.csv')
except FileNotFoundError:
    print("Arquivo 'data/RESULTADOS/df_final.csv' não encontrado. Verifique o caminho.")
    exit()
    
df = df.sort_values(by=['Município', 'Ano'])

# --- 2. Feature Engineering ---
# Criar Lags e Taxas de Crescimento
features_to_lag = [
    'Desmatamento (km²)', 'PIB per capita (R$)', 'VAB Agropecuária (R$ 1.000)',
    'Focos de Queimada', 'Área plantada soja (ha)', 'Total Rebanho (Bovino)',
    'Total de Benefícios Básicos (Bolsa Família)' # Adicionando o próprio benefício como feature
]

print("Criando features de lag e crescimento...")
for feature in features_to_lag:
    df[f'{feature}_lag1'] = df.groupby('Município')[feature].shift(1)
    df[f'{feature}_growth'] = df.groupby('Município')[feature].pct_change()

# --- 3. Definir Alvo e Features ---
# O alvo é o Total de Benefícios do ano seguinte
TARGET_COLUMN = 'Total de Benefícios Básicos (Bolsa Família)'
df['beneficios_target'] = df.groupby('Município')[TARGET_COLUMN].shift(-1)

# Remover linhas com valores nulos criados pelos lags/target
df_model = df.dropna().copy()
df_model.replace([np.inf, -np.inf], 0, inplace=True)

# Definir X e y
y = df_model['beneficios_target']
X = df_model.drop(columns=['beneficios_target', 'Município', TARGET_COLUMN]) # Remover o alvo original também

# --- 4. Divisão Temporal ---
# Treinar com dados até 2021, testar em 2022
print("Dividindo os dados em treino e teste...")
X_train = X[X['Ano'] <= 2021]
y_train = y[X['Ano'] <= 2021]
X_test = X[X['Ano'] > 2021]
y_test = y[X['Ano'] > 2021]

# Remover 'Ano' das features após a divisão
X_train = X_train.drop(columns=['Ano'])
X_test = X_test.drop(columns=['Ano'])

# --- 5. Treinamento do Modelo ---
print("Treinando o modelo LightGBM...")
lgb_params = {
    'objective': 'regression_l1',
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

print("\n--- Resultados do Modelo de Benefícios ---")
print(f"RMSE no conjunto de teste: {rmse:,.2f}")
print(f"R² no conjunto de teste: {r2:.2f}")

# Salvar o modelo e as colunas para usar no dashboard
MODEL_FILE = 'models/modelo_beneficios_pred.joblib'
COLUMNS_FILE = 'models/beneficios_model_columns.joblib'
joblib.dump(model, MODEL_FILE)
joblib.dump(X_train.columns.tolist(), COLUMNS_FILE)
print(f"\nModelo salvo como '{MODEL_FILE}'")
print(f"Colunas salvas como '{COLUMNS_FILE}'")

# --- Análise de Importância com SHAP (Opcional, para visualização) ---
print("\nGerando gráficos SHAP...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizar a importância global (gráfico de barras)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True, plot_size=(10, 8))

# Visualizar o impacto de cada feature (gráfico de pontos/beeswarm)
shap.summary_plot(shap_values, X_test, show=True, plot_size=(10, 8))