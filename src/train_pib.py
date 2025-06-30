import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import joblib # Para salvar o modelo

# --- 1. Carregar e Preparar Dados ---
df = pd.read_csv('data/RESULTADOS/df_final.csv')
df = df.sort_values(by=['Município', 'Ano'])

# --- 2. Feature Engineering ---
# Criar Lags e Deltas
features_to_lag = [
    'Desmatamento (km²)', 'PIB per capita (R$)', 'VAB Agropecuária (R$ 1.000)',
    'Focos de Queimada', 'Área plantada soja (ha)', 'Total Rebanho (Bovino)'
]

for feature in features_to_lag:
    # Lag de 1 ano
    df[f'{feature}_lag1'] = df.groupby('Município')[feature].shift(1)
    # Taxa de crescimento em relação ao ano anterior
    df[f'{feature}_growth'] = df.groupby('Município')[feature].pct_change()

# --- 3. Definir Alvo e Features ---
# O alvo é o PIB per capita do ano seguinte
df['pib_per_capita_target'] = df.groupby('Município')['PIB per capita (R$)'].shift(-1)

# Remover linhas com valores nulos criados pelos lags/target
df_model = df.dropna().copy()
df_model.replace([np.inf, -np.inf], 0, inplace=True) # Tratar possíveis divisões por zero

# Definir X e y
y = df_model['pib_per_capita_target']
X = df_model.drop(columns=['pib_per_capita_target', 'Município']) # Município não é uma feature numérica

# --- 4. Divisão Temporal ---
# Treinar com dados até 2021, testar em 2022
X_train = X[X['Ano'] <= 2021]
y_train = y[X['Ano'] <= 2021]
X_test = X[X['Ano'] > 2021]
y_test = y[X['Ano'] > 2021]

# Remover 'Ano' das features após a divisão
X_train = X_train.drop(columns=['Ano'])
X_test = X_test.drop(columns=['Ano'])

# --- 5. Treinamento do Modelo ---
# Usando LightGBM
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
          callbacks=[lgb.early_stopping(100, verbose=True)])

# --- 6. Avaliação e Análise ---
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"RMSE no conjunto de teste: {rmse:.2f}")
print(f"R² no conjunto de teste: {r2:.2f}")

# Salvar o modelo e as colunas para usar no dashboard
joblib.dump(model, 'models/modelo_pib_pred.joblib')
joblib.dump(X_train.columns.tolist(), 'models/model_columns.joblib')


# --- Análise de Importância com SHAP ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualizar a importância global
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# Visualizar o impacto de cada feature
shap.summary_plot(shap_values, X_test, show=True)