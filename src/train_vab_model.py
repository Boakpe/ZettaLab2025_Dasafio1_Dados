import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- 1. Carregar e Preparar Dados ---
print("Carregando dados...")
df = pd.read_csv('data/RESULTADOS/df_final.csv')
df = df.sort_values(by=['Município', 'Ano'])

# --- 2. Feature Engineering ---
# As features de lag são as mesmas
features_to_lag = [
    'Desmatamento (km²)', 'PIB per capita (R$)', 'VAB Agropecuária (R$ 1.000)',
    'Focos de Queimada', 'Área plantada soja (ha)', 'Total Rebanho (Bovino)'
]

print("Criando features de lag e growth...")
for feature in features_to_lag:
    df[f'{feature}_lag1'] = df.groupby('Município')[feature].shift(1)
    df[f'{feature}_growth'] = df.groupby('Município')[feature].pct_change()

# --- 3. Definir Alvo e Features (MUDANÇA AQUI) ---
# O alvo agora é o VAB da Agropecuária do ano seguinte
TARGET_VAB = 'VAB Agropecuária (R$ 1.000)'
df['vab_agro_target'] = df.groupby('Município')[TARGET_VAB].shift(-1)

# Remover linhas com valores nulos criados pelos lags/target
df_model = df.dropna().copy()
df_model.replace([np.inf, -np.inf], 0, inplace=True)

# Definir X e y
y = df_model['vab_agro_target']
# Município não é uma feature numérica, e o alvo é removido
X = df_model.drop(columns=['vab_agro_target', 'Município']) 

# --- 4. Divisão Temporal ---
print("Dividindo os dados em treino e teste...")
X_train = X[X['Ano'] <= 2021]
y_train = y[X['Ano'] <= 2021]
X_test = X[X['Ano'] > 2021]
y_test = y[X['Ano'] > 2021]

# Remover 'Ano' das features após a divisão
X_train = X_train.drop(columns=['Ano'])
X_test = X_test.drop(columns=['Ano'])

# --- 5. Treinamento do Modelo ---
print("Treinando o modelo LightGBM para VAB Agropecuária...")
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

model_vab = lgb.LGBMRegressor(**lgb_params)
model_vab.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

# --- 6. Avaliação e Salvamento ---
predictions = model_vab.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- Resultados do Modelo VAB Agropecuária ---")
print(f"RMSE no conjunto de teste: {rmse:,.2f}")
print(f"R² no conjunto de teste: {r2:.2f}")

# Salvar o novo modelo e suas colunas com nomes diferentes (MUDANÇA AQUI)
joblib.dump(model_vab, 'models/modelo_vab_pred.joblib')
joblib.dump(X_train.columns.tolist(), 'models/vab_model_columns.joblib')
print("\nModelo para VAB Agropecuária ('modelo_vab_pred.joblib') salvo com sucesso!")