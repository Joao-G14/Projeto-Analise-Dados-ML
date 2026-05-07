import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tamanho_base = 2000
df = pd.DataFrame({
    'Idade': np.random.randint(18, 70, tamanho_base),
    'Tempo_Contrato_Meses': np.random.randint(1, 48, tamanho_base),
    'Gasto_Mensal': np.random.uniform(50, 400, tamanho_base),
    'Chamados_Suporte': np.random.randint(0, 10, tamanho_base)
})

probabilidade = (df['Chamados_Suporte'] * 0.1) - (df['Tempo_Contrato_Meses'] * 0.02)
df['Churn'] = np.where(probabilidade + np.random.normal(0, 0.2, tamanho_base) > 0.3, 1, 0)

X = df.drop('Churn', axis=1)
y = df['Churn']             

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

modelo = Sequential()

modelo.add(Dense(16, input_dim=X_treino_scaled.shape[1], activation='relu'))

modelo.add(Dense(8, activation='relu'))

modelo.add(Dense(1, activation='sigmoid'))

modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Iniciando o treinamento do modelo...")
historico = modelo.fit(X_treino_scaled, y_treino, epochs=50, batch_size=32, validation_split=0.2, verbose=0)


perda, acuracia = modelo.evaluate(X_teste_scaled, y_teste, verbose=0)
print(f"\nAcurácia do Modelo nos Dados de Teste: {acuracia * 100:.2f}%")

novo_cliente = np.array([[30, 5, 150, 8]])
novo_cliente_scaled = scaler.transform(novo_cliente)
previsao = modelo.predict(novo_cliente_scaled, verbose=0)

status = "Risco de Cancelamento (Churn)" if previsao[0][0] > 0.5 else "Cliente Retido"
print(f"\nAnálise do Novo Cliente: {status} (Probabilidade: {previsao[0][0]*100:.2f}%)")