import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import triang
import numpy as np
import matplotlib.pyplot as plt

# Carga de datos
@st.cache
def load_data():
    return pd.read_csv('data.csv')

data = load_data()

# Creación del modelo de regresión lineal
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
model = LinearRegression().fit(X, y)

# Función para mostrar el gráfico de dispersión
def plot_regression(data, model):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    features = ['TV', 'Radio', 'Newspaper']

    for i, feature in enumerate(features):
        ax[i].scatter(data[feature], y, alpha=0.5)
        ax[i].plot(data[feature], model.predict(data), color='red')
        ax[i].set_title(f'{feature} vs. Sales')
        ax[i].set_xlabel(feature)
        ax[i].set_ylabel('Sales')

    return fig

def monte_carlo_simulation(num_simulations):
    min_values = data[['TV', 'Radio', 'Newspaper']].min().values
    max_values = data[['TV', 'Radio', 'Newspaper']].max().values
    mode_values = data[['TV', 'Radio', 'Newspaper']].mode().iloc[0].values

    best_budgets = [0, 0, 0]
    max_sales = 0

    for _ in range(num_simulations):
        tv_budget = triang.rvs((mode_values[0]-min_values[0])/(max_values[0]-min_values[0]),
                               loc=min_values[0], scale=max_values[0]-min_values[0])
        radio_budget = triang.rvs((mode_values[1]-min_values[1])/(max_values[1]-min_values[1]),
                                  loc=min_values[1], scale=max_values[1]-min_values[1])
        newspaper_budget = triang.rvs((mode_values[2]-min_values[2])/(max_values[2]-min_values[2]),
                                      loc=min_values[2], scale=max_values[2]-min_values[2])

        predicted_sales = model.predict([[tv_budget, radio_budget, newspaper_budget]])[0]
        
        if predicted_sales > max_sales:
            max_sales = predicted_sales
            best_budgets = [tv_budget, radio_budget, newspaper_budget]

    return best_budgets, max_sales

# Interfaz Streamlit
st.title('Simulación de Montecarlo para Inversión Publicitaria')

# Número de simulaciones
num_simulations = st.number_input('Número de simulaciones', min_value=100, value=1000)

# Mostrar el gráfico de regresión
st.write('Gráfico de regresión lineal:')
st.pyplot(plot_regression(X, model))

# Botón para ejecutar simulación
if st.button('Ejecutar simulación'):
    best_budgets, max_sales = monte_carlo_simulation(num_simulations)
    st.write(f'Mejor presupuesto para TV: ${best_budgets[0]:.2f}')
    st.write(f'Mejor presupuesto para Radio: ${best_budgets[1]:.2f}')
    st.write(f'Mejor presupuesto para Periódico: ${best_budgets[2]:.2f}')
    st.write(f'Ventas máximas previstas: {max_sales:.2f}')


