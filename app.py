import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import triang
import numpy as np

# Carga de datos
@st.cache
def load_data():
    return pd.read_csv('data.csv')

data = load_data()

# Creación del modelo de regresión lineal
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
model = LinearRegression().fit(X, y)

# Función de simulación de Montecarlo
def monte_carlo_simulation(n=1000):
    tv_min, tv_max = X['TV'].min(), X['TV'].max()
    radio_min, radio_max = X['Radio'].min(), X['Radio'].max()
    newspaper_min, newspaper_max = X['Newspaper'].min(), X['Newspaper'].max()

    tv_random = triang.rvs((X['TV'].median() - tv_min) / (tv_max - tv_min), loc=tv_min, scale=tv_max - tv_min, size=n)
    radio_random = triang.rvs((X['Radio'].median() - radio_min) / (radio_max - radio_min), loc=radio_min, scale=radio_max - radio_min, size=n)
    newspaper_random = triang.rvs((X['Newspaper'].median() - newspaper_min) / (newspaper_max - newspaper_min), loc=newspaper_min, scale=newspaper_max - newspaper_min, size=n)
    
    max_sales = -np.inf
    best_budgets = None
    
    for tv, radio, newspaper in zip(tv_random, radio_random, newspaper_random):
        sales_pred = model.predict([[tv, radio, newspaper]])[0]
        if sales_pred > max_sales:
            max_sales = sales_pred
            best_budgets = tv, radio, newspaper
            
    return best_budgets, max_sales

# Interfaz Streamlit
st.title('Simulación de Montecarlo para Inversión Publicitaria')

# Número de simulaciones
num_simulations = st.number_input('Número de simulaciones', min_value=100, value=1000)

# Botón para ejecutar simulación
if st.button('Ejecutar simulación'):
    best_budgets, max_sales = monte_carlo_simulation(num_simulations)
    st.write(f'Mejor presupuesto para TV: ${best_budgets[0]:.2f}')
    st.write(f'Mejor presupuesto para Radio: ${best_budgets[1]:.2f}')
    st.write(f'Mejor presupuesto para Periódico: ${best_budgets[2]:.2f}')
    st.write(f'Ventas máximas previstas: {max_sales:.2f}')

st.write('Distribución de los datos:')
st.write(data)


