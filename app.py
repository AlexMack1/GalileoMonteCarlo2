import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import triang

# Cargar datos
@st.cache  # Esta función permite que los datos solo se carguen una vez y no en cada reejecución
def load_data():
    return pd.read_csv('data.csv')

data = load_data()

# Modelo de regresión lineal
def train_linear_regression(data):
    X = data[['TV', 'Newspaper', 'Radio']]
    y = data['Sales']
    model = LinearRegression().fit(X, y)
    return model

model = train_linear_regression(data)

# Distribución triangular para generación de números aleatorios
def generate_random(media):
    c = (media.median() - media.min()) / (media.max() - media.min())
    return triang.rvs(c=c, loc=media.min(), scale=media.max()-media.min())

# Simulación de Montecarlo
def monte_carlo_simulation(N=10000):
    results = []
    for _ in range(N):
        random_tv = generate_random(data['TV'])
        random_radio = generate_random(data['Radio'])
        random_newspaper = generate_random(data['Newspaper'])
        predicted_sales = model.predict([[random_tv, random_newspaper, random_radio]])[0]
        results.append((predicted_sales, random_tv, random_newspaper, random_radio))
    return max(results)

best_sales, best_tv, best_newspaper, best_radio = monte_carlo_simulation()

# Normalización y obtención de porcentajes
total_budget = best_tv + best_newspaper + best_radio
percent_tv = best_tv / total_budget
percent_newspaper = best_newspaper / total_budget
percent_radio = best_radio / total_budget

# Interfaz de Streamlit
st.title('Simulación de Inversión en Publicidad')

# Entradas (por ejemplo, puedes permitir que los usuarios introduzcan presupuestos)
budget = st.number_input('Introduce el presupuesto total:', value=1000.0)

st.write(f"Presupuesto óptimo para TV: ${budget * percent_tv:.2f}")
st.write(f"Presupuesto óptimo para Periódico: ${budget * percent_newspaper:.2f}")
st.write(f"Presupuesto óptimo para Radio: ${budget * percent_radio:.2f}")

st.write(f"Ventas proyectadas: ${best_sales:.2f}")

