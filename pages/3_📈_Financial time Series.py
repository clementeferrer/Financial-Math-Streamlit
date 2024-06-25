################################################################### Libraries
# Streamlit
import streamlit as st
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# Tratamiento de datos
import pandas as pd
import numpy as np

# Gráficos
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Otros
import datetime as dt

###################################################################

st.set_page_config(page_title="Financial time Series", page_icon="🤌")

st.markdown("# Financial time Series: GARCH model 🤌")

st.markdown('''
            <div style="text-align: justify;">
            Standard and kindly proposal: Use our mathematical knowledge to stablish a system with clear
             definitions but also doing high emphazis on relevant Financial "real world" information. Also, trying to write and 
            construct everything to be useful for everyone without the enough technical knowledge.
            ''', unsafe_allow_html=True)

st.header('Motivation')

st.markdown('''
            <div style="text-align: justify;">
            The focus of this page is to stablish in a friendly way the framework of time series analysis in order to explain
             and create a playground of GARCH models, with simulation, study of real cases, etc.
            </div>
            ''', unsafe_allow_html=True)

st.markdown('''
            <div style="text-align: justify;">
            I will asume knownledge about time series analysis.
            </div>
            ''', unsafe_allow_html=True)

############################################################# To play and deplay

st.header('Definition')

st.latex(r'''
        \epsilon_t = \sigma_t \eta_t
    ''')
st.latex(r'''
        \sigma_t^2 = \omega 
        + \sum_{i=1}^p \alpha_i \epsilon_{t-i}^2
        + \sum_{j=1}^q\beta_j\sigma_{t-j}^2
    ''')


st.markdown('''
            <div style="text-align: justify;">
            ---------------------------
            </div>
            ''', unsafe_allow_html=True)

################# Simulación interactiva de GARCH #################
from arch import arch_model
from arch.univariate import GARCH, ConstantMean, SkewStudent, Normal
from arch.__future__ import reindexing

from statsmodels.tsa.stattools import acf, pacf

def acf_garch(a, b, h):
    '''
    Calculate the theoretical autocorrelation function (ACF) for a GARCH(1,1) model.
    
    Parameters:
    a (float): Alpha parameter of the GARCH model.
    b (float): Beta parameter of the GARCH model.
    h (int): Lag for which the ACF is calculated.
    
    Returns:
    float: Theoretical ACF value for lag h.
    '''
    # Calculate theoretical ACF
    p_1 = a * (1 - b * (a + b)) / (1 - (a + b)**2 + a**2)
    p_h = p_1 * (a + b)**(h - 1)
    return p_h

st.markdown('''
            <div style="text-align: justify;">
            Qualitative behaviour of the GARCH(1,1) model... add interpretation about
            α and β...
            </div>
            ''', unsafe_allow_html=True)

st.markdown('''
            <div style="text-align: justify;">
            Estationarity condition:
            </div>
            ''', unsafe_allow_html=True)

st.latex(r'''
        \alpha+\beta<1
    ''')  
   
# Crear inputs en Streamlit
col1, col2= st.columns(2)

with col1:
   n_datos = st.number_input(
       "Number of simulations (integer)",
       min_value=1,
       max_value=1000000,
       value=5000,
       step=50,
       format="%d")

with col2:
   seed_input = st.number_input(
       "Choose a seed value (integer)",
       min_value=1,
       max_value=1000,
       value=123,
       step=1,
       format="%d")
   
# Crear inputs en Streamlit
col1, col2, col3 = st.columns(3)

with col1:
   omega = st.number_input(
       "Enter a float value for omega (0 to 10)",
       min_value=0.0,
       max_value=10.0,
       value=0.1,
       step=0.01,
       format="%.2f")

with col2:
   alpha = st.number_input(
       "Enter a float value for alpha (0 to 1)",
       min_value=0.0,
       max_value=1.0,
       value=0.25,
       step=0.01,
       format="%.2f")
   
with col3:
   beta = st.number_input(
       "Enter a float value for beta (0 to 1)",
       min_value=0.0,
       max_value=1.0,
       value=0.25,
       step=0.01,
       format="%.2f")

# Crear un modelo GARCH utilizando arch_model
modelo_garch = arch_model(None, p=1, o=0, q=1)

# Establecer la secuencia de residuos
rs = np.random.RandomState([seed_input, 189201902, 129129894, 9890437])
modelo_garch.distribution = Normal(seed=rs)

# Definir los parámetros del modelo
cm_params = np.array([0])
garch_params = np.array([omega, alpha, beta])
params = np.concatenate((cm_params, garch_params))

# Simular trayectorias del modelo GARCH
sim_data = modelo_garch.simulate(params=params, nobs=n_datos)

# Inicializar la figura con subplots
fig, axs = plt.subplots(2, 2, figsize=(18, 10))

returns, volatility = sim_data['data'], sim_data['volatility']

# Create plots of returns and volatility
fig_returns = go.Figure()
fig_returns.update_layout(title=f'GARCH(1,1) Model Simulation<br>Returns', 
                          title_font_size=18)
fig_returns.add_trace(go.Scatter(y=returns, mode='lines', name=f'Returns with α = {round(alpha, 4)} and β = {round(beta, 4)}'))
fig_returns.update_layout(legend=dict(orientation="h"), showlegend=True)

fig_volatility = go.Figure()
fig_volatility.update_layout(title=f'GARCH(1,1) Model Simulation<br>Volatility', 
                             title_font_size=18)
fig_volatility.add_trace(go.Scatter(y=volatility, mode='lines', name=f'Volatility with α = {round(alpha, 4)} and β = {round(beta, 4)}', line=dict(color='green')))
fig_volatility.update_layout(legend=dict(orientation="h"), showlegend=True)

# Calculate ACF and PACF
acf_values = acf(np.square(returns), nlags=15)
pacf_values = pacf(np.square(returns), nlags=15)

# Calculate theoretical ACF values and add 1 at the beginning
theoretical_acf_values = [1] + [acf_garch(alpha, beta, h) for h in range(1, 16)]

# Significance bounds
conf_level = 1.96 / np.sqrt(len(np.square(returns)))

# Create ACF plot
fig_acf = go.Figure()
fig_acf.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'))
fig_acf.add_trace(go.Scatter(x=list(range(16)), y=theoretical_acf_values, mode='lines+markers', name='Theoretical ACF'))
fig_acf.add_shape(type="line", x0=0, y0=conf_level, x1=len(acf_values), y1=conf_level,
                  line=dict(color="red", dash="dash"))
fig_acf.add_shape(type="line", x0=0, y0=-conf_level, x1=len(acf_values), y1=-conf_level,
                  line=dict(color="red", dash="dash"))
fig_acf.update_layout(title='GARCH(1,1) Model Simulation<br>Autocorrelation Function (ACF)',
                      xaxis_title='Lags',
                      yaxis_title='ACF Value',
                      title_font_size=18)

# Create PACF plot
fig_pacf = go.Figure()
fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'))
fig_pacf.add_shape(type="line", x0=0, y0=conf_level, x1=len(pacf_values), y1=conf_level,
                   line=dict(color="red", dash="dash"))
fig_pacf.add_shape(type="line", x0=0, y0=-conf_level, x1=len(pacf_values), y1=-conf_level,
                   line=dict(color="red", dash="dash"))
fig_pacf.update_layout(title='GARCH(1,1) Model Simulation<br>Partial Autocorrelation Function (PACF)',
                       xaxis_title='Lags',
                       yaxis_title='PACF Value',
                       title_font_size=18)

# Initial state
if 'show_acf_pacf' not in st.session_state:
    st.session_state.show_acf_pacf = False

if 'show_other_graph' not in st.session_state:
    st.session_state.show_other_graph = True

# Button to change the position of the plots
col1, col2 = st.columns([5,1])
with col1:
    if st.button("Change Graph Position"):
            st.session_state.show_other_graph = not st.session_state.show_other_graph
            
with col2:
    if st.button("Show more"):
        st.session_state.show_acf_pacf = not st.session_state.show_acf_pacf

# Display plots based on user preference
if st.session_state.show_acf_pacf:
    if st.session_state.show_other_graph:
        st.plotly_chart(fig_acf)
        st.plotly_chart(fig_pacf)
    else:
        st.plotly_chart(fig_pacf)
        st.plotly_chart(fig_acf)
else:
    if st.session_state.show_other_graph:
        st.plotly_chart(fig_volatility)
        st.plotly_chart(fig_returns)
    else:
        st.plotly_chart(fig_returns)
        st.plotly_chart(fig_volatility)
        
def simular_GARCH_pq_con_rangos(p, q, alpha_range=(0, 1), beta_range=(0, 1), omega=0.5, nobs=500,
                                custom_seed=123, custom_random_state=1, make_stationarity=True):
    # Establecer la semilla para reproducibilidad
    np.random.seed(seed=custom_seed)

    # Generar parámetros aleatorios para alpha y beta dentro de los rangos especificados
    alpha_array = np.random.uniform(low=alpha_range[0], high=alpha_range[1], size=p)
    beta_array = np.random.uniform(low=beta_range[0], high=beta_range[1], size=q)
    
    # Garantizar estacionaridad
    if np.sum(alpha_array) + np.sum(beta_array) >= 1 and make_stationarity:
        scale_factor = 0.999 / (np.sum(alpha_array) + np.sum(beta_array))
        alpha_array = alpha_array * scale_factor
        beta_array = beta_array * scale_factor

    # Definir los parámetros del modelo
    mu = 0
    cm_params = np.array([mu])
    garch_params = np.concatenate((np.array([omega]), alpha_array, beta_array))
    params = np.concatenate((cm_params, garch_params))

    # Crear un modelo GARCH utilizando arch_model
    modelo_garch = arch_model(None, p=p, o=0, q=q)

    # Establecer la secuencia de residuos
    rs = np.random.RandomState([custom_random_state, 189201902, 129129894, 9890437])
    modelo_garch.distribution = Normal(seed=rs)

    # Simular una única trayectoria del modelo GARCH
    sim_data = modelo_garch.simulate(params=params, nobs=nobs)

    # Obtener los datos simulados
    data, variance = sim_data['data'], sim_data['volatility']

    return data, variance

# Crear inputs en Streamlit
col1, col2= st.columns(2)

with col1:
   input_seed = st.number_input(
       "Parameters seed",
       min_value=1,
       max_value=1000,
       value=123,
       step=1,
       format="%d")

with col2:
   input_random_state = st.number_input(
       "Residual seed",
       min_value=1,
       max_value=1000,
       value=123,
       step=1,
       format="%d")

# Ejemplo de uso más estándar para GARCH(p, q) con rangos específicos para alpha y beta
p = 2
q = 4
alpha_range = (0.3, 0.9)
beta_range = (0.3, 0.9)

returns, volatility = simular_GARCH_pq_con_rangos(p=p, q=q, alpha_range=alpha_range, beta_range=beta_range,
                                                  custom_random_state=input_random_state,
                                                  custom_seed=input_seed)

# Inicializar la figura con subplots
fig, axs = plt.subplots(2, 2, figsize=(18, 10))

# Create plots of returns and volatility
fig_returns = go.Figure()
fig_returns.update_layout(title=f'GARCH(1,1) Model Simulation<br>Returns', 
                          title_font_size=18)
fig_returns.add_trace(go.Scatter(y=returns, mode='lines', name=f'Returns with α = {round(alpha, 4)} and β = {round(beta, 4)}'))
fig_returns.update_layout(legend=dict(orientation="h"), showlegend=True)

fig_volatility = go.Figure()
fig_volatility.update_layout(title=f'GARCH(1,1) Model Simulation<br>Volatility', 
                             title_font_size=18)
fig_volatility.add_trace(go.Scatter(y=volatility, mode='lines', name=f'Volatility with α = {round(alpha, 4)} and β = {round(beta, 4)}', line=dict(color='green')))
fig_volatility.update_layout(legend=dict(orientation="h"), showlegend=True)

# Calculate ACF and PACF
acf_values = acf(np.square(returns), nlags=15)
pacf_values = pacf(np.square(returns), nlags=15)

# Significance bounds
conf_level = 1.96 / np.sqrt(len(np.square(returns)))

# Create ACF plot
fig_acf = go.Figure()
fig_acf.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'))
fig_acf.add_shape(type="line", x0=0, y0=conf_level, x1=len(acf_values), y1=conf_level,
                  line=dict(color="red", dash="dash"))
fig_acf.add_shape(type="line", x0=0, y0=-conf_level, x1=len(acf_values), y1=-conf_level,
                  line=dict(color="red", dash="dash"))
fig_acf.update_layout(title='GARCH(1,1) Model Simulation<br>Autocorrelation Function (ACF)',
                      xaxis_title='Lags',
                      yaxis_title='ACF Value',
                      title_font_size=18)

# Create PACF plot
fig_pacf = go.Figure()
fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'))
fig_pacf.add_shape(type="line", x0=0, y0=conf_level, x1=len(pacf_values), y1=conf_level,
                   line=dict(color="red", dash="dash"))
fig_pacf.add_shape(type="line", x0=0, y0=-conf_level, x1=len(pacf_values), y1=-conf_level,
                   line=dict(color="red", dash="dash"))
fig_pacf.update_layout(title='GARCH(1,1) Model Simulation<br>Partial Autocorrelation Function (PACF)',
                       xaxis_title='Lags',
                       yaxis_title='PACF Value',
                       title_font_size=18)

st.plotly_chart(fig_returns)
st.plotly_chart(fig_volatility)
st.plotly_chart(fig_acf)
st.plotly_chart(fig_pacf)
        

# Initial state
# if 'show_acf_pacf' not in st.session_state:
#     st.session_state.show_acf_pacf = False

# if 'show_other_graph' not in st.session_state:
#     st.session_state.show_other_graph = True

# # Button to change the position of the plots
# col1, col2 = st.columns([5,1])
# with col1:
#     if st.button("Change Graph Position"):
#             st.session_state.show_other_graph = not st.session_state.show_other_graph
            
# with col2:
#     if st.button("Show more"):
#         st.session_state.show_acf_pacf = not st.session_state.show_acf_pacf

# # Display plots based on user preference
# if st.session_state.show_acf_pacf:
#     if st.session_state.show_other_graph:
#         st.plotly_chart(fig_acf)
#         st.plotly_chart(fig_pacf)
#     else:
#         st.plotly_chart(fig_pacf)
#         st.plotly_chart(fig_acf)
# else:
#     if st.session_state.show_other_graph:
#         st.plotly_chart(fig_volatility)
#         st.plotly_chart(fig_returns)
#     else:
#         st.plotly_chart(fig_returns)
#         st.plotly_chart(fig_volatility)
