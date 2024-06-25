import streamlit as st
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from scipy.stats import norm
import yfinance as yf
from statsmodels.tsa.stattools import acf
from scipy.stats import expon, probplot
from scipy.stats import norm, multivariate_normal

st.set_page_config(page_title="Properties of Financial Data", page_icon="📈")

st.markdown("# Properties of Financial Data")

st.markdown("""
            <div style="text-align: justify;">
            For this example, we use cryptocurrency data provided by Yahoo Finance.
            We analyze two cryptocurrencies, BTC-USD and ETH-USD, because they have
            the highest market capitalization among this class of financial instruments.
            Now, use the buttons below to select the time interval for analysis.

            </div>
            """, unsafe_allow_html=True)

st.markdown("#\n" * 1)

###############################################################################################################

# Definir las fechas de fin
default_end_date = datetime.date.today()  # Establece la fecha de fin por defecto como la fecha de hoy

# Descargar datos preliminares para obtener la fecha mínima común
tickers = ['BTC-USD', 'ETH-USD']
df_prelim = yf.download(tickers, end=default_end_date)['Adj Close']

# Calcular la fecha mínima común entre ambos instrumentos financieros
min_common_date = df_prelim.dropna().index.min().date()

# Crear inputs de fecha en Streamlit
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input(
        "Select Start Date",
        value=min_common_date,
        min_value=min_common_date,
        max_value=default_end_date)

with col2:
    end_date = st.date_input(
        "Select End Date",
        value=default_end_date,
        min_value=start_date,
        max_value=default_end_date)

# Descargar datos de las acciones usando yfinance
df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
df.index = pd.to_datetime(df.index)
df = df.dropna()

# Crear un gráfico de precios ajustados cerrados usando Plotly
fig_adj_close = go.Figure()
for column in df.columns:
    fig_adj_close.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))
fig_adj_close.update_layout(title="Adjusted Close Price for Cryptocurrencies",
                            xaxis_title="Time",
                            yaxis_title="Adjusted Close Price",
                            legend_title="Ticker")
st.plotly_chart(fig_adj_close)

###############################################################################################################

st.markdown("""Then, we analize the daily returns of each cryptocurrency through""")

st.latex(r'''
    X_{t,i}=\log\left(P_{t,i}/P_{t-1,i}\right)
    ''')

st.markdown("""where $P_{t,i}$ denotes the price of cryptocurrency $i$ at time $t$.""")

###############################################################################################################

# Calcular los log returns
log_returns = np.log(df / df.shift(1)).dropna()

# Crear un gráfico de log returns usando Plotly
fig_log_returns = go.Figure()
for column in log_returns.columns:
    fig_log_returns.add_trace(go.Scatter(x=log_returns.index, y=log_returns[column], mode='lines', name=column))
fig_log_returns.update_layout(title="Log Returns for Cryptocurrencies",
                              xaxis_title="Time",
                              yaxis_title="Log Returns",
                              legend_title="Ticker")
st.plotly_chart(fig_log_returns)

###############################################################################################################

st.markdown("""
            <div style="text-align: justify;">
            Considering the previous figures, a natural question arises: do the log-returns follow a normal
            distribution? To investigate this, we fit a histogram for each cryptocurrency with a normal curve,
            using the mean and variance from their classical unbiased and consistent estimators. Additionally,
            we present a Q-Q plot alongside to verify the normality.

            </div>
            """, unsafe_allow_html=True)

st.markdown("#\n" * 1)

###############################################################################################################

# Función para crear histograma con ajuste de distribución normal
def plot_histogram_with_normal_fit(log_returns, title):
    mu, sigma = log_returns.mean(), log_returns.std()
    x = np.linspace(log_returns.min(), log_returns.max(), 100)
    pdf = norm.pdf(x, mu, sigma)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=log_returns, nbinsx=50, histnorm='probability density', name='Histogram'))
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='Normal Distribution'))
    fig.update_layout(
        title=title,
        xaxis_title='Log Returns',
        yaxis_title='Density',
        height=400,
        showlegend=False
    )
    return fig

# Función para crear Q-Q plot con plotly
def qqplot_to_plotly(data, dist, title):
    (osm, osr), (slope, intercept, r) = probplot(data, dist=dist, fit=True)
    line = slope * osm + intercept
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Ordered Data'))
    fig.add_trace(go.Scatter(x=osm, y=line, mode='lines', name='Fit'))
    fig.update_layout(
        title=title,
        xaxis_title="Ordered Data",
        yaxis_title=f"{dist.name.capitalize()} Quantiles",
        height=400,
        showlegend=False
    )
    return fig

# Crear histogramas con ajuste de distribución normal para BTC y ETH
fig_hist_btc = plot_histogram_with_normal_fit(log_returns['BTC-USD'], 'Histogram of Log Returns for BTC-USD')
fig_hist_eth = plot_histogram_with_normal_fit(log_returns['ETH-USD'], 'Histogram of Log Returns for ETH-USD')

# Crear Q-Q Plots con ajuste de distribución normal para BTC y ETH
fig_qq_btc_norm = qqplot_to_plotly(log_returns['BTC-USD'], norm, "Norm. Q-Q Plot for BTC-USD Log Returns")
fig_qq_eth_norm = qqplot_to_plotly(log_returns['ETH-USD'], norm, "Norm. Q-Q Plot for ETH-USD Log Returns")

# Mostrar gráficos en un panel 1x2
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_hist_btc, use_container_width=True)
    st.plotly_chart(fig_hist_eth, use_container_width=True)

with col2:
    st.plotly_chart(fig_qq_btc_norm, use_container_width=True)
    st.plotly_chart(fig_qq_eth_norm, use_container_width=True)

###############################################################################################################

st.markdown("""
            <div style="text-align: justify;">
            We observe that the log-returns for Bitcoin and Ethereum are leptokurtic, as they do not fit the normal
            curve well. Now, we would like to understand the nature of the data further. To do this, it is essential
            to see if the series are correlated with subsequent observations. Therefore, we use the ACF for each
            cryptocurrency. Additionally, to verify the magnitude of the log-returns regardless of the sign,
            we will also consider the absolute values of the series.

            </div>
            """, unsafe_allow_html=True)

st.markdown("#\n" * 1)

###############################################################################################################

# Calcular ACF para los log returns y valores absolutos de log returns para determinar los límites
all_acf_values = []
all_abs_acf_values = []

for column in log_returns.columns:
    acf_values = acf(log_returns[column], nlags=40)
    abs_acf_values = acf(log_returns[column].abs(), nlags=40)
    all_acf_values.extend(acf_values)
    all_abs_acf_values.extend(abs_acf_values)

ylim = (min(all_acf_values + all_abs_acf_values), max(all_acf_values + all_abs_acf_values))

# Función para crear gráficos de ACF usando plotly con límites en el eje y
def plot_acf_to_plotly(series, title, ylim):
    acf_values = acf(series, nlags=40)  # Puedes ajustar el número de lags si lo deseas
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(acf_values)),
        y=acf_values,
        mode='lines+markers',
        name='ACF'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Lag",
        yaxis_title="Autocorrelation",
        yaxis=dict(range=ylim),
        height=400
    )
    return fig

# Crear gráficos de ACF para los log returns y los valores absolutos de los log returns
tickers_acf = [('BTC-USD', 'BTC-USD'), ('ETH-USD', 'ETH-USD')]
for i, (ticker, ticker_abs) in enumerate(tickers_acf):
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acf_log_returns = plot_acf_to_plotly(log_returns[ticker], f"ACF for Log Returns of {ticker}", ylim)
        st.plotly_chart(fig_acf_log_returns, use_container_width=True)
        
    with col2:
        fig_acf_abs_log_returns = plot_acf_to_plotly(log_returns[ticker].abs(), f"ACF for Absolute Log Returns of {ticker_abs}", ylim)
        st.plotly_chart(fig_acf_abs_log_returns, use_container_width=True)

###############################################################################################################

st.markdown("""
            <div style="text-align: justify;">
            From the previous graphs, it is clear that positive and negative returns are correlated.
            This indicates that the log returns are neither independent nor identically distributed.

            </div>
            """, unsafe_allow_html=True)

st.markdown("#\n" * 1)

st.markdown("""
            <div style="text-align: justify;">
            On the other hand, focusing on extreme values, specifically the 100 largest losses for each
            cryptocurrency, if this subset occurred in equispaced periods, they should follow an exponential
            distribution with a common parameter. To verify this, we perform a Q-Q plot using exponential quantiles.

            </div>
            """, unsafe_allow_html=True)

st.markdown("#\n" * 1)

###############################################################################################################

# Obtener las 100 mayores pérdidas (valores negativos más grandes) de los log returns
largest_losses_btc = log_returns['BTC-USD'].nsmallest(100)
largest_losses_eth = log_returns['ETH-USD'].nsmallest(100)

# Crear un gráfico de barras para visualizar las 100 mayores pérdidas
fig_losses_btc = px.bar(largest_losses_btc, title='100 Largest Losses for BTC-USD')
fig_losses_eth = px.bar(largest_losses_eth, title='100 Largest Losses for ETH-USD')

fig_losses_btc.update_layout(showlegend=False)
fig_losses_eth.update_layout(showlegend=False)

# Crear Q-Q plots para las 100 mayores pérdidas con distribución exponencial
fig_qq_btc = qqplot_to_plotly(largest_losses_btc, expon, "Exp. Q-Q Plot for BTC-USD Largest Losses")
fig_qq_eth = qqplot_to_plotly(largest_losses_eth, expon, "Exp. Q-Q Plot for ETH-USD Largest Losses")

# Mostrar gráficos en un panel 2x2
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_losses_btc, use_container_width=True)
with col2:
    st.plotly_chart(fig_qq_btc, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig_losses_eth, use_container_width=True)
with col4:
    st.plotly_chart(fig_qq_eth, use_container_width=True)


###############################################################################################################

st.markdown("""
            <div style="text-align: justify;">
            The data for Bitcoin and Ethereum show a succession of periods without any discernible pattern, 
            as confirmed by the Q-Q plots. This indicates a clustering of extreme values in both cryptocurrencies.

            </div>
            """, unsafe_allow_html=True)

st.markdown("#\n" * 1)

st.markdown("""
            <div style="text-align: justify;">
            Finally, to study the joint structure of both cryptocurrencies, we will create a scatterplot of the
            log returns. Alongside this, we will fit a multivariate normal distribution and simulate the same number
            of samples.

            </div>
            """, unsafe_allow_html=True)

st.markdown("#\n" * 1)

###############################################################################################################

# Calcular los cuantiles del 5%
quantile_5_btc = log_returns['BTC-USD'].quantile(0.05)
quantile_5_eth = log_returns['ETH-USD'].quantile(0.05)

# Crear scatterplot de los log returns de BTC y ETH
fig_scatter_original = go.Figure()

fig_scatter_original.add_trace(go.Scatter(
    x=log_returns['BTC-USD'],
    y=log_returns['ETH-USD'],
    mode='markers',
    marker=dict(size=3),
    name='Log Returns'
))

# Añadir líneas verticales y horizontales para los cuantiles del 5% en los log returns originales
fig_scatter_original.add_shape(
    type='line',
    x0=quantile_5_btc, x1=quantile_5_btc, y0=log_returns['ETH-USD'].min(), y1=log_returns['ETH-USD'].max(),
    line=dict(color='Red', dash='dash'),
    xref='x', yref='y'
)

fig_scatter_original.add_shape(
    type='line',
    x0=log_returns['BTC-USD'].min(), x1=log_returns['BTC-USD'].max(), y0=quantile_5_eth, y1=quantile_5_eth,
    line=dict(color='Red', dash='dash'),
    xref='x', yref='y'
)

fig_scatter_original.update_layout(
    title='Sample Log Returns: BTC-USD vs. ETH-USD',
    xaxis_title='Log Returns BTC-USD',
    yaxis_title='Log Returns ETH-USD',
    showlegend=False
)


# Ajustar una distribución normal multivariada a los log returns
mean = log_returns.mean().values
cov = log_returns.cov().values

# Simular muestras de la distribución normal multivariada
simulated_data = multivariate_normal.rvs(mean=mean, cov=cov, size=len(log_returns))

fig_scatter_simulated = go.Figure()

fig_scatter_simulated.add_trace(go.Scatter(
    x=simulated_data[:, 0],
    y=simulated_data[:, 1],
    mode='markers',
    marker=dict(size=3),
    name='Simulated Data'
))

# Añadir líneas verticales y horizontales para los cuantiles del 5% en los datos simulados
fig_scatter_simulated.add_shape(
    type='line',
    x0=quantile_5_btc, x1=quantile_5_btc, y0=simulated_data[:, 1].min(), y1=simulated_data[:, 1].max(),
    line=dict(color='Red', dash='dash'),
    xref='x', yref='y'
)

fig_scatter_simulated.add_shape(
    type='line',
    x0=simulated_data[:, 0].min(), x1=simulated_data[:, 0].max(), y0=quantile_5_eth, y1=quantile_5_eth,
    line=dict(color='Red', dash='dash'),
    xref='x', yref='y'
)

fig_scatter_simulated.update_layout(
    title='Simulated Log-Returns: BTC-USD vs. ETH-USD',
    xaxis_title='Simulated Log Returns BTC-USD',
    yaxis_title='Simulated Log Returns ETH-USD',
    showlegend=False
)

# Mostrar los scatterplots lado a lado en Streamlit
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_scatter_original, use_container_width=True)
with col2:
    st.plotly_chart(fig_scatter_simulated, use_container_width=True)

# Contar las observaciones bivariadas menores que los cuantiles del 5% en ambos scatterplots
#count_original = np.sum((log_returns['BTC-USD'] < quantile_5_btc) & (log_returns['ETH-USD'] < quantile_5_eth))
#count_simulated = np.sum((simulated_data[:, 0] < quantile_5_btc) & (simulated_data[:, 1] < quantile_5_eth))

# Mostrar los scatterplots lado a lado en Streamlit
#st.markdown(f"Number of observations below 5th quantile (Original Data): {count_original}")
#st.markdown(f"Number of observations below 5th quantile (Simulated Data): {count_simulated}")

###############################################################################################################

st.markdown("""
            <div style="text-align: justify;">
            In periods of volatility, the extreme values of Bitcoin coincide with those of Ethereum. Additionally,
            the dependency structure differs in both cases. This raises more questions than answers, such as: How
            can we quantify extreme dependence? How can we accurately model the joint structure?

            </div>
            """, unsafe_allow_html=True)

st.markdown("#\n" * 1)
