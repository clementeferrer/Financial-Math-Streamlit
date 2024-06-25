import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from scipy.stats import norm
import yfinance as yf

st.set_page_config(page_title="Value at Risk and Expected Shortfall", page_icon="📈")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

st.markdown("# Value at Risk and Expected Shortfall")

st.header('Motivation')

st.markdown('''
            <div style="text-align: justify;">
            This is an actual problem from the realm of Quantitative Risk Management.
            Bigger banks and insurance companies are required to compute (on a daily
            basis) risk measures such as VaR and ES based on loss distributions
            (according to the Basel II/Solvency II guidelines).
            </div>
            ''', unsafe_allow_html=True)

st.header('Remark')

st.markdown(""" For a loss $L\\sim F_L$, value-at-risk (VaR) at confidence level 
            $\\alpha\\in (0,1)$ is defined by """)

st.latex(r'''
    \operatorname{VaR}_\alpha=\operatorname{VaR}_\alpha(L)=F_L^{\leftarrow}(\alpha)
         =\inf \left\{x \in \mathbb{R}: F_L(x) \geq \alpha\right\},
    ''')


st.markdown("""
            that is, the $\\alpha$-quantile of the distribution function $F_L$ of loss $L$. While, 
            expected shortfall is given by
            """)

st.latex(r'''
    \operatorname{ES}_\alpha=\operatorname{ES}_\alpha(L)=\frac{1}{1-\alpha} 
         \int_\alpha^1 \operatorname{VaR}_u(L) \mathrm{d} u.
    ''')

st.markdown("""Note that $\\operatorname{ES}_\\alpha$ is the average of $\\operatorname{VaR}_\\alpha$
            over all $u\\geq \\alpha$.""")

st.header('Real Data Application')

st.markdown("""
            <div style="text-align: justify;">
            For this example, we use stock data provided by Yahoo Finance.
            We analyze four stocks: Apple, Meta, Citigroup, and Disney, identified
            by their respective tickers: AAPL, META, C, and DIS. Now, with the buttons below,
            select the time interval to analyze.

            </div>
            """, unsafe_allow_html=True)

st.markdown("#\n" * 1)

# Definir las fechas de inicio y fin
default_start_date = datetime.date(2018, 1, 1)
default_end_date = datetime.date.today()  # Establece la fecha de fin por defecto como la fecha de hoy

# Crear inputs de fecha en Streamlit
col1, col2= st.columns(2)

with col1:
   start_date = st.date_input(
    "Select Start Date", 
    value=default_start_date, 
    min_value=datetime.date(2018, 1, 1), 
    max_value=default_end_date)

with col2:
   end_date = st.date_input(
    "Select End Date", 
    value=default_end_date, 
    min_value=start_date, 
    max_value=datetime.date.today())


# Descargar datos de las acciones usando yfinance
df = yf.download(['AAPL', 'META', 'C', 'DIS'], start=start_date, end=end_date)['Adj Close']
df.index = pd.to_datetime(df.index)
df = df.dropna()

# Crear un gráfico de precios ajustados cerrados usando Plotly
fig_adj_close = go.Figure()
for column in df.columns:
    fig_adj_close.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))
fig_adj_close.update_layout(title="Adjusted Close Price for Sample Stock Tickers",
                            xaxis_title="Time",
                            yaxis_title="Adjusted Close Price",
                            legend_title="Ticker")
st.plotly_chart(fig_adj_close)

st.markdown("""Then, we analize the daily returns of each assets through""")

st.latex(r'''
    r_{t, i}=\frac{p_{t,i}-p_{t-1,i}}{p_{t-1,{i}}},
    ''')

st.markdown("""where $p_{t,i}$ denotes the price of asset $i$ at time $t$.""")

# Crear un gráfico de cambios porcentuales usando Plotly
df_pct_change = df.pct_change().dropna()
fig_pct_change = go.Figure()
for column in df_pct_change.columns:
    fig_pct_change.add_trace(go.Scatter(x=df_pct_change.index, y=df_pct_change[column], mode='lines', name=column))
fig_pct_change.update_layout(title="Percentage Change for Sample Stock Tickers",
                             xaxis_title="Time",
                             yaxis_title="Percentage Change",
                             legend_title="Ticker")
st.plotly_chart(fig_pct_change)

# Calcula los cambios porcentuales para las acciones de Apple
apple_pct_change = df['AAPL'].pct_change().dropna()
meta_pct_change = df['META'].pct_change().dropna()
citigroup_pct_change = df['C'].pct_change().dropna()
disney_pct_change = df['DIS'].pct_change().dropna()

# Crea un histograma utilizando Plotly Express
fig1 = px.histogram(apple_pct_change, nbins=40, title="Apple Percentage Change Histogram")
fig2 = px.histogram(meta_pct_change, nbins=40, title="Meta Percentage Change Histogram")
fig3 = px.histogram(citigroup_pct_change, nbins=40, title="Citigroup Percentage Change Histogram")
fig4 = px.histogram(disney_pct_change, nbins=40, title="Disney Percentage Change Histogram")

# Actualiza el diseño del gráfico
fig1.update_layout(
    xaxis_title="Percentage Change",
    yaxis_title="Frequency",
    bargap=0.2,
    showlegend=False
)

fig1.update_traces(
    hovertemplate='<b>Percentage Change Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
)

fig2.update_layout(
    xaxis_title="Percentage Change",
    yaxis_title="Frequency",
    bargap=0.2,
    showlegend=False
)

fig2.update_traces(
    hovertemplate='<b>Percentage Change Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
)

fig3.update_layout(
    xaxis_title="Percentage Change",
    yaxis_title="Frequency",
    bargap=0.2,
    showlegend=False
)

fig3.update_traces(
    hovertemplate='<b>Percentage Change Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
)

fig4.update_layout(
    xaxis_title="Percentage Change",
    yaxis_title="Frequency",
    bargap=0.2,
    showlegend=False
)

fig4.update_traces(
    hovertemplate='<b>Percentage Change Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
)

st.markdown("""Afterwards, we plot the histogram of each asset's returns to identify the underlying distribution.""")

col1, col2= st.columns(2)

with col1:
   st.plotly_chart(fig1, use_container_width=True)
   st.plotly_chart(fig2, use_container_width=True)

with col2:
   st.plotly_chart(fig3, use_container_width=True)
   st.plotly_chart(fig4, use_container_width=True)


st.header('Portfolio creation')

st.markdown("""To illustrate the use of risk measures, we will construct a portfolio by indicating
             its value at the initial moment and the allocation percentage of each asset.""")

initialPortfolio = st.number_input("Initial Portfolio Value (USD):", min_value=0, max_value=1000000, value=100000, step=1000)

# Obtain percentage change per stock
returns = df.pct_change().dropna()

# Calculate the portfolio returns as the weighted average of the individual asset returns
col1, col2 = st.columns(2)

with col1:
    weight1 = st.number_input("Asset Allocation for Apple:", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    weight2 = st.number_input("Asset Allocation for Meta:", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
with col2:
    weight3 = st.number_input("Asset Allocation for Citigroup:", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    weight4 = st.number_input("Asset Allocation for Disney:", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

# Calcula la suma de los pesos
total_weight = weight1 + weight2 + weight3 + weight4

# Comprueba si la suma es 1
if total_weight == 1:
    st.success("The total weight is 1. Proceeding with calculations.")
    weights = np.array([weight1, weight2, weight3, weight4])
else:
    weights = np.array([weight1, weight2, weight3, weight4])
    st.error(f"The total weight is {total_weight}, which does not sum to 1. Please adjust the weights.")

port_returns = (weights * returns).sum(axis=1) # weighted sum

# Portfolio Percentage Returns
fig5 = px.histogram(port_returns, nbins=40, title="Portfolio Percentage Returns")

fig5.update_layout(
    xaxis_title="Percentage Change",
    yaxis_title="Frequency",
    bargap=0.2,
    showlegend=False
)

fig5.update_traces(
    hovertemplate='<b>Percentage Change Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
)

st.plotly_chart(fig5, use_container_width=True)

st.markdown("""Finally, select a confidence level to measure risk:""")

confidence_level = st.number_input("Confidence Value:", min_value=0.00, max_value=1.00, value=0.99, step=0.01)

################## METODO 1 #######################

st.header('Historical Simulation')

st.markdown("""
            <div style="text-align: justify;">
            In this method, we assume that future returns will follow a similar distribution to historical returns.
            Therefore, no specific distribution for the portfolio returns is provided.
            </div>
            """, unsafe_allow_html=True)

# Calculate P(Return <= VAR) = alpha
var = port_returns.quantile(q=1-confidence_level)
# Calculate CVAR by computing the average returns below the VAR level
cvar = port_returns[port_returns <= var].mean()

fig5.add_vline(x=var, line_dash='dash', line_color='blue',
               annotation_text="VaR: " + str(round(var, 3)),
               annotation_y=0.8,)

fig5.add_vline(x=cvar, line_dash='dash', line_color='firebrick',
               annotation_text="ES: " + str(round(cvar, 3)),
               annotation_y=0.5)

st.plotly_chart(fig5)

st.markdown(f'''
            <div style="text-align: justify;">
            From the sample estimation for VaR, we can deduce that there is a {confidence_level:.0%} confidence 
            level that the portfolio will not lose more than {abs(var):.2%} of its value on any given day. On the other 
            hand, if the portfolio experiences losses exceeding the VaR threshold, the average of those losses will
            be about {abs(cvar):.2%} of the portfolio's total value.
            </div>
            ''', unsafe_allow_html=True)

################## METODO 2 #######################

st.header('Variance-Covariance Method')

st.markdown("""
            <div style="text-align: justify;">
            This method assumes a known parametric model for the daily return on the upcoming day.
            Specifically, it suggests that returns follow a multivariate normal distribution.
            </div>
            """, unsafe_allow_html=True)

# Calculate mean and covariance matrix of returns
mean_returns = returns.mean()
cov_matrix = returns.cov()
port_mean_return = (weights * mean_returns).sum()
port_std_dev = np.sqrt(weights.T @ cov_matrix @ weights)
z_score = norm.ppf(q=1-confidence_level)
var = - (norm.ppf(confidence_level)*port_std_dev - port_mean_return)
cvar = 1 * (port_mean_return - port_std_dev * (norm.pdf(z_score)/(1-confidence_level)))

fig6 = px.histogram(port_returns, nbins=40, title="Portfolio Percentage Returns")

fig6.update_layout(
    xaxis_title="Percentage Change",
    yaxis_title="Frequency",
    bargap=0.2,
    showlegend=False
)

fig6.update_traces(
    hovertemplate='<b>Percentage Change Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
)


fig6.add_vline(x=var, line_dash='dash', line_color='blue',
               annotation_text="VaR: " + str(round(var, 3)),
               annotation_y=0.8,)

fig6.add_vline(x=cvar, line_dash='dash', line_color='firebrick',
               annotation_text="ES: " + str(round(cvar, 3)),
               annotation_y=0.5)

st.plotly_chart(fig6)

st.markdown(f'''
            <div style="text-align: justify;">
            From the calculation for VaR assuming a normal model, we can deduce that there is a {confidence_level:.0%} confidence 
            level that the portfolio will not lose more than {abs(var):.2%} of its value on any given day. On the other 
            hand, if the portfolio experiences losses exceeding the VaR threshold, the average of those losses will
            be about {abs(cvar):.2%} of the portfolio's total value.
            </div>
            ''', unsafe_allow_html=True)

################## METODO 3 #######################

st.header('Monte Carlo Method')

n_simulation = st.number_input("Number of simulations:", min_value=50, max_value=1000, value=100, step=50)


# n_simulation = 50  number of simulations
T = 252 # number of trading days in a year
weights = np.full((4), 0.25)
meanM = np.full(shape=(T, len(weights)), fill_value=mean_returns).T

# Simulation process
sim_pct_change = np.full(shape=(T, n_simulation), fill_value=0.0)
for m in range(n_simulation):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(cov_matrix)
    daily_pct_change = meanM + np.inner(L, Z)
    sim_pct_change[:, m] = np.inner(weights, daily_pct_change.T)

# Create Plotly figure
fig = go.Figure()

# Adding portfolio change for each simulation as a line
for i in range(n_simulation):
    fig.add_trace(go.Scatter(y=sim_pct_change[:, i], mode='lines', line=dict(width=1), showlegend=False))

# Update layout
fig.update_layout(title='MC Simulation of a Stock Portfolio Percentage Change',
                  xaxis_title='Days',
                  yaxis_title='Portfolio Percentage Change',
                  legend_title="Statistics")

fig.add_hline(y = np.percentile(sim_pct_change,5), line_dash='dash', line_color='white',
               annotation_text="5th Percentile: " + str(round(np.percentile(sim_pct_change,5), 3)),
               annotation_x=0.1)

fig.add_hline(y = np.percentile(sim_pct_change,95), line_dash='dash', line_color='white',
               annotation_text="95th Percentile: " + str(round(np.percentile(sim_pct_change,95), 3)),
               annotation_x=0.1)

fig.add_hline(y = np.mean(sim_pct_change), line_dash='dash', line_color='white',
               annotation_text="Mean: " + str(round(np.mean(sim_pct_change), 3)),
               annotation_x=0.1)

# Use Streamlit to display the plot
st.plotly_chart(fig)

confidence_level = 0.95
# Convert sim_returns to dataframe
port_pct_change = pd.Series(sim_pct_change[-1,:])
# Calculate the VAR and CVAR at 95% confidence level of the portfolio percentage change
mcVAR = port_pct_change.quantile(1 - confidence_level)
mcCVAR = port_pct_change[port_pct_change <= mcVAR].mean()

fig8 = px.histogram(port_pct_change, nbins=40, title="Portfolio Percentage Returns")

fig8.update_layout(
    xaxis_title="Percentage Change",
    yaxis_title="Frequency",
    bargap=0.2,
    showlegend=False
)

fig8.update_traces(
    hovertemplate='<b>Percentage Change Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>'
)


fig8.add_vline(x=mcVAR, line_dash='dash', line_color='blue',
               annotation_text="VaR: " + str(round(mcVAR, 3)),
               annotation_y=0.8)

fig8.add_vline(x=mcCVAR, line_dash='dash', line_color='firebrick',
               annotation_text="ES: " + str(round(mcCVAR, 3)),
               annotation_y=0.5)

st.plotly_chart(fig8)


# Create data frame fill with 0
portfolio_returns = np.full(shape=(T, n_simulation), fill_value=0.0)

# Convert the percentage change to actual portfolio value
for m in range(n_simulation):
    portfolio_returns[:,m] = np.cumprod(sim_pct_change[:,m]+1)*initialPortfolio

# Select the last simulated trading day records
last_portfolio_returns = portfolio_returns[-1, :]
# Calculate the VAR and CVAR at 95% confidence level of the portfolio returns
mc_var_returns = np.percentile(last_portfolio_returns, 5)
mc_cvar_returns = last_portfolio_returns[last_portfolio_returns <= mc_var_returns].mean()

# Create Plotly figure
fig = go.Figure()

# Adding portfolio change for each simulation as a line
for i in range(n_simulation):
    fig.add_trace(go.Scatter(y=portfolio_returns[:, i], mode='lines', line=dict(width=1), showlegend=False))

# Update layout
fig.update_layout(title='MC Simulation of the Stock Portfolio',
                  xaxis_title='Days',
                  yaxis_title='Portfolio Value ($)',
                  legend_title="Statistics")

fig.add_hline(y = mc_var_returns, line_dash='dash', line_color='white',
               annotation_text="VaR: " + str(round(mc_var_returns, 3)),
               annotation_x=0.1)

fig.add_hline(y = mc_cvar_returns, line_dash='dash', line_color='white',
               annotation_text="ES: " + str(round(mc_cvar_returns, 3)),
               annotation_x=0.1)


# Use Streamlit to display the plot
st.plotly_chart(fig)
