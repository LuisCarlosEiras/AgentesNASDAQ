import re
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

groq_api_key = st.secrets["GROQ_API_KEY"]

@st.cache_data
def dsa_extrai_dados(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist.reset_index(inplace=True)
    return hist

def dsa_prever_precos(hist):
    model = ARIMA(hist['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=90)
    return forecast

def dsa_plot_stock_price(hist, ticker):
    forecast = dsa_prever_precos(hist)
    forecast_dates = pd.date_range(hist['Date'].iloc[-1], periods=91, freq='D')[1:]
    
    fig = px.line(hist, x="Date", y="Close", title=f"{ticker} Preços das Ações e Previsão (Últimos 6 Meses + 3 Meses)", markers=True)
    fig.add_scatter(x=forecast_dates, y=forecast, mode='lines', name='Previsão')
    
    st.plotly_chart(fig)

def dsa_plot_candlestick(hist, ticker):
    fig = go.Figure(data=[go.Candlestick(x=hist['Date'], open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
    fig.update_layout(title=f"{ticker} Candlestick Chart (Últimos 6 Meses)")
    st.plotly_chart(fig)

def dsa_plot_media_movel(hist, ticker):
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
    fig = px.line(hist, x='Date', y=['Close', 'SMA_20', 'EMA_20'], title=f"{ticker} Médias Móveis (Últimos 6 Meses)")
    st.plotly_chart(fig)

def dsa_plot_volume(hist, ticker):
    fig = px.bar(hist, x='Date', y='Volume', title=f"{ticker} Trading Volume (Últimos 6 Meses)")
    st.plotly_chart(fig)

st.set_page_config(page_title="Agente IA para NASDAQ", page_icon=":954:", layout="wide")
st.sidebar.title("Instruções")
st.title(":954: Um Agente de IA para a NASDAQ")
st.header("Day Trade Analytics em Tempo Real")

ticker = st.text_input("Digite o Código (símbolo do ticker):").upper()

if st.button("Analisar"):
    if ticker:
        with st.spinner("Buscando os Dados em Tempo Real. Aguarde..."):
            hist = dsa_extrai_dados(ticker)
            st.subheader("Análise Gerada Por IA")
            st.subheader("Visualização dos Dados")
            dsa_plot_stock_price(hist, ticker)
            dsa_plot_candlestick(hist, ticker)
            dsa_plot_media_movel(hist, ticker)
            dsa_plot_volume(hist, ticker)
    else:
        st.error("Ticker inválido. Insira um símbolo de ação válido.")
