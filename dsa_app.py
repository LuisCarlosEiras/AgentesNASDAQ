# Imports
import re
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from prophet import Prophet
import pandas as pd

# Configuração da chave API (Token)
groq_api_key = st.secrets["GROQ_API_KEY"]

########## Analytics ##########

@st.cache_data
def dsa_extrai_dados(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist.reset_index(inplace=True)
    return hist

def dsa_plot_stock_price(hist, ticker):
    fig = px.line(hist, x="Date", y="Close", title=f"{ticker} Preços das Ações (Últimos 6 Meses)", markers=True)
    st.plotly_chart(fig)

def dsa_plot_candlestick(hist, ticker):
    fig = go.Figure(
        data=[go.Candlestick(x=hist['Date'],
                             open=hist['Open'],
                             high=hist['High'],
                             low=hist['Low'],
                             close=hist['Close'])]
    )
    fig.update_layout(title=f"{ticker} Candlestick Chart (Últimos 6 Meses)")
    st.plotly_chart(fig)

def dsa_plot_media_movel(hist, ticker):
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
    fig = px.line(hist, 
                  x='Date', 
                  y=['Close', 'SMA_20', 'EMA_20'],
                  title=f"{ticker} Médias Móveis (Últimos 6 Meses)",
                  labels={'value': 'Price (USD)', 'Date': 'Date'})
    st.plotly_chart(fig)

def dsa_plot_volume(hist, ticker):
    fig = px.bar(hist, 
                 x='Date', 
                 y='Volume', 
                 title=f"{ticker} Trading Volume (Últimos 6 Meses)")
    st.plotly_chart(fig)

def dsa_plot_prophet_forecast(hist, ticker):
    try:
        # Preparar os dados para o Prophet
        df_prophet = hist[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).copy()
        
        # Garantir que 'ds' é datetime e remover fuso horário
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
        
        # Verificar e remover valores ausentes
        if df_prophet['y'].isnull().any() or df_prophet['ds'].isnull().any():
            st.warning(f"Dados incompletos para {ticker}. Removendo valores ausentes.")
            df_prophet = df_prophet.dropna()
        
        # Verificar se há dados suficientes
        if len(df_prophet) < 2:
            st.error(f"Não há dados suficientes para previsão de {ticker}.")
            return
        
        # Inicializar e ajustar o modelo Prophet
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        # Criar um DataFrame para os próximos 90 dias
        future = model.make_future_dataframe(periods=90)
        
        # Gerar previsões
        forecast = model.predict(future)
        
        # Criar o gráfico com Plotly
        fig = go.Figure()
        
        # Adicionar os dados históricos
        fig.add_trace(go.Scatter(
            x=df_prophet['ds'],
            y=df_prophet['y'],
            mode='lines',
            name='Dados Históricos'
        ))
        
        # Adicionar a previsão
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Previsão',
            line=dict(color='orange')
        ))
        
        # Adicionar intervalos de confiança
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Limite Superior',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Limite Inferior',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.2)',
            showlegend=False
        ))
        
        # Atualizar o layout do gráfico
        fig.update_layout(
            title=f"{ticker} Previsão de Preços para os Próximos 3 Meses",
            xaxis_title="Data",
            yaxis_title="Preço (USD)",
            template="plotly_white"
        )
        
        # Exibir o gráfico no Streamlit
        st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f"Erro ao gerar previsão para {ticker}: {str(e)}")

########## Agentes de IA ##########

dsa_agente_web_search = Agent(name="DSA Agente Web Search",
                              role="Fazer busca na web",
                              model=Groq(id="deepseek-r1-distill-llama-70b"),
                              tools=[DuckDuckGo()],
                              instructions=["Sempre inclua as fontes"],
                              show_tool_calls=True, markdown=True)

dsa_agente_financeiro = Agent(name="DSA Agente Financeiro",
                              model=Groq(id="deepseek-r1-distill-llama-70b"),
                              tools=[YFinanceTools(stock_price=True,
                                                   analyst_recommendations=True,
                                                   stock_fundamentals=True,
                                                   company_news=True)],
                              instructions=["Use tabelas para mostrar os dados"],
                              show_tool_calls=True, markdown=True)

multi_ai_agent = Agent(team=[dsa_agente_web_search, dsa_agente_financeiro],
                       model=Groq(id="llama-3.3-70b-versatile"),
                       instructions=["Sempre inclua as fontes", "Use tabelas para mostrar os dados"],
                       show_tool_calls=True, markdown=True)

########## App Web ##########

st.set_page_config(page_title="Um Agente IA para acompanhar o tarifaço na NASDAQ", page_icon=":954:", layout="wide")

st.sidebar.title("Instruções")
st.sidebar.markdown("""
### Como Utilizar a App:

- Insira o símbolo do ticker da ação desejada no campo central.
- Clique em **Analisar** para gerar a análise e gráficos em tempo real.

### Exemplos de tickers válidos:
- MSFT (Microsoft)
- DJT (Trump)
- TSLA (Tesla)
- AMZN (Amazon)
- GOOG (Alphabet)
- NDAQ (NASDAQ)

Mais tickers podem ser encontrados aqui: https://stockanalysis.com/list/nasdaq-stocks/

### Finalidade da App:
Ações da Nasdaq analisadas em tempo real por Agentes de IA usando DeepSeek através do Groq, previsão pelo Prophet e infraestrutura Streamlit.""")

if st.sidebar.button("Suporte"):
    st.sidebar.write("No caso de dúvidas envie e-mail para: luiscarloseiras@gmail.com")

st.title(":satellite_antenna: Um Agente IA para acompanhar o tarifaço na NASDAQ")
st.header("Day Trade Analytics em Tempo Real e Previsão para os Próximos 3 Meses")

ticker = st.text_input("Digite o Código (símbolo do ticker):").upper()

if st.button("Analisar"):
    if ticker:
        with st.spinner("Buscando os Dados em Tempo Real. Aguarde..."):
            hist = dsa_extrai_dados(ticker)
            st.subheader("Análise Gerada Por IA")
            ai_response = multi_ai_agent.run(f"Resumir a recomendação do analista, compartilhar as últimas notícias para {ticker} incluindo as consequências do tarifaço do Trump e prever as variações da ação nos próximos 3 meses")
            clean_response = re.sub(r"(Running:[\s\S]*?\n\n)|(^transfer_task_to_finance_ai_agent.*\n?)","", ai_response.content, flags=re.MULTILINE).strip()
            st.markdown(clean_response)
            
            st.subheader("Visualização dos Dados")
            dsa_plot_stock_price(hist, ticker)
            dsa_plot_candlestick(hist, ticker)
            dsa_plot_media_movel(hist, ticker)
            dsa_plot_volume(hist, ticker)
            dsa_plot_prophet_forecast(hist, ticker)
    else:
        st.error("Ticker inválido. Insira um símbolo de ação válido.")
