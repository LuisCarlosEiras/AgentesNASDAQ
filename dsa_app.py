# Módulo Especial de Consultoria na Área de Dados com Agentes de IA
# Projeto Prático Para Consultoria na Área de Dados com Agentes de IA
# Deploy de App Para Day Trade Analytics em Tempo Real com Agentes de IA, Groq, DeepSeek e AWS Para Monetização

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
import pandas as pd
import numpy as np
from datetime import timedelta

# Configuração da chave API (Token)
groq_api_key = st.secrets["GROQ_API_KEY"]

########## Analytics ##########

@st.cache_data
def dsa_extrai_dados(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist.reset_index(inplace=True)
    return hist

# Função para gerar previsões simples baseadas em EMA
def dsa_gera_previsao(hist, periods=90):
    last_date = hist['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
    last_ema = hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    forecast = [last_ema * (1 + np.random.normal(0, 0.01)) for _ in range(periods)]  # Simulação simples
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
    return forecast_df

def dsa_plot_stock_price(hist, ticker):
    forecast_df = dsa_gera_previsao(hist)
    fig = px.line(hist, x="Date", y="Close", title=f"{ticker} Preços das Ações (Últimos 6 Meses + Previsão)", markers=True)
    fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Previsão (3 meses)', line=dict(dash='dash'))
    st.plotly_chart(fig)
    st.markdown(f"**Comentário sobre a previsão:** A previsão para {ticker} nos próximos 3 meses é baseada na tendência da EMA recente com uma leve variação simulada. Observe que a volatilidade pode aumentar dependendo de eventos de mercado.")

def dsa_plot_candlestick(hist, ticker):
    forecast_df = dsa_gera_previsao(hist)
    fig = go.Figure(
        data=[go.Candlestick(x=hist['Date'], open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']),
              go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Previsão (3 meses)', line=dict(dash='dash'))]
    )
    fig.update_layout(title=f"{ticker} Candlestick Chart (Últimos 6 Meses + Previsão)")
    st.plotly_chart(fig)
    st.markdown(f"**Comentário sobre a previsão:** A linha tracejada mostra uma projeção conservadora para {ticker}. Fatores como volume e notícias podem influenciar os preços além dessa estimativa.")

def dsa_plot_media_movel(hist, ticker):
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
    forecast_df = dsa_gera_previsao(hist)
    fig = px.line(hist, x='Date', y=['Close', 'SMA_20', 'EMA_20'], title=f"{ticker} Médias Móveis (Últimos 6 Meses + Previsão)", labels={'value': 'Price (USD)', 'Date': 'Date'})
    fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Previsão (3 meses)', line=dict(dash='dash'))
    st.plotly_chart(fig)
    st.markdown(f"**Comentário sobre a previsão:** A previsão para {ticker} segue a EMA de 20 dias, sugerindo continuidade da tendência atual, mas ajustes podem ser necessários com novos dados.")

def dsa_plot_volume(hist, ticker):
    forecast_df = dsa_gera_previsao(hist)  # Não usamos volume na previsão, mas mantemos consistência
    fig = px.bar(hist, x='Date', y='Volume', title=f"{ticker} Trading Volume (Últimos 6 Meses)")
    st.plotly_chart(fig)
    st.markdown(f"**Comentário sobre a previsão:** O volume não é previsto diretamente, mas um aumento no volume futuro pode indicar maior interesse em {ticker}, impactando a previsão de preço.")

########## Agentes de IA ##########

dsa_agente_web_search = Agent(name="DSA Agente Web Search",
                              role="Fazer busca na web",
                              model=Groq(id="deepseek-r1-distill-llama-70b"),
                              tools=[DuckDuckGo()],
                              instructions=["Sempre inclua as fontes"],
                              show_tool_calls=True, markdown=True)

dsa_agente_financeiro = Agent(name="DSA Agente Financeiro",
                              model=Groq(id="deepseek-r1-distill-llama-70b"),
                              tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
                              instructions=["Use tabelas para mostrar os dados"],
                              show_tool_calls=True, markdown=True)

multi_ai_agent = Agent(team=[dsa_agente_web_search, dsa_agente_financeiro],
                       model=Groq(id="llama-3.3-70b-versatile"),
                       instructions=["Sempre inclua as fontes", "Use tabelas para mostrar os dados"],
                       show_tool_calls=True, markdown=True)

########## App Web ##########

st.set_page_config(page_title="Agente IA para NASDAQ", page_icon=":954:", layout="wide")

st.sidebar.title("Instruções")
st.sidebar.markdown("""
### Como Utilizar a App:
- Insira o símbolo do ticker da ação desejada no campo central.
- Clique no botão **Analisar** para obter a análise em tempo real com visualizações e insights gerados por IA.

### Exemplos de tickers válidos:
- MSFT (Microsoft)
- TSLA (Tesla)
- AMZN (Amazon)
- GOOG (Alphabet)

Mais tickers podem ser encontrados aqui: https://stockanalysis.com/list/nasdaq-stocks/

### Finalidade da App:
Este aplicativo realiza análises avançadas de preços de ações da Nasdaq em tempo real utilizando Agentes de IA com modelo DeepSeek através do Groq e infraestrutura AWS para apoio a estratégias de Day Trade para monetização.
""")

if st.sidebar.button("Suporte"):
    st.sidebar.write("No caso de dúvidas envie e-mail para: luiscarloseiras@gmail.com")

st.title(":954: Um Agente de IA para a NASDAQ")
st.header("Day Trade Analytics em Tempo Real")

ticker = st.text_input("Digite o Código (símbolo do ticker):").upper()

if st.button("Analisar"):
    if ticker:
        with st.spinner("Buscando os Dados em Tempo Real. Aguarde..."):
            hist = dsa_extrai_dados(ticker)
            st.subheader("Análise Gerada Por IA")
            ai_response = multi_ai_agent.run(f"Resumir a recomendação do analista e compartilhar as últimas notícias para {ticker}")
            clean_response = re.sub(r"(Running:[\s\S]*?\n\n)|(^transfer_task_to_finance_ai_agent.*\n?)","", ai_response.content, flags=re.MULTILINE).strip()
            st.markdown(clean_response)
            st.subheader("Visualização dos Dados")
            dsa_plot_stock_price(hist, ticker)
            dsa_plot_candlestick(hist, ticker)
            dsa_plot_media_movel(hist, ticker)
            dsa_plot_volume(hist, ticker)
    else:
        st.error("Ticker inválido. Insira um símbolo de ação válido.")
