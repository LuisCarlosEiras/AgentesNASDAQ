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
            return None, None
        
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
        
        # Retornar df_prophet e forecast para análise
        return df_prophet, forecast
    
    except Exception as e:
        st.error(f"Erro ao gerar previsão para {ticker}: {str(e)}")
        return None, None

def dsa_analyze_prophet_forecast(df_prophet, forecast, ticker):
    try:
        # Verificar se os dados são válidos
        if df_prophet is None or forecast is None:
            st.warning(f"Não foi possível analisar a previsão para {ticker}: dados ausentes.")
            return
        
        # Depuração: Confirmar que a função está sendo chamada
        st.write(f"Executando análise da previsão para {ticker}...")
        
        # Último preço histórico
        last_price = df_prophet['y'].iloc[-1]
        
        # Preço previsto no final do período (90 dias)
        final_predicted_price = forecast['yhat'].iloc[-1]
        
        # Intervalos de confiança no final do período
        final_upper = forecast['yhat_upper'].iloc[-1]
        final_lower = forecast['yhat_lower'].iloc[-1]
        
        # Calcular variação percentual
        percent_change = ((final_predicted_price - last_price) / last_price) * 100
        
        # Determinar a tendência
        if percent_change > 2:
            trend = "alta"
        elif percent_change < -2:
            trend = "baixa"
        else:
            trend = "estabilidade"
        
        # Calcular volatilidade (média da largura dos intervalos de confiança)
        confidence_width = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
        volatility = "alta" if confidence_width / last_price > 0.1 else "moderada" if confidence_width / last_price > 0.05 else "baixa"
        
        # Resumo da análise
        analysis = f"""
        ### Análise da Previsão para {ticker} (Próximos 3 Meses)
        
        - **Tendência**: A previsão indica uma tendência de **{trend}**.
        - **Variação Esperada**: O preço deve variar em aproximadamente **{percent_change:.2f}%**, de ${last_price:.2f} para ${final_predicted_price:.2f}.
        - **Intervalo de Confiança**: No final do período, o preço pode estar entre **${final_lower:.2f}** e **${final_upper:.2f}**.
        - **Volatilidade**: A incerteza da previsão é **{volatility}**, com base nos intervalos de confiança.
        - **Observação**: A previsão é baseada em dados históricos e não considera eventos externos inesperados (ex.: mudanças regulatórias ou "tarifaço"). Considere isso ao tomar decisões.
        """
        
        # Exibir a análise no Streamlit
        st.markdown(analysis)
    
    except Exception as e:
        st.error(f"Erro ao analisar previsão para {ticker}: {str(e)}")

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
Ações da Nasdaq analisadas em tempo real por Agentes de IA usando DeepSeek através do Groq e infraestrutura Streamlit. 
""")

if st.sidebar.button("Suporte"):
    st.sidebar.write("No caso de dúvidas envie e-mail para: luiscarloseiras@gmail.com")

st.title(":satellite_antenna: Um Agente IA para acompanhar o tarifaço na NASDAQ")
st.header("Day Trade Analytics em Tempo Real")

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
            df_prophet, forecast = dsa_plot_prophet_forecast(hist, ticker)
            dsa_analyze_prophet_forecast(df_prophet, forecast, ticker)
    else:
        st.error("Ticker inválido. Insira um símbolo de ação válido.")
