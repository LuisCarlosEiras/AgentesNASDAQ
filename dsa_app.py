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
from phi.llm.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import os

# --------------------------------------------------------------------------
# !! IMPORTANTE: st.set_page_config DEVE SER O PRIMEIRO COMANDO STREAMLIT !!
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Agente IA para NASDAQ",
    page_icon="📊",
    layout="wide"
)
# --------------------------------------------------------------------------

# --- Configuração da Chave API (Token) ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (AttributeError, KeyError):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Chave API GROQ não configurada. Defina GROQ_API_KEY nos secrets do Streamlit ou como variável de ambiente.")
        st.stop()

# --- Modelo Groq a ser Usado ---
MODELO_GROQ_SELECIONADO = "llama-3.3-70b-versatile"  # Usado no orquestrador
st.sidebar.info(f"Usando modelo Groq principal: `{MODELO_GROQ_SELECIONADO}`")

# --- Subclasse Personalizada do Agent para Evitar OpenAI ---
class CustomAgent(Agent):
    def update_model(self):
        if self.llm is None:
            raise ValueError("Nenhum LLM configurado para o agente.")
        pass

########## Analytics ##########

@st.cache_data
def dsa_extrai_dados(ticker, period="1y"):
    """Extrai dados históricos de uma ação (cacheado)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            st.toast(f"Nenhum dado histórico encontrado para {ticker} no período {period}.", icon="⚠️")
            return None
        hist.reset_index(inplace=True)
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
        return hist
    except Exception as e:
        st.error(f"Erro ao buscar dados do yfinance para {ticker}: {e}")
        return None

@st.cache_data
def dsa_gera_previsao(hist_df, periods=90):
    """Gera previsão futura usando Prophet (cacheado)."""
    if hist_df is None or hist_df.empty:
        return None, None

    df_prophet = hist_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    if df_prophet.empty or len(df_prophet) < 2:
        st.warning("Dados insuficientes para gerar previsão com Prophet.")
        return None, None

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True if len(df_prophet) > 365 else False,
        changepoint_prior_scale=0.05
    )

    try:
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        st.error(f"Erro ao treinar ou prever com Prophet: {e}")
        return None, None

# --- Funções de Plotagem ---
def dsa_plot_stock_price(hist, forecast, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines+markers', name='Histórico'))
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão (yhat)', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Incerteza Superior', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Incerteza Inferior', line=dict(width=0), fillcolor='rgba(255, 165, 0, 0.2)', fill='tonexty', showlegend=False))
    fig.update_layout(title=f"{ticker} Preços das Ações e Previsão (3 Meses)", yaxis_title='Preço (USD)')
    st.plotly_chart(fig, use_container_width=True)

def dsa_plot_candlestick(hist, forecast, ticker):
    fig = go.Figure(data=[go.Candlestick(x=hist['Date'], open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Histórico')])
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão Fechamento (yhat)', line=dict(color='orange', dash='dash')))
    fig.update_layout(title=f"{ticker} Candlestick e Previsão de Fechamento (3 Meses)", yaxis_title='Preço (USD)', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def dsa_plot_media_movel(hist, forecast, ticker):
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', name='Fechamento'))
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['SMA_20'], mode='lines', name='SMA 20 Dias'))
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['EMA_20'], mode='lines', name='EMA 20 Dias'))
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão Fechamento (yhat)', line=dict(color='orange', dash='dash')))
    fig.update_layout(title=f"{ticker} Médias Móveis e Previsão de Fechamento (3 Meses)", yaxis_title='Preço (USD)')
    st.plotly_chart(fig, use_container_width=True)

def dsa_plot_volume(hist, ticker):
    fig = px.bar(hist, x='Date', y='Volume', title=f"{ticker} Volume de Negociação (Último Ano)")
    fig.update_layout(yaxis_title='Volume')
    st.plotly_chart(fig, use_container_width=True)

# --- Geração de Comentário da IA ---
def dsa_gera_comentario_previsao(ticker, hist_df, forecast_df, agent):
    if forecast_df is None or forecast_df.empty or hist_df is None or hist_df.empty:
        return "Dados insuficientes ou erro na previsão para gerar comentário."

    try:
        last_date_hist = hist_df['Date'].iloc[-1].strftime('%Y-%m-%d')
        last_price_hist = hist_df['Close'].iloc[-1]
        first_future_index = len(hist_df)
        if first_future_index >= len(forecast_df):
            return "Erro ao alinhar previsão com histórico para gerar comentário."

        forecast_start_date = forecast_df['ds'].iloc[first_future_index].strftime('%Y-%m-%d')
        forecast_end_date = forecast_df['ds'].iloc[-1].strftime('%Y-%m-%d')
        forecast_end_price = forecast_df['yhat'].iloc[-1]
        forecast_max_price = forecast_df['yhat_upper'].iloc[-1]
        forecast_min_price = forecast_df['yhat_lower'].iloc[-1]

        prompt = f"""
        Analise a seguinte previsão de preço para a ação {ticker} para os próximos 3 meses, gerada pelo modelo Prophet.

        Dados Históricos Relevantes:
        - Última data histórica: {last_date_hist}
        - Último preço de fechamento histórico: ${last_price_hist:.2f}

        Previsão para os Próximos 3 Meses (de {forecast_start_date} até {forecast_end_date}):
        - Preço previsto para {forecast_end_date}: ${forecast_end_price:.2f}
        - Faixa de Incerteza para {forecast_end_date}: entre ${forecast_min_price:.2f} e ${forecast_max_price:.2f}

        Com base *apenas* nesses dados de previsão e no último preço histórico:
        1. Descreva a tendência geral prevista (alta, baixa, estável).
        2. Comente brevemente sobre a confiança da previsão, mencionando a faixa de incerteza.
        3. Forneça uma breve conclusão sobre o que a previsão sugere para os próximos 3 meses.

        Seja conciso e direto ao ponto. Não use informações externas ou de ferramentas, baseie-se somente nos dados fornecidos aqui.
        """

        ai_comment_response = agent.run(prompt, stream=False)
        clean_comment = str(ai_comment_response).strip() if not isinstance(ai_comment_response, str) else ai_comment_response.strip()
        clean_comment = re.sub(r"(Running|Calling|Using) tool.*?\n", "", clean_comment, flags=re.IGNORECASE | re.DOTALL)
        clean_comment = re.sub(r"\[.*?\]\(.*?\)", "", clean_comment)
        clean_comment = clean_comment.replace("```json", "").replace("```", "").strip()

        if not clean_comment or len(clean_comment) < 30:
            st.toast("Resposta da IA sobre previsão foi curta ou vazia.", icon="ℹ️")
            return "Não foi possível gerar um comentário detalhado da IA sobre a previsão neste momento."

        return clean_comment

    except Exception as e:
        st.warning(f"Erro ao gerar comentário da IA: {e}")
        return "Ocorreu um erro ao gerar o comentário da IA sobre a previsão."

########## Agentes de IA ##########

dsa_agente_web_search = None
dsa_agente_financeiro = None
multi_ai_agent = None
agents_initialized = False

try:
    dsa_agente_web_search = CustomAgent(
        name="DSA Agente Web Search",
        role="Fazer busca na web",
        llm=Groq(model="deepseek-r1-distill-llama-70b", api_key=groq_api_key),  # Alterado para 'model' em vez de 'id'
        tools=[DuckDuckGo()],
        instructions=["Sempre inclua as fontes"],
        show_tool_calls=True,
        markdown=True
    )

    dsa_agente_financeiro = CustomAgent(
        name="DSA Agente Financeiro",
        llm=Groq(model="deepseek-r1-distill-llama-70b", api_key=groq_api_key),  # Alterado para 'model' em vez de 'id'
        tools=[YFinanceTools(stock_price=True,
                             analyst_recommendations=True,
                             stock_fundamentals=True,
                             company_news=True)],
        instructions=["Use tabelas para mostrar os dados"],
        show_tool_calls=True,
        markdown=True
    )

    multi_ai_agent = CustomAgent(
        team=[dsa_agente_web_search, dsa_agente_financeiro],
        llm=Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key),  # Alterado para 'model' em vez de 'id'
        instructions=["Sempre inclua as fontes", "Use tabelas para mostrar os dados"],
        show_tool_calls=True,
        markdown=True
    )
    agents_initialized = True

except Exception as e:
    st.error(f"Erro ao inicializar os Agentes de IA: {e}")
    agents_initialized = False

########## App Web ##########

st.sidebar.title("Instruções")
st.sidebar.markdown(f"""
### Como Utilizar:
1. Insira o símbolo do ticker da ação (ex: `MSFT`, `AAPL`) no campo abaixo.
2. Clique em **Analisar**.
3. Aguarde enquanto os dados são buscados, a previsão é gerada e a IA analisa as informações.

**Modelo de IA Principal:** `{MODELO_GROQ_SELECIONADO}`
""")
st.sidebar.markdown("### Sobre a Previsão:")
st.sidebar.markdown("""
A previsão de 3 meses é gerada usando o modelo estatístico **Prophet**. Lembre-se:
- Previsões são estimativas baseadas no histórico e **não garantias**.
- O mercado financeiro é volátil e influenciado por muitos fatores.
- Use esta análise como **uma ferramenta de apoio**, não como única base para decisões de investimento.
""")

if st.sidebar.button("Suporte"):
    st.sidebar.write("Contato: luiscarloseiras@gmail.com")

st.title("📈 Agente de IA para Análise de Ações da NASDAQ")
st.header("Análise Histórica, Previsão (3 Meses) e Insights de IA")

ticker = st.text_input("Digite o Código da Ação (ticker):", placeholder="Ex: AAPL, MSFT, NVDA", disabled=not agents_initialized).upper()

if st.button("Analisar", key="analyze_button", disabled=not agents_initialized):
    if ticker and agents_initialized:
        st.markdown("---")
        progress_bar = st.progress(0, text="Iniciando análise...")
        analysis_successful = False
        ai_forecast_comment = "Comentário da previsão não gerado."
        clean_analysis_response = "Análise da IA não gerada."

        analysis_placeholder = st.empty()
        charts_placeholder = st.container()

        try:
            progress_bar.progress(10, text=f"Buscando dados históricos para {ticker}...")
            hist_data = dsa_extrai_dados(ticker)
            if hist_data is None:
                st.error(f"Falha ao obter dados históricos para {ticker}. Análise interrompida.")
                progress_bar.empty()
                st.stop()

            progress_bar.progress(30, text=f"Gerando previsão de 3 meses para {ticker}...")
            with st.spinner("Treinando modelo Prophet e gerando previsão..."):
                model_prophet, forecast_data = dsa_gera_previsao(hist_data)

            if forecast_data is None:
                st.toast(f"Não foi possível gerar a previsão para {ticker}.", icon="⚠️")
                ai_forecast_comment = "Previsão não disponível."
            else:
                progress_bar.progress(50, text="Gerando comentário da IA sobre a previsão...")
                with st.spinner("IA analisando a previsão..."):
                    ai_forecast_comment = dsa_gera_comentario_previsao(ticker, hist_data, forecast_data, multi_ai_agent)

            progress_bar.progress(70, text=f"Buscando notícias e recomendações para {ticker}...")
            with st.spinner(f"Consultando IA para recomendações e notícias de {ticker}..."):
                prompt_analise = f"Forneça um resumo das recomendações de analistas e as últimas notícias para a ação {ticker}. Use o `DSA Agente Financeiro` para obter os dados."
                ai_analysis_response = multi_ai_agent.run(prompt_analise, stream=False)
                clean_analysis_response = str(ai_analysis_response).strip() if not isinstance(ai_analysis_response, str) else ai_analysis_response.strip()
                clean_analysis_response = re.sub(r"(Running|Using|Calling tool|Delegate to).*?\n", "", clean_analysis_response, flags=re.IGNORECASE | re.DOTALL).strip()
                clean_analysis_response = re.sub(r"DSA Agente Financeiro", "Agente Financeiro", clean_analysis_response)
                clean_analysis_response = re.sub(r"`", "", clean_analysis_response)

            progress_bar.progress(90, text="Renderizando resultados...")

            analysis_placeholder.empty()
            charts_placeholder.empty()

            with analysis_placeholder.container():
                st.subheader(f"Análise por IA para {ticker}")
                if clean_analysis_response and clean_analysis_response != "Análise da IA não gerada.":
                    st.markdown(clean_analysis_response)
                else:
                    st.warning("Não foi possível obter a análise de notícias/recomendações da IA.")
                st.markdown("---")

            with charts_placeholder.container():
                st.subheader("Visualização dos Dados e Previsão")
                st.markdown("##### Preço de Fechamento Histórico e Previsão")
                dsa_plot_stock_price(hist_data, forecast_data, ticker)
                if forecast_data is not None and not forecast_data.empty:
                    st.markdown(f"**Comentário da IA sobre a Previsão:**\n {ai_forecast_comment}")
                st.markdown("---")

                st.markdown("##### Candlestick Histórico e Previsão de Fechamento")
                dsa_plot_candlestick(hist_data, forecast_data, ticker)
                if forecast_data is not None and not forecast_data.empty:
                    st.markdown(f"**Comentário da IA sobre a Previsão:**\n {ai_forecast_comment}")
                st.markdown("---")

                st.markdown("##### Médias Móveis Históricas e Previsão de Fechamento")
                dsa_plot_media_movel(hist_data, forecast_data, ticker)
                if forecast_data is not None and not forecast_data.empty:
                    st.markdown(f"**Comentário da IA sobre a Previsão:**\n {ai_forecast_comment}")
                st.markdown("---")

                st.markdown("##### Volume de Negociação Histórico")
                dsa_plot_volume(hist_data, ticker)
                st.markdown("**Nota:** A previsão de volume não está incluída nesta análise.")
                st.markdown("---")

            progress_bar.progress(100, text="Análise concluída!")
            analysis_successful = True

        except Exception as e:
            st.error(f"Ocorreu um erro inesperado durante a análise principal: {e}")
            st.exception(e)
        finally:
            progress_bar.empty()
            if analysis_successful:
                st.success(f"Análise para {ticker} concluída!")
            else:
                st.error(f"Análise para {ticker} encontrou problemas.")

    elif not ticker:
        st.warning("Por favor, insira um código de ação (ticker).")
    elif not agents_initialized:
        st.error("Os agentes de IA não foram inicializados corretamente. Verifique os erros acima e a configuração.")
