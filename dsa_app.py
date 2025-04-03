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
from phi.llm.groq import Groq # Importação mais específica para LLM
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
# from dotenv import load_dotenv # Descomente se usar .env localmente
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import os # Para API key se usar .env

# --- Configuração da Chave API (Token) ---
# Use st.secrets para Streamlit Cloud
# load_dotenv() # Descomente para carregar de .env localmente
# groq_api_key = os.getenv("GROQ_API_KEY") # Para uso local com .env
try:
    # Tenta obter a chave do Streamlit Secrets (preferencial)
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (AttributeError, KeyError):
    # Fallback para variável de ambiente se secrets não funcionar (útil para dev local)
    # Certifique-se de definir GROQ_API_KEY no seu ambiente se não usar secrets
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Chave API GROQ não configurada. Defina GROQ_API_KEY nos secrets do Streamlit ou como variável de ambiente.")
        st.stop()

# --- Modelo Groq a ser Usado ---
# Escolha um modelo suportado e estável. Mixtral é uma boa opção.
# Verifique https://console.groq.com/docs/models para modelos atuais.
MODELO_GROQ_SELECIONADO = "mixtral-8x7b-32768"
# MODELO_GROQ_SELECIONADO = "llama3-70b-8192" # Alternativa, se Mixtral falhar

st.sidebar.info(f"Usando modelo Groq: `{MODELO_GROQ_SELECIONADO}`")

########## Analytics ##########

@st.cache_data
def dsa_extrai_dados(ticker, period="1y"):
    """Extrai dados históricos de uma ação (cacheado)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            st.error(f"Nenhum dado histórico encontrado para {ticker} no período {period}.")
            return None
        hist.reset_index(inplace=True)
        # Garante que 'Date' seja datetime sem timezone
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

    # Prepara o dataframe para o Prophet
    df_prophet = hist_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    if df_prophet.empty or len(df_prophet) < 2: # Prophet precisa de pelo menos 2 pontos
        st.warning("Dados insuficientes para gerar previsão com Prophet.")
        return None, None

    # Cria e treina o modelo Prophet
    # Desativar sazonalidades pode ser necessário se os dados forem muito curtos
    model = Prophet(daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True if len(df_prophet) > 365 else False, # Só ativa se tiver mais de 1 ano
                    changepoint_prior_scale=0.05)

    try:
        model.fit(df_prophet)
    except Exception as e:
        st.error(f"Erro ao treinar o modelo Prophet: {e}")
        return None, None

    # Cria datas futuras e gera previsão
    try:
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        st.error(f"Erro ao gerar datas futuras ou prever com Prophet: {e}")
        return model, None # Retorna modelo treinado, mas sem previsão

# --- Funções de Plotagem (Atualizadas) ---

def dsa_plot_stock_price(hist, forecast, ticker):
    """Plota preço histórico e previsão."""
    fig = go.Figure()
    # Histórico
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines+markers', name='Histórico'))

    # Previsão (se disponível)
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão (yhat)', line=dict(color='orange', dash='dash')))
        # Área de Incerteza
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Incerteza Superior', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Incerteza Inferior', line=dict(width=0), fillcolor='rgba(255, 165, 0, 0.2)', fill='tonexty', showlegend=False))

    fig.update_layout(title=f"{ticker} Preços das Ações e Previsão (3 Meses)", yaxis_title='Preço (USD)')
    st.plotly_chart(fig, use_container_width=True)

def dsa_plot_candlestick(hist, forecast, ticker):
    """Plota candlestick histórico e linha de previsão."""
    fig = go.Figure(data=[go.Candlestick(x=hist['Date'], open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Histórico')])

    # Previsão (se disponível)
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão Fechamento (yhat)', line=dict(color='orange', dash='dash')))

    fig.update_layout(title=f"{ticker} Candlestick e Previsão de Fechamento (3 Meses)", yaxis_title='Preço (USD)', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def dsa_plot_media_movel(hist, forecast, ticker):
    """Plota médias móveis históricas e previsão de fechamento."""
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()

    fig = go.Figure()
    # Histórico e Médias
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', name='Fechamento'))
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['SMA_20'], mode='lines', name='SMA 20 Dias'))
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['EMA_20'], mode='lines', name='EMA 20 Dias'))

    # Previsão (se disponível)
    if forecast is not None and not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Previsão Fechamento (yhat)', line=dict(color='orange', dash='dash')))

    fig.update_layout(title=f"{ticker} Médias Móveis e Previsão de Fechamento (3 Meses)", yaxis_title='Preço (USD)')
    st.plotly_chart(fig, use_container_width=True)

def dsa_plot_volume(hist, ticker):
    """Plota volume histórico."""
    fig = px.bar(hist, x='Date', y='Volume', title=f"{ticker} Volume de Negociação (Último Ano)")
    fig.update_layout(yaxis_title='Volume')
    st.plotly_chart(fig, use_container_width=True)

# --- Geração de Comentário da IA ---

def dsa_gera_comentario_previsao(ticker, hist_df, forecast_df, agent):
    """Gera comentário da IA sobre a previsão."""
    if forecast_df is None or forecast_df.empty or hist_df is None or hist_df.empty:
        return "Dados insuficientes ou erro na previsão para gerar comentário."

    try:
        last_date_hist = hist_df['Date'].iloc[-1].strftime('%Y-%m-%d')
        last_price_hist = hist_df['Close'].iloc[-1]
        # Encontra o índice correspondente à primeira data futura
        first_future_index = len(hist_df) # Índice logo após o último histórico
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

        Seja conciso e direto ao ponto. Não use informações externas ou de ferramentas, baseie-se somente nos dados fornecidos aqui. Não inclua tabelas na resposta, apenas texto. Não inicie a resposta com "Okay" ou frases de confirmação.
        """

        with st.spinner(f"Gerando comentário da IA sobre a previsão para {ticker}..."):
            ai_comment_response = agent.run(prompt)

            # Limpeza robusta da resposta
            clean_comment = ai_comment_response
            if hasattr(ai_comment_response, 'content'): # Se for um objeto de resposta
                 clean_comment = ai_comment_response.content

            # Remove padrões comuns de log/tool call que podem vazar
            clean_comment = re.sub(r"(Running|Calling|Using) tool.*?\n", "", clean_comment, flags=re.IGNORECASE | re.DOTALL)
            clean_comment = re.sub(r"\[.*?\]\(.*?\)", "", clean_comment) # Remove links markdown
            clean_comment = clean_comment.replace("```json", "").replace("```", "").strip() # Remove blocos de código
            clean_comment = re.sub(r"Okay, here's the analysis:", "", clean_comment, flags=re.IGNORECASE).strip()
            clean_comment = re.sub(r"^\s*Okay,\s*", "", clean_comment, flags=re.IGNORECASE).strip() # Remove "Okay," no início

            if not clean_comment or len(clean_comment) < 30:
                return "Não foi possível gerar um comentário da IA sobre a previsão (resposta vazia ou curta)."

            return clean_comment

    except IndexError:
         st.warning("Erro de índice ao processar dados para comentário da IA. Verifique o alinhamento histórico/previsão.")
         return "Erro ao processar dados para comentário da IA."
    except Exception as e:
        st.warning(f"Erro inesperado ao gerar comentário da IA: {e}")
        return "Ocorreu um erro ao gerar o comentário da IA sobre a previsão."


########## Agentes de IA ##########

# Instanciação dos agentes com o modelo selecionado
try:
    # Agente de Busca Web
    dsa_agente_web_search = Agent(
        name="DSA_Agente_Web_Search", # Nomes sem espaços podem ser mais seguros
        # role="Faz busca na web", # Role opcional
        llm=Groq(model=MODELO_GROQ_SELECIONADO, api_key=groq_api_key),
        tools=[DuckDuckGo()],
        instructions=["Você é um assistente de busca web.", "Use a ferramenta DuckDuckGo para encontrar informações atualizadas.", "Sempre inclua as fontes (URLs) nas suas respostas.", "Seja direto e informativo."],
        show_tool_calls=False, # Mantido como False para UI limpa
        markdown=True,
        output_schema=str # Força saída de string simples
    )

    # Agente Financeiro
    dsa_agente_financeiro = Agent(
        name="DSA_Agente_Financeiro",
        # role="Analista financeiro assistente",
        llm=Groq(model=MODELO_GROQ_SELECIONADO, api_key=groq_api_key),
        tools=[YFinanceTools(stock_price=False, # Preço atual pode vir do histórico
                             analyst_recommendations=True,
                             stock_fundamentals=True,
                             company_news=True)],
        instructions=["Você é um assistente de análise financeira.", "Use as ferramentas YFinanceTools para obter dados.", "Apresente recomendações de analistas e fundamentos da empresa em tabelas markdown.", "Resuma as notícias de forma concisa (3-5 pontos principais).", "Seja objetivo e foque nos dados."],
        show_tool_calls=False,
        markdown=True,
        output_schema=str
    )

    # Agente Orquestrador (Multi-Agente)
    multi_ai_agent = Agent(
        name="Orquestrador_Financeiro",
        llm=Groq(model=MODELO_GROQ_SELECIONADO, api_key=groq_api_key),
        # O team permite que o orquestrador chame os outros agentes
        team=[dsa_agente_web_search, dsa_agente_financeiro],
        # Ferramentas que o orquestrador pode precisar usar diretamente (ex: gerar comentário)
        # Nenhuma ferramenta adicional necessária aqui se ele só orquestrar ou gerar texto
        instructions=[
            "Sua tarefa principal é responder às consultas do usuário sobre análise de ações.",
            "**Delegação:**",
            "  - Para notícias, recomendações de analistas ou fundamentos da ação, delegue para `DSA_Agente_Financeiro`.",
            "  - Para informações gerais ou buscas na web, delegue para `DSA_Agente_Web_Search`.",
            "**Geração Própria:**",
             "  - Para gerar resumos ou comentários (como análises de previsões, quando o *prompt interno* solicitar), use sua própria capacidade de linguagem com base nos dados fornecidos no prompt.",
            "**Formatação:**",
            "  - Combine as informações dos agentes delegados de forma coesa.",
            "  - Use tabelas markdown para dados financeiros.",
            "  - Inclua fontes de buscas na web.",
            "**Estilo:**",
            "  - Seja claro, conciso e profissional.",
            "  - **IMPORTANTE:** Evite absolutamente qualquer menção a nomes de agentes, chamadas de ferramentas ou logs internos no seu output final para o usuário. Apenas forneça a resposta final consolidada."
        ],
        show_tool_calls=False,
        markdown=True,
         # debug_mode=True # Ative para ver o que está acontecendo internamente
        output_schema=str
    )

except Exception as e:
    st.error(f"Erro ao inicializar os Agentes de IA: {e}")
    st.error("Verifique o nome do modelo, a chave API Groq e a instalação das bibliotecas.")
    # Define os agentes como None para evitar erros posteriores
    dsa_agente_web_search = None
    dsa_agente_financeiro = None
    multi_ai_agent = None
    st.stop()


########## App Web ##########

# Configuração da página
st.set_page_config(page_title="Agente IA para NASDAQ", page_icon="📊", layout="wide")

# Barra Lateral
st.sidebar.title("Instruções")
st.sidebar.markdown(f"""
### Como Utilizar:

1.  Insira o símbolo do ticker da ação (ex: `MSFT`, `AAPL`) no campo abaixo.
2.  Clique em **Analisar**.
3.  Aguarde enquanto os dados são buscados, a previsão é gerada e a IA analisa as informações.

**Modelo de IA:** `{MODELO_GROQ_SELECIONADO}`
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

# Interface Principal
st.title("📈 Agente de IA para Análise de Ações da NASDAQ")
st.header("Análise Histórica, Previsão (3 Meses) e Insights de IA")

ticker = st.text_input("Digite o Código da Ação (ticker):", placeholder="Ex: AAPL, MSFT, NVDA").upper()

if st.button("Analisar", key="analyze_button"):
    if ticker and multi_ai_agent: # Verifica se o ticker foi inserido e os agentes inicializados
        st.markdown("---")
        progress_bar = st.progress(0, text="Iniciando análise...")
        analysis_successful = False # Flag para controlar o sucesso

        try:
            # 1. Obter Dados Históricos
            progress_bar.progress(10, text=f"Buscando dados históricos para {ticker}...")
            hist_data = dsa_extrai_dados(ticker)
            if hist_data is None:
                st.error(f"Falha ao obter dados históricos para {ticker}. Análise interrompida.")
                st.stop()

            # 2. Gerar Previsão
            progress_bar.progress(30, text=f"Gerando previsão de 3 meses para {ticker}...")
            model_prophet, forecast_data = dsa_gera_previsao(hist_data)
            if forecast_data is None:
                st.warning(f"Não foi possível gerar a previsão para {ticker}. Análise prosseguirá sem previsão.")
                ai_forecast_comment = "Previsão não disponível."
            else:
                # 3. Gerar Comentário da IA sobre a Previsão
                progress_bar.progress(50, text="Gerando comentário da IA sobre a previsão...")
                # Usa o agente orquestrador para o comentário
                ai_forecast_comment = dsa_gera_comentario_previsao(ticker, hist_data, forecast_data, multi_ai_agent)

            # 4. Obter Análise de Notícias e Recomendações
            progress_bar.progress(70, text=f"Buscando notícias e recomendações para {ticker}...")
            with st.spinner(f"Consultando IA para recomendações e notícias de {ticker}..."):
                 # Prompt claro para o orquestrador delegar
                 prompt_analise = f"Forneça um resumo das recomendações de analistas e as últimas notícias para a ação {ticker}. Use o `DSA_Agente_Financeiro` para obter os dados."
                 ai_analysis_response = multi_ai_agent.run(prompt_analise)

                 # Limpeza da resposta
                 clean_analysis_response = ai_analysis_response
                 if hasattr(ai_analysis_response, 'content'):
                      clean_analysis_response = ai_analysis_response.content
                 # Remove padrões de log que podem vazar
                 clean_analysis_response = re.sub(r"(Running|Using|Calling tool|Delegate to).*?\n", "", clean_analysis_response, flags=re.IGNORECASE | re.DOTALL).strip()
                 clean_analysis_response = re.sub(r"DSA_Agente_Financeiro", "Agente Financeiro", clean_analysis_response) # Generaliza nome
                 clean_analysis_response = re.sub(r"`", "", clean_analysis_response) # Remove backticks


            progress_bar.progress(90, text="Renderizando resultados...")

            # --- Exibição dos Resultados ---
            st.subheader(f"Análise por IA para {ticker}")
            if clean_analysis_response:
                st.markdown(clean_analysis_response)
            else:
                st.warning("Não foi possível obter a análise de notícias/recomendações da IA.")

            st.markdown("---")
            st.subheader("Visualização dos Dados e Previsão")

            # Gráfico de Preços
            st.markdown("##### Preço de Fechamento Histórico e Previsão")
            dsa_plot_stock_price(hist_data, forecast_data, ticker)
            if forecast_data is not None and not forecast_data.empty:
                st.markdown(f"**Comentário da IA sobre a Previsão:**\n {ai_forecast_comment}")
            st.markdown("---")

            # Gráfico Candlestick
            st.markdown("##### Candlestick Histórico e Previsão de Fechamento")
            dsa_plot_candlestick(hist_data, forecast_data, ticker)
            if forecast_data is not None and not forecast_data.empty:
                st.markdown(f"**Comentário da IA sobre a Previsão:**\n {ai_forecast_comment}")
            st.markdown("---")

            # Gráfico de Médias Móveis
            st.markdown("##### Médias Móveis Históricas e Previsão de Fechamento")
            dsa_plot_media_movel(hist_data, forecast_data, ticker)
            if forecast_data is not None and not forecast_data.empty:
                st.markdown(f"**Comentário da IA sobre a Previsão:**\n {ai_forecast_comment}")
            st.markdown("---")

            # Gráfico de Volume
            st.markdown("##### Volume de Negociação Histórico")
            dsa_plot_volume(hist_data, ticker)
            st.markdown("**Nota:** A previsão de volume não está incluída nesta análise.")
            st.markdown("---")

            progress_bar.progress(100, text="Análise concluída!")
            analysis_successful = True

        except Exception as e:
            st.error(f"Ocorreu um erro inesperado durante a análise: {e}")
            st.exception(e) # Mostra detalhes do erro para depuração
        finally:
            # Limpa a barra de progresso, mostra mensagem final
            progress_bar.empty()
            if analysis_successful:
                st.success(f"Análise para {ticker} concluída!")
            else:
                st.error(f"Análise para {ticker} encontrou problemas.")

    elif not ticker:
        st.error("Por favor, insira um código de ação (ticker).")
    elif not multi_ai_agent:
         st.error("Os agentes de IA não puderam ser inicializados. Verifique a configuração e os erros acima.")

# Fim
