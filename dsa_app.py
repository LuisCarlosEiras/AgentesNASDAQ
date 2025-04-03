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
import pandas as pd # Adicionado para manipulação de datas
from prophet import Prophet # Adicionado para previsão
from prophet.plot import plot_plotly # Adicionado para plotar com plotly

# Configuração da chave API (Token) - Assumindo que está no secrets do Streamlit
# load_dotenv() # Descomente se estiver usando um arquivo .env localmente
# groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key = st.secrets["GROQ_API_KEY"] # Mantido para Streamlit Cloud

########## Analytics ##########

# Usa o cache de dados do Streamlit para armazenar os resultados da função e evitar reprocessamento
# Define a função que extrai dados históricos de uma ação com base no ticker e período especificado
@st.cache_data
def dsa_extrai_dados(ticker, period="1y"): # Aumentado para 1 ano para dar mais dados à previsão

    # Cria um objeto Ticker do Yahoo Finance para a ação especificada
    stock = yf.Ticker(ticker)

    # Obtém o histórico de preços da ação para o período definido
    hist = stock.history(period=period)

    # Reseta o índice do DataFrame para transformar a coluna de data em uma coluna normal
    hist.reset_index(inplace=True)

    # Converte a coluna 'Date' para datetime (removendo timezone se houver)
    hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)


    # Retorna o DataFrame com os dados históricos da ação
    return hist

# Define a função para gerar a previsão futura usando Prophet
@st.cache_data
def dsa_gera_previsao(hist_df, periods=90): # periods=90 para ~3 meses
    # Prepara o dataframe para o Prophet (requer colunas 'ds' e 'y')
    df_prophet = hist_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    # Cria e treina o modelo Prophet
    model = Prophet(daily_seasonality=False, # Mercado financeiro não costuma ter sazonalidade diária forte
                    weekly_seasonality=True, # Sazonalidade semanal pode existir
                    yearly_seasonality=True, # Sazonalidade anual também
                    changepoint_prior_scale=0.05) # Ajuste padrão
    try:
        model.fit(df_prophet)
    except Exception as e:
        st.error(f"Erro ao treinar o modelo Prophet: {e}")
        return None, None # Retorna None se houver erro no treinamento

    # Cria um dataframe com datas futuras para previsão
    future = model.make_future_dataframe(periods=periods)

    # Gera a previsão
    forecast = model.predict(future)

    # Retorna o modelo treinado e o dataframe de previsão
    return model, forecast

# Define a função para plotar o preço das ações com base no histórico e previsão
def dsa_plot_stock_price(hist, forecast, ticker):
    # Cria figura base com dados históricos
    fig = px.line(hist, x="Date", y="Close", title=f"{ticker} Preços das Ações (Último Ano) e Previsão (Próximos 3 Meses)")

    # Adiciona a linha de previsão (yhat)
    fig.add_trace(go.Scatter(x=forecast['ds'],
                             y=forecast['yhat'],
                             mode='lines',
                             name='Previsão (yhat)',
                             line=dict(color='orange', dash='dash')))

    # Adiciona a área de incerteza (opcional, mas útil)
    fig.add_trace(go.Scatter(x=forecast['ds'],
                             y=forecast['yhat_upper'],
                             mode='lines', name='Incerteza Superior',
                             line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'],
                             y=forecast['yhat_lower'],
                             mode='lines', name='Incerteza Inferior',
                             line=dict(width=0),
                             fillcolor='rgba(255, 165, 0, 0.2)', # Cor laranja semitransparente
                             fill='tonexty', # Preenche até o traço anterior (yhat_upper)
                             showlegend=False))

    # Adiciona marcadores aos dados históricos originais para melhor visualização
    fig.data[0].update(mode='lines+markers') # Assumindo que o histórico é o primeiro traço

    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True) # Ajusta à largura do container

# Define a função para plotar um gráfico de candlestick com previsão sobreposta
def dsa_plot_candlestick(hist, forecast, ticker):

    # Cria um objeto Figure do Plotly para armazenar o gráfico
    fig = go.Figure(
        # Adiciona um gráfico de candlestick com os dados do histórico da ação
        data=[go.Candlestick(x=hist['Date'],
                             open=hist['Open'],
                             high=hist['High'],
                             low=hist['Low'],
                             close=hist['Close'],
                             name='Histórico')]
    )

    # Adiciona a linha de previsão (yhat) sobre o gráfico candlestick
    fig.add_trace(go.Scatter(x=forecast['ds'],
                             y=forecast['yhat'],
                             mode='lines',
                             name='Previsão (Fechamento Estimado)',
                             line=dict(color='orange', dash='dash')))

    # Atualiza o layout do gráfico, incluindo um título dinâmico com o ticker da ação
    fig.update_layout(title=f"{ticker} Candlestick (Último Ano) com Previsão de Fechamento (Próximos 3 Meses)")

    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True) # Ajusta à largura do container

# Define a função para plotar médias móveis com previsão sobreposta
def dsa_plot_media_movel(hist, forecast, ticker):

    # Calcula a Média Móvel Simples (SMA) de 20 períodos e adiciona ao DataFrame
    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()

    # Calcula a Média Móvel Exponencial (EMA) de 20 períodos e adiciona ao DataFrame
    hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()

    # Cria um gráfico de linha interativo usando Plotly Express
    # Plota os preços de fechamento, a SMA de 20 períodos e a EMA de 20 períodos
    fig = px.line(hist,
                  x='Date',
                  y=['Close', 'SMA_20', 'EMA_20'],
                  title=f"{ticker} Médias Móveis (Último Ano) com Previsão de Fechamento", # Define o título do gráfico
                  labels={'value': 'Preço (USD)', 'variable': 'Métrica'}) # Define os rótulos dos eixos

    # Adiciona a linha de previsão (yhat)
    fig.add_trace(go.Scatter(x=forecast['ds'],
                             y=forecast['yhat'],
                             mode='lines',
                             name='Previsão Fechamento (yhat)',
                             line=dict(color='orange', dash='dash')))

    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True) # Ajusta à largura do container

# Define a função para plotar o volume de negociação da ação (sem previsão direta de volume)
def dsa_plot_volume(hist, ticker):

    # Cria um gráfico de barras interativo usando Plotly Express
    # O eixo X representa a data e o eixo Y representa o volume negociado
    fig = px.bar(hist,
                 x='Date',
                 y='Volume',
                 title=f"{ticker} Volume de Negociação (Último Ano)") # Define o título do gráfico

    # Exibe o gráfico no Streamlit
    st.plotly_chart(fig, use_container_width=True) # Ajusta à largura do container

# Função para gerar comentário da IA sobre a previsão
def dsa_gera_comentario_previsao(ticker, hist_df, forecast_df, agent):
    try:
        # Extrai informações relevantes para o prompt
        last_date_hist = hist_df['Date'].iloc[-1].strftime('%Y-%m-%d')
        last_price_hist = hist_df['Close'].iloc[-1]
        forecast_start_date = forecast_df['ds'].iloc[len(hist_df)].strftime('%Y-%m-%d') # Primeira data da previsão
        forecast_end_date = forecast_df['ds'].iloc[-1].strftime('%Y-%m-%d')
        forecast_end_price = forecast_df['yhat'].iloc[-1]
        forecast_max_price = forecast_df['yhat_upper'].iloc[-1]
        forecast_min_price = forecast_df['yhat_lower'].iloc[-1]

        # Cria o prompt para o agente de IA
        prompt = f"""
        Analise a seguinte previsão de preço para a ação {ticker} para os próximos 3 meses, gerada pelo modelo Prophet.

        Dados Históricos Relevantes:
        - Última data histórica: {last_date_hist}
        - Último preço de fechamento histórico: ${last_price_hist:.2f}

        Previsão para os Próximos 3 Meses (até {forecast_end_date}):
        - Preço previsto para {forecast_end_date}: ${forecast_end_price:.2f}
        - Faixa de Incerteza para {forecast_end_date}: entre ${forecast_min_price:.2f} e ${forecast_max_price:.2f}

        Com base *apenas* nesses dados de previsão e no último preço histórico:
        1. Descreva a tendência geral prevista (alta, baixa, estável).
        2. Comente brevemente sobre a confiança da previsão, mencionando a faixa de incerteza.
        3. Forneça uma breve conclusão sobre o que a previsão sugere para os próximos 3 meses.

        Seja conciso e direto ao ponto. Não use informações externas ou de ferramentas, baseie-se somente nos dados fornecidos aqui. Não inclua tabelas na resposta, apenas texto.
        """

        # Executa o agente para gerar o comentário
        with st.spinner(f"Gerando comentário da IA sobre a previsão para {ticker}..."):
            ai_comment = agent.run(prompt)

            # Limpa a resposta da IA (se necessário, ajuste conforme o output do seu agente)
            clean_comment = re.sub(r"(Running:[\s\S]*?\n\n)|(^transfer_task_to.*?\n?)","", ai_comment.content, flags=re.MULTILINE).strip()
            # Remove possíveis chamadas de ferramenta ou texto indesejado
            clean_comment = re.sub(r"\[.*?\]\(.*?\)", "", clean_comment) # Remove links markdown se houver
            clean_comment = clean_comment.replace("```json", "").replace("```", "").strip() # Remove blocos de código se houver

            # Verifica se o comentário parece razoável (evita mensagens de erro do agente)
            if "não posso" in clean_comment.lower() or "erro" in clean_comment.lower() or len(clean_comment) < 50 :
                 return "Não foi possível gerar um comentário detalhado da IA sobre a previsão neste momento."

            return clean_comment

    except Exception as e:
        st.warning(f"Erro ao gerar comentário da IA: {e}")
        return "Ocorreu um erro ao gerar o comentário da IA sobre a previsão."

########## Agentes de IA ##########

# Agentes de IA
dsa_agente_web_search = Agent(
    name="DSA Agente Web Search",
    role="Fazer busca na web",
    # model=Groq(model="mixtral-8x7b-32768", api_key=groq_api_key), # Use um modelo disponível se deepseek não estiver
    model=Groq(model="llama3-70b-8192", api_key=groq_api_key), # Llama3 70b é potente
    tools=[DuckDuckGo()],
    instructions=["Sempre inclua as fontes", "Seja direto e informativo."],
    show_tool_calls=False, # Desligar para output mais limpo
    markdown=True
)

dsa_agente_financeiro = Agent(
    name="DSA Agente Financeiro",
    # model=Groq(model="mixtral-8x7b-32768", api_key=groq_api_key),
    model=Groq(model="llama3-70b-8192", api_key=groq_api_key),
    tools=[YFinanceTools(stock_price=True,
                         analyst_recommendations=True,
                         stock_fundamentals=True,
                         company_news=True)],
    instructions=["Use tabelas markdown para mostrar os dados financeiros e recomendações.", "Resuma as notícias de forma concisa."],
    show_tool_calls=False, # Desligar para output mais limpo
    markdown=True
)

multi_ai_agent = Agent(
    # O agente orquestrador pode ser um modelo mais capaz
    # model=Groq(model="llama3-70b-8192", api_key=groq_api_key),
    model=Groq(model="llama3-70b-8192", api_key=groq_api_key),
    team=[dsa_agente_web_search, dsa_agente_financeiro],
    # Adicionei o agente financeiro aqui também para que o orquestrador possa usar suas ferramentas diretamente se necessário
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Sua tarefa principal é orquestrar os outros agentes ou usar ferramentas para responder à consulta do usuário.",
                  "Para notícias e recomendações, use o 'DSA Agente Financeiro'.",
                  "Para buscas gerais na web, use o 'DSA Agente Web Search'.",
                  "Para gerar comentários sobre previsões (quando solicitado explicitamente no prompt interno), use sua própria capacidade de linguagem.",
                  "Sempre inclua as fontes quando usar busca na web.",
                  "Use tabelas markdown para dados financeiros.",
                  "Combine as informações de forma coesa e clara para o usuário.",
                  "Seja direto e evite mensagens de 'Running tool' ou nomes de agentes no output final."],
    show_tool_calls=False, # Desligar para output mais limpo
    markdown=True
)


########## App Web ##########

# Configuração da página do Streamlit
st.set_page_config(page_title="Agente IA para NASDAQ", page_icon="📊", layout="wide")

# Barra Lateral com instruções
st.sidebar.title("Instruções")
st.sidebar.markdown("""
### Como Utilizar a App:

- Insira o símbolo do ticker da ação desejada (ex: `MSFT`, `AAPL`, `GOOGL`) no campo central.
- Clique no botão **Analisar** para obter a análise com gráficos históricos, previsões para 3 meses e insights gerados por IA.

### Exemplos de tickers válidos:
- MSFT (Microsoft)
- AAPL (Apple)
- TSLA (Tesla)
- AMZN (Amazon)
- GOOG (Alphabet)

Mais tickers podem ser encontrados aqui: https://stockanalysis.com/list/nasdaq-stocks/

### Finalidade da App:
Este aplicativo realiza análises de preços de ações da Nasdaq, incluindo previsões de curto prazo (3 meses) usando modelos estatísticos e comentários gerados por IA para apoiar a tomada de decisão.
""")

# Botão de suporte na barra lateral
if st.sidebar.button("Suporte"):
    st.sidebar.write("Em caso de dúvidas, contate: luiscarloseiras@gmail.com") # Corrigido email

# Título principal
st.title("📈 Agente de IA para Análise de Ações da NASDAQ")

# Interface principal
st.header("Análise Histórica, Previsão e Insights de IA")

# Caixa de texto para input do usuário
ticker = st.text_input("Digite o Código da Ação (ticker):", placeholder="Ex: AAPL, MSFT, NVDA").upper()

# Se o usuário pressionar o botão, entramos neste bloco
if st.button("Analisar", key="analyze_button"):

    # Se temos o código da ação (ticker)
    if ticker:

        # Inicia o processamento
        st.markdown("---") # Linha divisória
        progress_bar = st.progress(0, text="Iniciando análise...")

        try:
            # 1. Obter dados históricos
            progress_bar.progress(10, text=f"Buscando dados históricos para {ticker}...")
            hist = dsa_extrai_dados(ticker)
            if hist is None or hist.empty:
                 st.error(f"Não foi possível obter dados históricos para {ticker}. Verifique o código ou tente novamente.")
                 st.stop() # Para a execução se não houver dados

            # 2. Gerar Previsão
            progress_bar.progress(30, text=f"Gerando previsão de 3 meses para {ticker}...")
            model, forecast = dsa_gera_previsao(hist)
            if forecast is None:
                st.warning(f"Não foi possível gerar a previsão para {ticker}. Os gráficos históricos ainda serão exibidos.")
                # Define forecast como um DataFrame vazio para não quebrar as funções de plotagem
                forecast = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
                ai_forecast_comment = "Previsão não disponível." # Define comentário padrão
            else:
                 # 3. Gerar Comentário da IA sobre a Previsão (se a previsão foi bem-sucedida)
                 progress_bar.progress(50, text=f"Gerando comentário da IA sobre a previsão...")
                 # Passa o agente orquestrador para gerar o comentário
                 ai_forecast_comment = dsa_gera_comentario_previsao(ticker, hist, forecast, multi_ai_agent)


            # 4. Obter Análise de Notícias e Recomendações da IA
            progress_bar.progress(70, text=f"Buscando notícias e recomendações de analistas para {ticker}...")
            with st.spinner(f"Buscando recomendações e notícias para {ticker}..."):
                 # Usamos diretamente o agente orquestrador que delegará a tarefa
                 ai_response = multi_ai_agent.run(f"Forneça um resumo das recomendações de analistas e as últimas notícias para a ação {ticker}. Use o 'DSA Agente Financeiro'.")

                 # Limpeza da resposta (pode precisar de ajustes)
                 clean_response = re.sub(r"(Running:[\s\S]*?\n\n)|(^transfer_task_to.*?\n?)|(```json[\s\S]*?```)", "", ai_response.content, flags=re.MULTILINE).strip()
                 clean_response = clean_response.replace("Okay, contacting DSA Agente Financeiro...", "").strip() # Exemplo de limpeza adicional


            progress_bar.progress(90, text="Renderizando resultados...")

            # 5. Exibir Análise da IA (Notícias/Recomendações)
            st.subheader(f"Análise por IA para {ticker}")
            if clean_response:
                 st.markdown(clean_response)
            else:
                 st.warning("Não foi possível obter a análise de notícias/recomendações da IA.")


            # 6. Exibir Gráficos com Previsão e Comentários
            st.markdown("---")
            st.subheader("Visualização dos Dados e Previsão")

            # Gráfico de Preços
            st.markdown("##### Preço de Fechamento Histórico e Previsão")
            dsa_plot_stock_price(hist, forecast, ticker)
            if not forecast.empty: # Só mostra comentário se a previsão existe
                 st.markdown(f"**Comentário da IA sobre a Previsão:**\n {ai_forecast_comment}")
                 st.markdown("---") # Separador

            # Gráfico Candlestick
            st.markdown("##### Candlestick Histórico e Previsão de Fechamento")
            dsa_plot_candlestick(hist, forecast, ticker)
            if not forecast.empty:
                 st.markdown(f"**Comentário da IA sobre a Previsão:**\n {ai_forecast_comment}")
                 st.markdown("---")

            # Gráfico de Médias Móveis
            st.markdown("##### Médias Móveis Históricas e Previsão de Fechamento")
            dsa_plot_media_movel(hist, forecast, ticker)
            if not forecast.empty:
                 st.markdown(f"**Comentário da IA sobre a Previsão:**\n {ai_forecast_comment}")
                 st.markdown("---")

            # Gráfico de Volume (sem previsão)
            st.markdown("##### Volume de Negociação Histórico")
            dsa_plot_volume(hist, ticker)
            st.markdown("**Nota:** A previsão de volume não está incluída nesta análise.")
            st.markdown("---")

            progress_bar.progress(100, text="Análise concluída!")
            st.success(f"Análise para {ticker} concluída com sucesso!")

        except Exception as e:
            progress_bar.empty() # Remove a barra de progresso em caso de erro
            st.error(f"Ocorreu um erro inesperado durante a análise: {e}")
            st.exception(e) # Mostra o traceback para depuração

    else:
        st.error("Ticker inválido ou vazio. Insira um símbolo de ação válido (ex: AAPL).")


# Fim
