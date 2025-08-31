import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import json
import locale

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

def load_page_config():
    try:
        st.set_page_config(
            page_title="Chatbot de Vendas", 
            page_icon="sesi-logo.png",
            layout="wide")
    except Exception as e:
        st.set_page_config(
            page_title="Chatbot de Vendas",
            layout="wide")
        st.error(f"Não foi possível carregar logo da página. Erro: {e}")

load_page_config()

client = OpenAI(api_key="OPENAI_API_KEY")

st.title("🤖 Chatbot")

# ---- Carregar dados ----
def load_data():
    produtos = pd.read_excel("base_produtos.xlsx")
    vendas = pd.read_excel("base_vendas.xlsx")
    vendedores = pd.read_excel("base_vendedores.xlsx")

    df = vendas.merge(produtos, on="Id_Produto")
    df = df.merge(vendedores, on="Id_Vendedor")
    columns = ['Id_Venda', "Nome_Vendedor", "Região", "Nome_Produto", "Categoria", "Data_Venda", "Quantidade","R$_Unit_x", "R$_Total"]
    df = df[columns]
    df = df.rename(columns={"R$_Unit_x": "Valor_Unitario", "R$_Total": "Valor_Venda"})
    df['Data_Venda'] = pd.to_datetime(df['Data_Venda'], errors='coerce')
    df = df.fillna(0)
    df["ano"] = df["Data_Venda"].dt.year
    return df

st.write("### Dados de Vendas")

df = load_data()
st.dataframe(df)

# ---- Modelo preditivo ----
def prepare_regression_frame(series):
    X = pd.DataFrame({"ds": series.index, "y": series.values})
    X["t"] = np.arange(len(X))
    X["month"] = X["ds"].dt.month.astype(int)
    X["year"] = X["ds"].dt.year.astype(int)
    return X


def fit_and_forecast(series, horizon=3, test_quarter=True):
    X = prepare_regression_frame(series)
    features = ["t","month","year"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["month"]),
        ("num", "passthrough", ["t","year"])
    ])

    model = Pipeline([
        ("pre", pre),
        ("xgb", XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])

    # Split último trimestre como teste
    test_size = horizon if test_quarter else 0
    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:] if test_quarter else None

    # Treina
    model.fit(X_train[features], X_train["y"])

    # Avaliação no último trimestre
    if test_quarter:
        y_pred = model.predict(X_test[features])
        mae = mean_absolute_error(X_test["y"], y_pred)
        rmse = mean_squared_error(X_test["y"], y_pred) ** 0.5
        mape = (np.abs((X_test["y"] - y_pred) / X_test["y"])).mean() * 100
        mape = round(mape, 2)
        metrics = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}
        #st.warning(f"Média percentual de erro: {mape}%")
    else:
        metrics = {}

    # Previsão futura
    last_date = X["ds"].max()
    future_idx = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    Xf = pd.DataFrame({"ds": future_idx})
    Xf["t"] = np.arange(len(X), len(X)+horizon)
    Xf["month"] = Xf["ds"].dt.month.astype(int)
    Xf["year"] = Xf["ds"].dt.year.astype(int)
    yhat_future = model.predict(Xf[features])

    return X_test, y_pred if test_quarter else None, Xf, yhat_future, metrics

# remover vendas com valor zero/negativo
if "Valor_Venda" in df.columns:
    df = df[df["Valor_Venda"] > 0]

monthly = df.groupby(pd.Grouper(key="Data_Venda", freq="MS"))["Valor_Venda"].sum().sort_index().asfreq("MS").fillna(0)

def modelagem_preditiva(n=3):
    X_test, y_pred_test, Xf, yhat_future, metrics = fit_and_forecast(monthly, horizon=n, test_quarter=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, label="Histórico", marker="o")
    #if X_test is not None and y_pred_test is not None:
        #ax.plot(X_test["ds"], y_pred_test, "o--", label="Previsão (teste)")
    ax.plot(Xf["ds"], yhat_future, "s--", label="Previsão (futuro)")
    ax.set_ylabel("Valor (R$)")
    ax.set_xlabel("Mês")
    ax.legend()
    st.pyplot(fig)


    df_prev = pd.DataFrame({"Mês": Xf["ds"],
                            "Previsão de Vendas": yhat_future})
    df_prev["Mês"] = pd.to_datetime(df_prev["Mês"]).dt.strftime('%B/%Y')
    df_prev["Previsão de Vendas"] = df_prev["Previsão de Vendas"].map("R$ {:,.2f}".format)
    df_prev["Previsão de Vendas"] = df_prev["Previsão de Vendas"].str.replace(",", "X").str.replace(".", ",").str.replace("X", ".")
    #st.write(df_prev)

    return df_prev.to_dict(orient="records")  # Converte para algo que o GPT consegue exibir

def vend_poten(n=3):
    crescimento = []
    min_months = max(6, n)  # mínimo de meses com dados para considerar
    for vendedor in df["Nome_Vendedor"].unique():
        vendas = df[df["Nome_Vendedor"] == vendedor].groupby(
            pd.Grouper(key="Data_Venda", freq="MS")
        )["Valor_Venda"].sum().sort_index().asfreq("MS").fillna(0)

        if vendas.notna().sum() >= min_months:
            X_test_v, y_pred_test_v, Xf_v, yhat_future_v, metrics_v = fit_and_forecast(vendas, horizon=n, test_quarter=True)
            ultimo_periodo = vendas.iloc[-n:].sum() if len(vendas) >= n else vendas.sum()
            prev_ult_period = float(np.sum(y_pred_test_v))
            prox_periodo = float(np.sum(yhat_future_v))
            dif = ultimo_periodo/prev_ult_period 
            # crescimento %
            if ultimo_periodo == 0:
                perc_growth = np.nan
            else:
                perc_growth = (prox_periodo - ultimo_periodo) / ultimo_periodo * 100.0
            crescimento.append({
                "Vendedor": vendedor,
                "Último Período": ultimo_periodo,
                "Previsão p/ últ. período": prev_ult_period,
                "Percentual de acerto": dif,
                "Crescimento (%)": perc_growth,
                "Receita Prevista Próx. Período": prox_periodo,
                "MAE_teste": metrics_v.get("MAE") if isinstance(metrics_v, dict) else None,
                "RMSE_teste": metrics_v.get("RMSE") if isinstance(metrics_v, dict) else None,
                "MAPE_teste(%)": metrics_v.get("MAPE (%)") if isinstance(metrics_v, dict) else None
            })
        
    crescimento_df = pd.DataFrame(crescimento).sort_values("Crescimento (%)", ascending=False)
    #st.dataframe(crescimento_df.head(20))
    
    return crescimento_df.to_dict(orient="records")  # Converte para algo que o GPT consegue exibir
    


# ---- Funções para consulta ----
def calcular_vendas_totais():
    return float(df["Valor_Venda"].sum())

def top_prod_cat_ano():
    prod_cat_period = df.groupby(["Nome_Produto", "Categoria", "ano"])["Valor_Venda"].sum().reset_index()
    return prod_cat_period.sort_values(["Categoria","ano", "Valor_Venda"], ascending=[True, True, False]).groupby(["Categoria","ano"]).head(5)

def top5_vende_prod(n=5):
    vende_prod = df.groupby(["Nome_Vendedor", "Nome_Produto"])["Valor_Venda"].sum().reset_index()
    return vende_prod.sort_values(["Nome_Vendedor", "Valor_Venda"], ascending=[True, False]).groupby("Nome_Vendedor").head(5)

def detalhe_prod():
    produto = df.groupby(['Nome_Produto', "Categoria", "Valor_Unitario"])[["Valor_Venda", "Quantidade"]].sum().reset_index()
    return produto[produto["Valor_Unitario"] != 0.00]

def detalhe_vendedor():
    return df.groupby(["Nome_Vendedor", "ano", "Região"])[["Quantidade", "Valor_Venda"]].sum().reset_index()

def regiao():
    return df.groupby("Região")[["Quantidade", "Valor_Venda"]].sum().reset_index()

# ---- Definição das funções para o GPT ----
tools = [
    {
        "type": "function",
        "function": {
            "name": "calcular_vendas_totais",
            "description": "Retorna o valor total de vendas no DataFrame.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "top_prod_cat_ano",
            "description": "Lista a soma dos valores de vendas dos produtos mais vendidos por categoria e ano.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Quantidade de produtos a retornar"}},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "top5_vende_prod",
            "description": "Lista os 5 produtos mais vendidos por vendedor, com o total de vendas.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Quantidade de produtos a retornar"}},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detalhe_prod",
            "description": "Detalhes do produto, como nome, categoria, preço (Valor_Unitario), "
                            "total de quantidades vendidas e total de valor de venda (faturamento)",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Quantidade de produtos a retornar"}},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detalhe_vendedor",
            "description": "Detalhes das vendas do vendedor, como nome, ano, região, "
                            "total de quantidades vendidas e total de valor de venda (faturamento)",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Quantidade de produtos a retornar"}},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "regiao",
            "description": "Quantidade e total de vendas (faturamento) por região",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Quantidade de produtos a retornar"}},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modelagem_preditiva",
            "description": (
                "Realiza previsão de vendas futuras com base no histórico. "
                "Use esta função quando o usuário pedir análise preditiva, "
                "projeção, previsão ou estimativa de vendas futuras. "
                "Retorna gráfico e tabela com os meses previstos e seus valores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Quantidade de meses a projetar (padrão = 3)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vend_poten",
            "description": (
                "Elabora um ranking dos vendedores com maior potencial de crescimento das vendas no período previsto. "
                "Use esta função quando o usuário pedir vendedores com potencial, "
                "projeção ou previsão de crescimento nas vendas futuras. "
                #"Retorna gráfico e tabela com os meses previstos e seus valores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Quantidade de meses a projetar (padrão = 3)."
                    }
                },
                "required": []
            }
        }
    }
]


# ---- Histórico de conversa ----
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": (
            "Você é um assistente especializado em vendas. "
            "Você possui acesso a funções que consultam os dados de vendas e também "
            "a funções que realizam análises preditivas e projeções futuras de vendas. "
            "Sempre que o usuário pedir previsões, projeções ou análises preditivas, "
            "use a função 'modelagem_preditiva'."
            " Sempre que o usuário pedir vendedores com potencial de crescimento, "
            "use a função 'vend_poten'."
            "Sempre passar valores finais em Reais (R$) com formatação brasileira."
            )
        }
    ]


for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ---- Entrada do usuário ----
if user_input := st.chat_input("Pergunte sobre as vendas..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- Envia para GPT com tools ----
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=st.session_state.messages,
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        # GPT pediu para chamar uma função
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if func_name == "calcular_vendas_totais":
                result = calcular_vendas_totais()
            elif func_name == "top_prod_cat_ano":
                result = top_prod_cat_ano()
            elif func_name == "top5_vende_prod":
                result = top5_vende_prod(**args)
            elif func_name == "detalhe_prod":
                result = detalhe_prod()
            elif func_name == "detalhe_vendedor":
                result = detalhe_vendedor()
            elif func_name == "regiao":
                result = regiao()
            elif func_name == "modelagem_preditiva":
                result = modelagem_preditiva(**args)
            elif func_name == "vend_poten":
                result = vend_poten(**args)
            else:
                result = "Função não reconhecida."

            # Resposta do GPT após ver o resultado da função
            followup = client.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.messages + [msg, {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}]
            )

            bot_reply = followup.choices[0].message.content
    else:
        bot_reply = msg.content

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
