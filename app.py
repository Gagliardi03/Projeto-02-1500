from openai import OpenAI
import streamlit as st
import sqlite3  # Adicionado para usar SQLite
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import os
import time

# Configurações iniciais
OPENAI_API_KEY = 

# Configuração do SQLite
def init_db():
    conn = sqlite3.connect("chat_thec.db")
    cursor = conn.cursor()
    # Tabela para chats ativos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            messages TEXT
        )
    """)
    # Tabela para chats excluídos
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deleted_chats (
            id TEXT PRIMARY KEY,
            messages TEXT
        )
    """)
    conn.commit()
    return conn

# Inicializar o banco de dados
conn = init_db()

# Configuração do Qdrant
qdrant_client = QdrantClient(host="localhost", port=6333)

# Configuração do LangChain com a OpenAI
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Configuração do Qdrant como vectorstore
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="openai_collection",
    embeddings=embeddings
)

# Função para carregar e dividir o texto em chunks
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Carregar a base de dados
with open("./base_dados.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Dividir o texto em chunks e adicionar ao Qdrant
texts = get_chunks(raw_text)
vectorstore.add_texts(texts)

# Configuração do RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Função para consultar a base de dados vetorizada
def consultar_base_dados(modelo, problema):
    query = f"Modelo: {modelo}, Problema: {problema}"
    return qa.run(query)

# Carregar a base de técnicos
tecnicos_df = pd.read_csv("tecnicos.tsv", sep="\t")

# Funções para gerenciar técnicos agendados
def load_tecnicos_agendados():
    if "tecnicos_agendados" not in st.session_state:
        st.session_state["tecnicos_agendados"] = []
    return st.session_state["tecnicos_agendados"]

def save_tecnicos_agendados(tecnicos_agendados):
    st.session_state["tecnicos_agendados"] = tecnicos_agendados

# Função para consultar técnicos disponíveis
def consultar_tecnicos(modelo_impressora):
    modelo_impressora = modelo_impressora.strip().lower()
    tecnicos_agendados = load_tecnicos_agendados()
    
    # Filtra técnicos que têm a especialidade correspondente ao modelo da impressora
    tecnicos_disponiveis = tecnicos_df[
        (tecnicos_df["Especialidade"].str.strip().str.lower().str.contains(modelo_impressora, case=False, regex=False)) &
        (~tecnicos_df["Nome"].isin(tecnicos_agendados))  # Exclui técnicos já agendados
    ]
    
    return tecnicos_disponiveis

# Função para agendar técnico
def agendar_tecnico(modelo_impressora, tecnico_nome):
    tecnicos_agendados = load_tecnicos_agendados()
    
    # Verifica se o técnico já está agendado
    if tecnico_nome in tecnicos_agendados:
        st.error(f"Técnico {tecnico_nome} já está agendado.")
        return
    
    # Adiciona o técnico à lista de agendados
    tecnicos_agendados.append(tecnico_nome)
    save_tecnicos_agendados(tecnicos_agendados)
    st.success(f"Técnico {tecnico_nome} agendado com sucesso!")
# Funções para gerenciar chats no SQLite
def load_chat_history(chat_id):
    cursor = conn.cursor()
    cursor.execute("SELECT messages FROM chats WHERE id = ?", (chat_id,))
    result = cursor.fetchone()
    return eval(result[0]) if result else []

def save_chat_history(chat_id, messages):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO chats (id, messages)
        VALUES (?, ?)
    """, (chat_id, str(messages)))
    conn.commit()

def load_deleted_chats():
    cursor = conn.cursor()
    cursor.execute("SELECT id, messages FROM deleted_chats")
    return {row[0]: eval(row[1]) for row in cursor.fetchall()}

def save_deleted_chats(chat_id, messages):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO deleted_chats (id, messages)
        VALUES (?, ?)
    """, (chat_id, str(messages)))
    conn.commit()

def delete_chat(chat_id):
    # Carrega as mensagens do chat que será excluído
    messages = load_chat_history(chat_id)
    
    # Salva o chat excluído na tabela de chats excluídos
    save_deleted_chats(chat_id, messages)
    
    # Remove o chat da tabela de chats ativos
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    
    # Remove o chat da lista de chats ativos na sessão
    if chat_id in st.session_state["chat_ids"]:
        st.session_state["chat_ids"].remove(chat_id)
    
    # Se o chat atual for o que está sendo excluído, redefine o chat atual como None
    if st.session_state["current_chat_id"] == chat_id:
        st.session_state["current_chat_id"] = None
        st.session_state["messages"] = []
    
    # Força a atualização da interface
    st.rerun()
def delete_all_deleted_chats():
    cursor = conn.cursor()
    cursor.execute("DELETE FROM deleted_chats")
    conn.commit()

# Inicializar ou carregar o estado da sessão
if "current_chat_id" not in st.session_state:
    st.session_state["current_chat_id"] = None
if "chat_ids" not in st.session_state:
    st.session_state["chat_ids"] = []
if "show_deleted_chats" not in st.session_state:
    st.session_state["show_deleted_chats"] = False
if "selected_deleted_chat" not in st.session_state:
    st.session_state["selected_deleted_chat"] = None
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Interface do Streamlit
st.title("CHAT THEC")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Sidebar para gerenciar conversas e histórico de chats excluídos
# Sidebar para gerenciar conversas e histórico de chats excluídos
with st.sidebar:
    st.header("Conversas")
    if st.button("Nova Conversa"):
        new_chat_id = f"chat_{int(time.time())}"
        st.session_state["chat_ids"].append(new_chat_id)
        st.session_state["current_chat_id"] = new_chat_id
        st.session_state["messages"] = []
        st.session_state["selected_deleted_chat"] = None
    
    for chat_id in st.session_state["chat_ids"]:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"Conversa {chat_id}"):
                st.session_state["current_chat_id"] = chat_id
                st.session_state["messages"] = load_chat_history(chat_id)
                st.session_state["selected_deleted_chat"] = None
        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}"):
                delete_chat(chat_id)  # Chama a função delete_chat
    
    if st.button("Histórico de Chats Excluídos"):
        st.session_state["show_deleted_chats"] = not st.session_state["show_deleted_chats"]
    
    if st.session_state["show_deleted_chats"]:
        deleted_chats = load_deleted_chats()
        if deleted_chats:
            for chat_id, messages in deleted_chats.items():
                if st.button(f"Chat Excluído: {chat_id}"):
                    st.session_state["selected_deleted_chat"] = chat_id
                    st.session_state["current_chat_id"] = None
            if st.button("Excluir Tudo Permanentemente", key="delete_all_deleted_chats"):
                delete_all_deleted_chats()
                st.rerun()
        else:
            st.write("Nenhum chat excluído encontrado.")

# Área principal do chat
if st.session_state["current_chat_id"]:
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history(st.session_state["current_chat_id"])
    
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Como eu posso te ajudar?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            full_response = ""

            if "impressora" in prompt.lower() and "step" not in st.session_state:
                full_response = "Entendi que você está com problemas na impressora. Qual é o modelo da sua impressora?"
                st.session_state["step"] = 1
            elif "step" in st.session_state:
                if st.session_state["step"] == 1:
                    st.session_state["modelo_impressora"] = prompt
                    full_response = "Obrigado! Agora, por favor, descreva o problema que você está enfrentando."
                    st.session_state["step"] = 2
                elif st.session_state["step"] == 2:
                    st.session_state["problema_impressora"] = prompt
                    modelo = st.session_state["modelo_impressora"]
                    problema = st.session_state["problema_impressora"]
                    resposta = consultar_base_dados(modelo, problema)
                    full_response = f"Aqui está uma solução possível para o problema da sua impressora {modelo}:\n\n{resposta}"
                    full_response += "\n\nO problema foi resolvido? (Sim/Não)"
                    st.session_state["step"] = 3
                elif st.session_state["step"] == 3:
                    if "não" in prompt.lower():
                        modelo = st.session_state["modelo_impressora"]
                        tecnicos_disponiveis = consultar_tecnicos(modelo)
                        if not tecnicos_disponiveis.empty:
                            full_response = "Selecione um técnico disponível:\n"
                            for index, row in tecnicos_disponiveis.iterrows():
                                full_response += f"- {row['Nome']} - {row['Especialidade']} - {row['Data Disponível']}\n"
                            full_response += "\nDigite o nome do técnico que deseja agendar."
                            st.session_state["step"] = 4
                        else:
                            full_response = "Nenhum técnico disponível para o modelo informado."
                    else:
                        full_response = "Que bom que o problema foi resolvido! Se precisar de mais alguma coisa, estou à disposição."
                        del st.session_state["step"]
                        del st.session_state["modelo_impressora"]
                        del st.session_state["problema_impressora"]
                elif st.session_state["step"] == 4:
                    tecnico_nome = prompt.strip()
                    agendar_tecnico(st.session_state["modelo_impressora"], tecnico_nome)
                    full_response = f"Técnico {tecnico_nome} agendado com sucesso!"
                    del st.session_state["step"]
                    del st.session_state["modelo_impressora"]
                    del st.session_state["problema_impressora"]
            else:
                openai_client = OpenAI(api_key=OPENAI_API_KEY)
                for response in openai_client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=st.session_state["messages"],
                    stream=True,
                ):
                    full_response += response.choices[0].delta.content or ""
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_chat_history(st.session_state["current_chat_id"], st.session_state.messages)

elif st.session_state["selected_deleted_chat"]:
    deleted_chats = load_deleted_chats()
    if st.session_state["selected_deleted_chat"] in deleted_chats:
        st.subheader(f"Chat Excluído: {st.session_state['selected_deleted_chat']}")
        for message in deleted_chats[st.session_state["selected_deleted_chat"]]:
            avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.write("Chat excluído não encontrado.")

else:
    st.write("Bem-vindo ao CHAT THEC! Crie uma nova conversa para começar.")