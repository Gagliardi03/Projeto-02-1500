from openai import OpenAI
import streamlit as st
import sqlite3
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import os
import time
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verificação de arquivos necessários
def check_initial_files():
    required_files = {
        "base_dados.txt": "Base de dados de soluções",
        "tecnicos.tsv": "Base de dados de técnicos"
    }
    
    missing_files = []
    for file, description in required_files.items():
        if not os.path.exists(file):
            missing_files.append(f"{file} ({description})")
    
    if missing_files:
        st.error(f"Arquivos necessários não encontrados: {', '.join(missing_files)}")
        return False
    return True

if not check_initial_files():
    st.stop()

# Configurações iniciais
OPENAI_API_KEY =

# Configuração do SQLite
def init_db():
    conn = sqlite3.connect("chat_thec.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            messages TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deleted_chats (
            id TEXT PRIMARY KEY,
            messages TEXT
        )
    """)
    conn.commit()
    return conn

conn = init_db()

# Configuração do Qdrant
try:
    qdrant_client = QdrantClient(host="localhost", port=6333)
    qdrant_client.get_collections()  # Testa a conexão
except Exception as e:
    st.error(f"Erro ao conectar ao Qdrant: {str(e)}")
    st.stop()

# Configuração do LangChain
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
try:
    with open("./base_dados.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    texts = get_chunks(raw_text)
    vectorstore.add_texts(texts)
except Exception as e:
    st.error(f"Erro ao carregar base de dados: {str(e)}")
    st.stop()

# Configuração do RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Função para consultar a base de dados
def consultar_base_dados(modelo, problema):
    query = f"Modelo: {modelo}, Problema: {problema}"
    return qa.run(query)

# Carregar a base de técnicos
try:
    tecnicos_df = pd.read_csv("tecnicos.tsv", sep="\t")
    if tecnicos_df.empty:
        st.error("Arquivo de técnicos está vazio!")
        st.stop()
except Exception as e:
    st.error(f"Erro ao carregar arquivo de técnicos: {str(e)}")
    st.stop()

# Funções para gerenciar técnicos agendados
def load_tecnicos_agendados():
    if "tecnicos_agendados" not in st.session_state:
        st.session_state["tecnicos_agendados"] = []
    return st.session_state["tecnicos_agendados"]

def save_tecnicos_agendados(tecnicos_agendados):
    st.session_state["tecnicos_agendados"] = tecnicos_agendados

def consultar_tecnicos(modelo_impressora):
    try:
        modelo_impressora = modelo_impressora.strip().lower()
        tecnicos_agendados = load_tecnicos_agendados()
        
        tecnicos_disponiveis = tecnicos_df[
            (tecnicos_df["Especialidade"].str.strip().str.lower().str.contains(modelo_impressora, case=False, na=False)) &
            (~tecnicos_df["Nome"].isin(tecnicos_agendados))
        ]
        
        logger.info(f"Técnicos encontrados para {modelo_impressora}: {len(tecnicos_disponiveis)}")
        return tecnicos_disponiveis
    except Exception as e:
        logger.error(f"Erro em consultar_tecnicos: {str(e)}")
        st.error("Erro ao consultar técnicos disponíveis")
        return pd.DataFrame()

def agendar_tecnico(modelo_impressora, tecnico_nome):
    try:
        tecnicos_agendados = load_tecnicos_agendados()
        
        if tecnico_nome not in tecnicos_df["Nome"].values:
            st.error(f"Técnico {tecnico_nome} não encontrado na base de dados.")
            return False
        
        if tecnico_nome in tecnicos_agendados:
            st.error(f"Técnico {tecnico_nome} já está agendado.")
            return False
        
        tecnicos_agendados.append(tecnico_nome)
        save_tecnicos_agendados(tecnicos_agendados)
        st.success(f"Técnico {tecnico_nome} agendado com sucesso!")
        return True
    except Exception as e:
        logger.error(f"Erro em agendar_tecnico: {str(e)}")
        st.error("Erro ao agendar técnico")
        return False

# Funções para gerenciar chats
def load_chat_history(chat_id):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT messages FROM chats WHERE id = ?", (chat_id,))
        result = cursor.fetchone()
        return eval(result[0]) if result else []
    except Exception as e:
        logger.error(f"Erro ao carregar histórico do chat {chat_id}: {str(e)}")
        return []

def save_chat_history(chat_id, messages):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO chats (id, messages)
            VALUES (?, ?)
        """, (chat_id, str(messages)))
        conn.commit()
    except Exception as e:
        logger.error(f"Erro ao salvar chat {chat_id}: {str(e)}")

def load_deleted_chats():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, messages FROM deleted_chats")
        return {row[0]: eval(row[1]) for row in cursor.fetchall()}
    except Exception as e:
        logger.error(f"Erro ao carregar chats excluídos: {str(e)}")
        return {}

def save_deleted_chats(chat_id, messages):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO deleted_chats (id, messages)
            VALUES (?, ?)
        """, (chat_id, str(messages)))
        conn.commit()
    except Exception as e:
        logger.error(f"Erro ao salvar chat excluído {chat_id}: {str(e)}")

def delete_chat(chat_id):
    try:
        messages = load_chat_history(chat_id)
        save_deleted_chats(chat_id, messages)
        
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
        
        if chat_id in st.session_state["chat_ids"]:
            st.session_state["chat_ids"].remove(chat_id)
        
        if st.session_state["current_chat_id"] == chat_id:
            st.session_state["current_chat_id"] = None
            st.session_state["messages"] = []
            if "step" in st.session_state:
                del st.session_state["step"]
            if "modelo_impressora" in st.session_state:
                del st.session_state["modelo_impressora"]
            if "problema_impressora" in st.session_state:
                del st.session_state["problema_impressora"]
        
        st.rerun()
    except Exception as e:
        logger.error(f"Erro ao excluir chat {chat_id}: {str(e)}")
        st.error("Erro ao excluir conversa")

def delete_all_deleted_chats():
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM deleted_chats")
        conn.commit()
        st.rerun()
    except Exception as e:
        logger.error(f"Erro ao excluir todos os chats: {str(e)}")
        st.error("Erro ao limpar histórico")

# Inicialização do estado da sessão
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

# Sidebar
with st.sidebar:
    st.header("Conversas")
    if st.button("Nova Conversa"):
        new_chat_id = f"chat_{int(time.time())}"
        st.session_state["chat_ids"].append(new_chat_id)
        st.session_state["current_chat_id"] = new_chat_id
        st.session_state["messages"] = []
        st.session_state["selected_deleted_chat"] = None
        if "step" in st.session_state:
            del st.session_state["step"]
        if "modelo_impressora" in st.session_state:
            del st.session_state["modelo_impressora"]
        if "problema_impressora" in st.session_state:
            del st.session_state["problema_impressora"]
        st.rerun()
    
    for chat_id in st.session_state["chat_ids"]:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"Conversa {chat_id}", key=f"switch_{chat_id}"):
                st.session_state["current_chat_id"] = chat_id
                st.session_state["messages"] = load_chat_history(chat_id)
                st.session_state["selected_deleted_chat"] = None
                if "step" in st.session_state:
                    del st.session_state["step"]
                if "modelo_impressora" in st.session_state:
                    del st.session_state["modelo_impressora"]
                if "problema_impressora" in st.session_state:
                    del st.session_state["problema_impressora"]
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}"):
                delete_chat(chat_id)
    
    if st.button("Histórico de Chats Excluídos"):
        st.session_state["show_deleted_chats"] = not st.session_state["show_deleted_chats"]
    
    if st.session_state["show_deleted_chats"]:
        deleted_chats = load_deleted_chats()
        if deleted_chats:
            for chat_id, messages in deleted_chats.items():
                if st.button(f"Chat Excluído: {chat_id}"):
                    st.session_state["selected_deleted_chat"] = chat_id
                    st.session_state["current_chat_id"] = None
                    st.rerun()
            if st.button("Excluir Tudo Permanentemente", key="delete_all_deleted_chats"):
                delete_all_deleted_chats()
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
                    # Normaliza a resposta removendo acentos e convertendo para minúsculas
                    resposta_normalizada = prompt.lower()
                    resposta_normalizada = resposta_normalizada.replace('ã', 'a').replace('õ', 'o')
                    
                    # Verifica se a resposta contém "nao" ou "não" em qualquer formato
                    if any(negacao in resposta_normalizada for negacao in ['nao', 'não', 'no']):
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
                    # Verifica se a resposta contém "sim" em qualquer formato
                    elif 'sim' in resposta_normalizada:
                        full_response = "Que bom que o problema foi resolvido! Se precisar de mais alguma coisa, estou à disposição."
                        del st.session_state["step"]
                        del st.session_state["modelo_impressora"]
                        del st.session_state["problema_impressora"]
                    else:
                        full_response = "Por favor, responda com Sim ou Não. O problema foi resolvido?"
                elif st.session_state["step"] == 4:
                    tecnico_nome = prompt.strip()
                    if agendar_tecnico(st.session_state["modelo_impressora"], tecnico_nome):
                        tecnico_info = tecnicos_df[tecnicos_df['Nome'] == tecnico_nome].iloc[0]
                        full_response = f"""
                        Agendamento confirmado!
                        - Técnico: {tecnico_nome}
                        - Especialidade: {tecnico_info['Especialidade']}
                        - Data: {tecnico_info['Data Disponível']}
                        - Modelo atendido: {st.session_state['modelo_impressora']}
                        """
                    else:
                        full_response = "Por favor, selecione outro técnico da lista."
                        st.session_state["step"] = 3  # Volta para seleção de técnicos
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