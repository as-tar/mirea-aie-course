import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
import streamlit as st
import requests
from dotenv import load_dotenv
from src.config import SETTINGS

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

ASK_TIMEOUT = SETTINGS.get("timeouts", {}).get("ask", 180)
ADMIN_TIMEOUT = SETTINGS.get("timeouts", {}).get("admin", 120)

st.set_page_config(
    page_title="НейроСети ИИ-Помощник",
    page_icon="🤖",
    layout="wide"
)

st.sidebar.title("ООО «НейроСети»")
st.sidebar.markdown("### Корпоративный ИИ-помощник")

role = st.sidebar.radio(
    "Выберите режим работы:",
    ["Чат с ИИ-помощником", "Панель администратора"]
)

try:
    health_resp = requests.get(f"{API_URL}/health", timeout=2)
    api_ready = health_resp.status_code == 200 and health_resp.json().get("retriever_ready", False)
except Exception:
    api_ready = False

if not api_ready:
    st.sidebar.error("Бэкенд RAG не готов к работе. Запустите API-сервер.")

if role == "Чат с ИИ-помощником":
    st.title("Чат с корпоративной базой знаний")
    st.caption("Задавайте любые вопросы по внутренним правилам, ДМС, регламентам и технической поддержке.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Очистить историю чата", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources_docs" in message:
                with st.expander("Фрагменты из базы знаний"):
                    for doc in message["sources_docs"]:
                        source = doc.get("source", "Неизвестно") if isinstance(doc, dict) else doc.source
                        content = doc.get("content", "") if isinstance(doc, dict) else doc.content
                        
                        st.markdown(f"**Документ:** `{source}`")
                        st.info(content)

    if query := st.chat_input("Напишите ваш вопрос здесь..."):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            if not api_ready:
                err_text = "Извините, сейчас я не могу ответить. Сервер базы знаний отключен или ещё инициализирует индекс."
                st.error(err_text)
                st.session_state.messages.append({"role": "assistant", "content": err_text})
            else:
                with st.spinner("Ищу информацию в базе знаний и формулирую ответ..."):
                    try:
                        response = requests.post(f"{API_URL}/ask", json={"query": query}, timeout=ASK_TIMEOUT)
                        
                        if response.status_code == 200:
                            data = response.json()
                            answer = data["response"]["answer"]
                            sources = data["response"]["sources"]
                            relevant_indices = data["response"]["relevant_chunk_indices"]
                            all_docs = data["retrieved_docs"]
                            
                            st.markdown(answer)
                            
                            if sources:
                                st.markdown("**Источники:** " + ", ".join([f"`{s}`" for s in sources]))
                            
                            relevant_docs = []
                            for idx in relevant_indices:
                                if idx < len(all_docs):
                                    relevant_docs.append(all_docs[idx])
                            
                            if relevant_docs:
                                with st.expander("Фрагменты из базы знаний"):
                                    for doc in relevant_docs:
                                        st.markdown(f"**Документ:** `{doc['source']}`")
                                        st.info(doc["content"])
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer,
                                "sources_docs": relevant_docs
                            })
                            
                        else:
                            err_msg = f"Ошибка сервера (Код {response.status_code}). Попробуйте позже."
                            st.error(err_msg)
                            st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    except Exception as e:
                        err_msg = f"Не удалось связаться с сервером API: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})

else:
    st.title("Панель администратора базы знаний")
    st.caption("Загружайте новые регламенты, файлы или запускайте переиндексацию поиска.")
    
    if not ADMIN_PASSWORD:
        st.error("Доступ к панели администратора заблокирован: пароль не настроен в переменной окружения ADMIN_PASSWORD.")
    else:
        password = st.text_input("Введите пароль администратора:", type="password")
        
        if password == ADMIN_PASSWORD:
            st.success("Авторизация успешна!")
            
            st.markdown("### Загрузить новый документ")
            uploaded_file = st.file_uploader(
                "Выберите файл для добавления в базу знаний (Разрешены: PDF, DOCX, XLSX, TXT, MD):",
                type=["pdf", "docx", "xlsx", "txt", "md"]
            )
            
            if uploaded_file is not None:
                if st.button("Сохранить документ в базу знаний"):
                    with st.spinner("Загрузка файла на сервер..."):
                        try:
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            upload_resp = requests.post(f"{API_URL}/admin/upload", files=files, timeout=ADMIN_TIMEOUT)
                            
                            if upload_resp.status_code == 200:
                                st.success(f"Файл {uploaded_file.name} успешно сохранён на сервере.")
                                st.rerun()
                            else:
                                st.error(f"Ошибка загрузки: {upload_resp.json().get('detail')}")
                        except Exception as e:
                            st.error(f"Ошибка соединения с API: {e}")
                            
            st.markdown("---")
            
            st.markdown("### Управление существующими документами")
            try:
                doc_resp = requests.get(f"{API_URL}/admin/documents", timeout=ADMIN_TIMEOUT)
                if doc_resp.status_code == 200:
                    documents = doc_resp.json().get("documents", [])
                    
                    if not documents:
                        st.info("База знаний пуста. Загрузите файлы выше.")
                    else:
                        for doc_name in documents:
                            col_name, col_btn = st.columns([5, 1])
                            col_name.write(f"**{doc_name}**")
                            
                            if col_btn.button("Удалить", key=f"del_{doc_name}"):
                                with st.spinner(f"Удаление {doc_name}..."):
                                    del_resp = requests.delete(f"{API_URL}/admin/documents/{doc_name}", timeout=ADMIN_TIMEOUT)
                                    if del_resp.status_code == 200:
                                        st.success(f"Документ {doc_name} удален.")
                                        st.rerun()
                                    else:
                                        st.error(f"Ошибка удаления: {del_resp.json().get('detail')}")
                else:
                    st.error("Не удалось получить список документов с сервера.")
            except Exception as e:
                st.error(f"Ошибка соединения при получении списка документов: {e}")
                
            st.markdown("---")
            
            st.markdown("### Обновить поисковый индекс")
            st.write("Нажмите кнопку ниже, чтобы запустить фоновый процесс конвертации, чанкинга и пересчёта индекса BM25.")
            
            if st.button("Переиндексировать базу знаний"):
                with st.spinner("Запуск процесса переиндексации..."):
                    try:
                        reindex_resp = requests.post(f"{API_URL}/admin/reindex", timeout=ADMIN_TIMEOUT)
                        if reindex_resp.status_code == 202 or reindex_resp.status_code == 200:
                            st.success("Процесс успешно запущен в фоновом режиме на сервере!")
                        else:
                            st.error(f"Ошибка запуска: {reindex_resp.json().get('detail')}")
                    except Exception as e:
                        st.error(f"Ошибка соединения с API: {e}")
                        
        elif password != "":
            st.error("Неверный пароль администратора!")