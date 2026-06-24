from src.config import PROJECT_ROOT, SETTINGS, RAW_DATA_DIR

import os
import shutil
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor


LOG_FILE = PROJECT_ROOT / "artifacts" / "api.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("api")

from src.data_pipeline.converter import convert_raw_to_markdown
from src.data_pipeline.chunker import split_markdown_documents
from src.retriever.bm25 import create_bm25_retriever
from src.retriever.ensemble import create_reranked_retriever
from src.pipelines.rag_chain import run_rag_pipeline, RAGResponse

class QueryRequest(BaseModel):
    query: str = Field(..., description="Вопрос сотрудника к базе знаний")

class DocumentInfo(BaseModel):
    source: str
    content: str

class AskResponse(BaseModel):
    response: RAGResponse = Field(..., description="Структурированный ответ от LLM")
    retrieved_docs: list[DocumentInfo] = Field(..., description="Чанки, реально использованные для ответа")

def initialize_retriever_pipeline(app: FastAPI):
    logger.info("Запуск инициализации поискового пайплайна...")
    try:
        converted = convert_raw_to_markdown()
        if not converted:
            raise ValueError("Папка data/raw пуста.")
        chunks = split_markdown_documents(converted)
        k_bm25 = SETTINGS["retrievers"]["k_bm25"]
        base_bm25 = create_bm25_retriever(chunks, k=k_bm25)
        
        app.state.retriever = create_reranked_retriever(base_bm25)
        logger.info("Поисковый пайплайн успешно инициализирован и готов к работе!")
    except Exception as e:
        logger.error(f"Ошибка инициализации пайплайна: {e}", exc_info=True)
        app.state.retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = None 
    
    logger.info("Запуск веб-сервера FastAPI...")
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
    try:
        tracer_provider = register(endpoint=phoenix_endpoint)
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info(f"Трассировка LangChain активирована (Отправка на {phoenix_endpoint}).")
    except Exception as e:
        logger.warning(f"Ошибка подключения к Phoenix (Трассировка отключена): {e}")
    
    initialize_retriever_pipeline(app)
    
    yield
    
    logger.info("Остановка веб-сервера FastAPI...")
    try:
        LangChainInstrumentor().uninstrument()
    except Exception:
        pass

app = FastAPI(
    title="NeuroNetworks RAG API",
    description="Корпоративный ИИ-помощник ООО «НейроСети» по локальной базе знаний.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", tags=["System"])
def health_check():
    retriever = getattr(app.state, "retriever", None)
    is_ready = retriever is not None
    
    logger.info(f"Получен запрос /health. Готовность: {is_ready}")
    return {
        "status": "ok" if is_ready else "error",
        "retriever_ready": is_ready,
        "active_pipeline": "BM25 + Reranker"
    }

@app.post("/ask", response_model=AskResponse, tags=["Employee"])
def ask_question(request: QueryRequest):
    logger.info(f"Получен запрос /ask от сотрудника. Длина: {len(request.query)} символов.")

    retriever = getattr(app.state, "retriever", None)
    
    if not retriever:
        logger.warning("Попытка вызова /ask при неинициализированном ретривере.")
        raise HTTPException(status_code=503, detail="Поисковый индекс не готов.")
        
    try:
        result = run_rag_pipeline(request.query, retriever)
        
        retrieved_docs = [
            DocumentInfo(source=doc.metadata.get("source", "Неизвестно"), content=doc.page_content)
            for doc in result["docs"]
        ]
        
        logger.info(f"Ответ сгенерирован успешно. Извлечено источников: {len(retrieved_docs)}.")
        return AskResponse(
            response=result["response"],
            retrieved_docs=retrieved_docs
        )
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа. Ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при генерации ответа.")

@app.get("/admin/documents", tags=["Admin"])
def list_documents():
    logger.info("Запрос списка документов администратором...")
    try:
        files = [f.name for f in RAW_DATA_DIR.iterdir() if f.is_file()]
        return {"documents": files}
    except Exception as e:
        logger.error(f"Ошибка получения списка документов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось получить список документов.")
    
@app.delete("/admin/documents/{filename}", tags=["Admin"])
def delete_document(filename: str):
    logger.info(f"Запрос на удаление файла: {filename}...")
    file_path = RAW_DATA_DIR / filename
    
    if not file_path.exists():
        logger.warning(f"Попытка удаления несуществующего файла: {filename}")
        raise HTTPException(status_code=404, detail="Файл не найден.")
        
    try:
        file_path.unlink()
        logger.info(f"Файл {filename} успешно удален с диска администратором.")
        return {"status": "success", "message": f"Файл {filename} успешно удален."}
    except Exception as e:
        logger.error(f"Ошибка удаления файла {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось удалить файл.")

@app.post("/admin/upload", tags=["Admin"])
def upload_document(file: UploadFile = File(...)):
    logger.info(f"Загрузка файла администратором: {file.filename}...")
    valid_extensions = {'.docx', '.pdf', '.xlsx', '.txt', '.md'}
    file_path = Path(file.filename)
    
    if file_path.suffix.lower() in valid_extensions:
        try:
            output_path = RAW_DATA_DIR / file.filename
            with output_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Файл {file.filename} успешно сохранен на диске.")
            return {"status": "success", "message": f"Файл {file.filename} успешно загружен."}
        except Exception as e:
            logger.error(f"Ошибка сохранения файла {file.filename}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Не удалось сохранить файл.")
    else:
        logger.warning(f"Отклонена загрузка файла {file.filename} (неверный формат).")
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла.")

@app.post("/admin/reindex", tags=["Admin"])
def trigger_reindex(background_tasks: BackgroundTasks):
    logger.info("Запрос переиндексации администратором...")
    try:
        background_tasks.add_task(initialize_retriever_pipeline, app)
        return {"status": "accepted", "message": "Переиндексация базы знаний запущена в фоновом режиме."}
    except Exception as e:
        logger.error(f"Не удалось запустить фоновую переиндексацию: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ошибка запуска переиндексации.")