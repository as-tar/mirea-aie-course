import shutil
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.config import VECTOR_DB_DIR, MODELS_CACHE_DIR, SETTINGS

class DeepVKEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed_texts = [f"search_document: {t}" for t in texts]
        return super().embed_documents(prefixed_texts)

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(f"search_query: {text}")

def get_embedding_model() -> DeepVKEmbeddings:
    device = SETTINGS["retrievers"].get("device", "cpu")

    return DeepVKEmbeddings(
        model_name="deepvk/USER2-base",
        cache_folder=str(MODELS_CACHE_DIR),
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

def build_vector_store(documents: list[Document], clear_old: bool = True) -> Chroma:
    if clear_old and VECTOR_DB_DIR.exists():
        shutil.rmtree(VECTOR_DB_DIR)
        
    embedder = get_embedding_model()
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        persist_directory=str(VECTOR_DB_DIR)
    )
    return vector_store

def create_vector_retriever(vector_store: Chroma, k: int = None):
    k = k or SETTINGS["retrievers"]["k_vector"]
    return vector_store.as_retriever(search_kwargs={"k": k})