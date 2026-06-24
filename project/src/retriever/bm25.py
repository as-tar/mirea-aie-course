import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.config import NLTK_DIR, SETTINGS

nltk.data.path = [str(NLTK_DIR)]

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab', download_dir=str(NLTK_DIR), quiet=True)
    nltk.download('punkt', download_dir=str(NLTK_DIR), quiet=True)

_stemmer = SnowballStemmer("russian")

def russian_preprocess(text: str) -> list[str]:
    text = text.lower().replace('ё', 'е')
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = text.split()
    return [_stemmer.stem(token) for token in tokens]

def create_bm25_retriever(documents: list[Document], k: int = None) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(
        documents=documents,
        preprocess_func=russian_preprocess
    )
    retriever.k = k or SETTINGS["retrievers"]["k_bm25"]
    return retriever