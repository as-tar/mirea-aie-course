from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from src.config import SETTINGS, MODELS_CACHE_DIR

def create_hybrid_retriever(bm25_retriever, vector_retriever):
    w_bm25 = SETTINGS["retrievers"]["hybrid_weights"]["bm25"]
    w_vec = SETTINGS["retrievers"]["hybrid_weights"]["vector"]
    
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[w_bm25, w_vec]
    )

def create_reranked_retriever(base_retriever):
    device = SETTINGS["retrievers"].get("device", "cpu")

    model = HuggingFaceCrossEncoder(
        model_name="DiTy/cross-encoder-russian-msmarco",
        model_kwargs={
            "device": device,
            "cache_folder": str(MODELS_CACHE_DIR)
            }
    )
    
    compressor = CrossEncoderReranker(
        model=model, 
        top_n=SETTINGS["retrievers"]["k_final"]
    )
    
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )