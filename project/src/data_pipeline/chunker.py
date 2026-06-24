from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import SETTINGS

from langchain_experimental.text_splitter import SemanticChunker
from src.retriever.vector import get_embedding_model

def split_markdown_documents(converted_files: list[tuple]) -> list[Document]:
    sem_type = SETTINGS["chunking"]["semantic"].get("breakpoint_threshold_type", "percentile")
    sem_amount = SETTINGS["chunking"]["semantic"].get("breakpoint_threshold_amount", 95.0)

    safety_size = SETTINGS["chunking"]["safety"].get("chunk_size", 1000)
    safety_overlap = SETTINGS["chunking"]["safety"].get("chunk_overlap", 150)
    
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )
    
    embedder = get_embedding_model()
    
    russian_sentence_regex = (
        r"(?<="
        r"(?<!\bг)(?<!\bд)"
        r"(?<!\bул)(?<!\bоф)(?<!\bим)"
        r"(?<!\bруб)(?<!\bкоп)(?<!\bстр)"
        r"(?<!\bкорп)"
        r"(?<!\b\d)(?<!\b\d\d)"
        r"[.?!]"
        r")"
        r"\s+"
        r"(?=[А-ЯЁA-Z])"
    )
    
    semantic_splitter = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type=sem_type,
        breakpoint_threshold_amount=sem_amount,
        sentence_split_regex=russian_sentence_regex
    )
    
    safety_splitter = RecursiveCharacterTextSplitter(
        chunk_size=safety_size,
        chunk_overlap=safety_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )
    
    all_chunks = []
    
    for md_path, original_filename in converted_files:
        
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        header_splits = header_splitter.split_text(text)
        
        for doc in header_splits:
            metadata = doc.metadata.copy()
            metadata["source"] = original_filename
            
            semantic_sub_chunks = semantic_splitter.split_text(doc.page_content)
            
            for sub_text in semantic_sub_chunks:
                if len(sub_text) > 1200:
                    final_texts = safety_splitter.split_text(sub_text)
                else:
                    final_texts = [sub_text]
                
                for text_to_save in final_texts:
                    if text_to_save.strip():
                        all_chunks.append(Document(page_content=text_to_save, metadata=metadata))
                
    return all_chunks