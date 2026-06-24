import re
from markitdown import MarkItDown
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def normalize_headers(text: str) -> str:
    text = re.sub(
        r'^(Раздел\s+\d+\.\s+.+)$', 
        r'## \1', 
        text, 
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    text = re.sub(
        r'^(\d+\.\s+[А-ЯA-Z].{2,80})$', 
        r'## \1', 
        text, 
        flags=re.MULTILINE
    )
    return text

def convert_raw_to_markdown() -> list[tuple]:
    for file in PROCESSED_DATA_DIR.iterdir():
        if file.is_file():
            file.unlink()

    md_converter = MarkItDown()
    valid_extensions = {'.docx', '.pdf', '.xlsx', '.txt', '.md'}
    
    raw_files = [f for f in RAW_DATA_DIR.iterdir() if f.suffix.lower() in valid_extensions]
    converted_files = []
    
    for file_path in raw_files:
        try:
            result = md_converter.convert(str(file_path))
            markdown_content = result.text_content
            
            normalized_content = normalize_headers(markdown_content)
            
            output_path = PROCESSED_DATA_DIR / f"{file_path.stem}.md"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(normalized_content)
                
            converted_files.append((output_path, file_path.name))
        except Exception as e:
            print(f"Ошибка при обработке {file_path.name}: {e}")
            
    return converted_files