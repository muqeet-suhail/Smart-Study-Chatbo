from utils.libraries import *

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def clean_and_chunk_text(text, max_length=500, filename="Unknown Document"):
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_length):
        chunk_text = " ".join(words[i:i+max_length])
        chunks.append({
            "text": chunk_text,
            "source": filename
        })

    return chunks
