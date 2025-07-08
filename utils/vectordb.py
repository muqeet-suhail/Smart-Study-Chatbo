from utils import libraries

def save_chunks_to_faiss(new_chunks, index_path, texts_path):
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in new_chunks]
    model = libraries.SentenceTransformer("all-MiniLM-L6-v2")
    new_embeddings = model.encode(texts)

    # Load or initialize index and stored chunks
    if libraries.os.path.exists(index_path) and libraries.os.path.exists(texts_path):
        index = libraries.faiss.read_index(index_path)
        with open(texts_path, "rb") as f:
            stored_chunks = libraries.pickle.load(f)
    else:
        index = libraries.faiss.IndexFlatL2(new_embeddings.shape[1])
        stored_chunks = []

    # Add new embeddings and extend stored chunks
    index.add(new_embeddings)
    stored_chunks.extend(new_chunks)

    # Save updated index and chunks
    libraries.faiss.write_index(index, index_path)
    with open(texts_path, "wb") as f:
        libraries.pickle.dump(stored_chunks, f)

