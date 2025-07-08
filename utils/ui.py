from utils import libraries
from utils.geminiconfig import configure_gemini, load_model, ask_gemini
from utils.pdfreading import extract_text_from_pdf, clean_and_chunk_text
from utils.vectordb import save_chunks_to_faiss

st = libraries.st
os = libraries.os
tempfile = libraries.tempfile
fitz = libraries.fitz

# ✅ Initialize Gemini
configure_gemini()
gemini_model = load_model()  # ✅ Load once globally

VECTORSTORE_PATH = "vectorstore.faiss"
TEXTS_PATH = "vectorstore_texts.pkl"

BASIC_CONVO_KEYWORDS = [
    "hello", "hi", "how are you", "thanks", "thank you", "okay", "ok", "great"
]

def main():
    st.set_page_config(page_title="Chat With Files")
    st.header("📄 Chatgoat (Your Study Buddy)")
    st.text("Upload your PDF files and ask questions about their content.")

    # Session state init
    for key, default in {
        "chat_history": [],
        "processComplete": False,
        "processed_files": [],
        "phase": "ask_question",
        "current_context": None,
        "quiz_display": "",
        "trigger_quiz": False
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    with st.sidebar:
        uploaded_files = st.file_uploader("📄 Upload your PDF files", type=['pdf'], accept_multiple_files=True)
        if st.button("🧠 Generate Quiz"):
            if st.session_state.current_context:
                with st.spinner("Generating quiz..."):
                    quiz = generate_quiz(st.session_state.current_context)
                    st.session_state.quiz_display = quiz
                    st.session_state.chat_history.append({"user": "Generate Quiz","bot": "### 🧠 Here's your generated quiz:\n\n" + quiz})
                    st.session_state.trigger_quiz = False
                    st.session_state.phase = "ask_question"
                    st.session_state.current_context = None
            else:
                st.warning("⚠️ Please ask a question first. I need some context to create a quiz.")

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files:
            all_chunks = []
            for uploaded_file in new_files:
                try:
                    file_bytes = uploaded_file.read()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    text = extract_text_from_pdf(tmp_path)
                    os.remove(tmp_path)

                    if not text.strip():
                        st.warning(f"⚠️ '{uploaded_file.name}' is empty or unreadable. Skipping.")
                        continue

                    chunks = clean_and_chunk_text(text, max_length=500, filename=uploaded_file.name)
                    all_chunks.extend(chunks)
                    st.session_state.processed_files.append(uploaded_file.name)

                except Exception as e:
                    st.error(f"❌ Failed to process '{uploaded_file.name}': {e}")
                    continue

            if all_chunks:
                st.session_state.processComplete = True
                st.info("🔄 Generating vector embeddings...")
                save_chunks_to_faiss(all_chunks, VECTORSTORE_PATH, TEXTS_PATH)
                st.success(f"✅ Embedded {len(all_chunks)} new chunks from {len(new_files)} files.")
            else:
                st.error("⚠️ No valid content found in new uploads.")
        else:
            placeholder = st.empty()
            placeholder.info("ℹ️ No new files to process. You can ask questions about the already processed documents.")
            libraries.time.sleep(2)
            placeholder.empty()


    display_chat_history()

    if st.session_state.phase == "ask_question":
        user_input = st.chat_input("💬 Ask a question about your document")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            lower_question = user_input.lower()

            if any(kw in lower_question for kw in BASIC_CONVO_KEYWORDS):
                bot_response = "Hi there! I'm here to help you with your documents whenever you're ready."
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
                st.session_state.chat_history.append({"user": user_input, "bot": bot_response})
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer, context = answer_with_gemini(user_input)
                        st.markdown(answer)
                st.session_state.chat_history.append({"user": user_input, "bot": answer})
                st.session_state.current_context = context

def answer_with_gemini(question: str):
    if not gemini_model:
        return "⚠️ Model not loaded.", None

    try:
        model = libraries.SentenceTransformer("all-MiniLM-L6-v2")
        question_embedding = model.encode([question])

        index = libraries.faiss.read_index(VECTORSTORE_PATH)
        with open(TEXTS_PATH, "rb") as f:
            stored_texts = libraries.pickle.load(f)

        top_k = min(20, len(stored_texts))
        D, I = index.search(question_embedding, k=top_k)

        matched_chunks = []
        for idx in I[0]:
            chunk = stored_texts[idx]
            if isinstance(chunk, dict) and "text" in chunk:
                matched_chunks.append(chunk["text"])
            elif isinstance(chunk, str):
                matched_chunks.append(chunk)

        combined_context = "\n\n".join(matched_chunks)

        initial_prompt = f"""
You are an expert study assistant.

Given the following context, complete these tasks:
1. Answer the user's question in detail.
2. Generate 5 relevant FAQs based on the same topic.
3. Answer each FAQ in detail.
4. If any answer includes lists (e.g. advantages, disadvantages, types, steps), format them using Markdown bullets (-).
5. Add a confidence score (out of 100) after each answer like: confidence in question's relaibility out of hundred: score.

Context:
{combined_context}

User Question:
{question}

Respond in clean Markdown format with proper line spacing and clear section headers.
"""
        return ask_gemini(gemini_model, initial_prompt), combined_context

    except Exception as e:
        return f"❌ Error while answering: {e}", None

def generate_quiz(context: str):
    quiz_prompt = f"""
You are a helpful study assistant.

Using the following context, generate a quiz with **5 multiple choice questions**.

Each question must follow this exact format:

Q1. What is the capital of France?  

A. Berlin  
B. London  
C. Paris  
D. Rome 

Answer: C

Rules:
- Each question must have a space after the question text and before the options.
- Each option must be on its own line starting with "A.", "B.", etc.
- Add two spaces at the end of each line to force line breaks.
- Add `Answer: <Correct Option Letter>` at the end of each question.

Context:
{context}
"""
    return ask_gemini(gemini_model, quiz_prompt)




def display_chat_history():
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"])
        st.markdown("---")
