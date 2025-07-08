## 🤖 Smart Study Chatbot (Your Ultimate Study Companion) 🧠📚 

Welcome to *Chatgoat, your intelligent and personalized study assistant powered by **Gemini LLM* and *FAISS VectorDB*!, Chatgoat transforms your PDF documents into an interactive, AI-driven learning experience. Simply upload your study materials, and Chatgoat helps you explore, understand, and quiz yourself on the content, all in one clean, user-friendly interface.


<h3> 🚀 Key Features </h3>

1. ✨ *Upload multiple PDFs at once*, no limits on your study materials  
2. 🔍 *Advanced text extraction & chunking* using state-of-the-art NLP techniques  
3. 💾 *Efficient storage* of knowledge chunks in FAISS vector database for lightning-fast retrieval  
4. 🤖 *AI-powered Q&A*, ask any question and receive detailed, context-aware answers  
5. ❓ *Automatic generation of 5 related FAQs* along with confidence scores to deepen your understanding  
6. 📝 *Interactive quiz generation*, test your knowledge with customized 5-question quizzes on any topic  
7. 🎨 *Sleek, intuitive UI* built with Streamlit for a seamless study experience


<h3> 🛠 How It Works </h3>

1. 📄 *Upload PDFs*: load one or multiple documents at your convenience  
2. 🧠 *Text Extraction & Chunking*: the content is intelligently processed and segmented  
3. 💽 *Store in FAISS Vector DB*: indexed for efficient semantic search  
4. ❓ *Ask Questions*: the system retrieves the top 20 relevant chunks and queries Gemini LLM  
5. 💬 *Receive Detailed Responses*: along with 5 curated FAQs and confidence scores  
6. 🎯 *Generate Quizzes*: optional 5-question quizzes based on your queries to reinforce learning


<h3> 💻 Getting Started </h3>

<b>To run this project locally, follow these simple steps:</b> <br>

▶ Step 1: Open the project folder in VS Code <br>
▶ Step 2: Open the Bash terminal <br>
▶ Step 3: Install dependencies using: <br>

bash
pip install -r required-libraries.txt


▶ Step 4: Run the app: <br>

bash
python -m streamlit run Smart-Study-Rag-Chatbot.py
