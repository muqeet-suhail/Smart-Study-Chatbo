# 📁 File handling and PDF parsing
import fitz                 # PyMuPDF — to extract text from PDF documents
import docx                 # For reading Microsoft Word (.docx) files
import time               # To handle delays in the UI (e.g., for user feedback)

# 🧠 Text embedding and similarity search
from sentence_transformers import SentenceTransformer  # To convert text into dense embeddings for semantic search

# 🔍 Vector store (indexing + searching)
import faiss               # Facebook AI Similarity Search — fast similarity search for embeddings
import pickle              # For serializing and deserializing Python objects (e.g., vector indices)
# 📊 Numerical computations
import numpy as np         # To handle embedding arrays and perform vector operations

# 🤖 Gemini API (text generation)
import google.generativeai as genai  # Google's SDK to connect with Gemini Pro API for content generation

# 🔐 Secure environment configuration
from dotenv import load_dotenv       # Loads environment variables (e.g., API keys) from a .env file
import os                            # For path handling and accessing environment variables

# ✂️ Text cleaning and preprocessing
import re                            # Regular expressions — used for cleaning and splitting text
import nltk                          # Natural Language Toolkit — used for tokenization and linguistic processing
import spacy                         # Advanced NLP library for parsing and tokenization

# 🌐 UI creation
import streamlit as st               # Streamlit — to build a web-based user interface
from streamlit_chat import message  # Adds a chatbot-like chat UI in Streamlit

# 📂 File handling
import tempfile                      # Creates temporary files safely for file uploads

# 🧠 Langchain (optional OpenAI integration)
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# 🔑 Misc utilities
import uuid                          # For unique message keys
