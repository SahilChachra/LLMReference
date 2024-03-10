import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from ctransformers import AutoModelForCausalLM

DB_FAISS_PATH = "vectorstore/db_faiss"


def load_llm():
	llm = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id="./llama-2-7b-chat.Q5_K_M.gguf", 
		model_type="llama", 
		gpu_layers=20,
		max_new_tokens=512,
		temperature=0.1)
	print("Model loaded")
	print(llm("AI is going to"))


load_llm()
