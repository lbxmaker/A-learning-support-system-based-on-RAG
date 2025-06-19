from sentence_transformers import SentenceTransformer
import torch
import streamlit as st
from typing import List

@st.cache_resource
def get_embedding_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer('BAAI/bge-large-zh-v1.5', device=device)

@st.cache_resource
def get_embedding_cache() -> dict:
    return {}

def emb_text(text: str) -> List[float]:
    """编码单条文本"""
    embedding_cache = get_embedding_cache()
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        with torch.inference_mode():
            embedding = model.encode(text, normalize_embeddings=True)
            embedding_cache[text] = embedding.tolist()
            return embedding

model = get_embedding_model()
