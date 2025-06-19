import os
import streamlit as st
import logging
from tqdm import tqdm
import re
from typing import List

MAX_CHUNK_SIZE = 2000

from encoder import emb_text
from milvus_utils import get_milvus_client, create_collection
from dotenv import load_dotenv

load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

def split_text(content: str) -> List[str]:
    """分割文本为块"""
    img_pattern = r'!\[.*?\]\(images/.*?\)'
    ref_pattern = r'图\d+\.\d+'
    chunks = []
    current_chunk = []
    current_length = 0
    
    lines = content.split('\n')
    for line in lines:
        if re.search(ref_pattern, line):
            img_found = False
            for j in range(len(current_chunk), min(len(lines), len(current_chunk) + 3)):
                if re.match(img_pattern, lines[j]):
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                    current_chunk.extend(lines[len(current_chunk):j+1])
                    img_found = True
                    break
            if not img_found:
                current_chunk.append(line)
        else:
            if current_length + len(line) > MAX_CHUNK_SIZE and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(line)
            current_length += len(line)
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def get_text(test_file: str) -> List[str]:
    """加载测试文档并分块"""
    try:
        with open(test_file, "r", encoding="utf-8") as file:
            content = file.read()
            return split_text(content)
    except Exception as e:
        logging.error(f"Error reading {test_file}: {e}")
        return []

milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)

test_text = "测试文本"
test_vector = emb_text(test_text)
dim = len(test_vector)
print(f"向量维度: {dim}")

if milvus_client.has_collection(COLLECTION_NAME):
    print(f"删除已存在的集合: {COLLECTION_NAME}")
    milvus_client.drop_collection(COLLECTION_NAME)

create_collection(milvus_client=milvus_client, collection_name=COLLECTION_NAME, dim=dim)
print(f"创建新的集合: {COLLECTION_NAME}, 维度: {dim}")

test_file = "wz.md"
text_chunks = get_text(test_file)
print(f"文档分块数量: {len(text_chunks)}")

data = []
count = 0
for chunk in tqdm(text_chunks, desc="创建文档向量"):
    try:
        vector = emb_text(chunk)
        data.append({"vector": vector, "text": chunk})
        count += 1
    except Exception as e:
        logging.error(f"处理文档块时出错:\n{e}")

print("成功处理的文档块数量:", count)

if data:
    mr = milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
    print("成功插入向量数据库的文档块数量:", mr["insert_count"])
else:
    print("没有数据可以插入")
