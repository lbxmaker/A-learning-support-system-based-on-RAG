import os
import streamlit as st
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import json
import re
import logging
from typing import Dict, List

st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed"
)

from pyvis.network import Network
import streamlit.components.v1 as components
from dotenv import load_dotenv
from encoder import emb_text
from milvus_utils import get_milvus_client, get_search_results
from ask_llm import OllamaAPI, stream_llm_answer, extract_kg_from_text

load_dotenv()
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

if 'retrieved_lines_with_distances' not in st.session_state:
    st.session_state.retrieved_lines_with_distances = []

executor = ThreadPoolExecutor(max_workers=8)
milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)
ollama_client = OllamaAPI()

logging.basicConfig(
    filename='rag_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    encoding='utf-8'
)

@st.cache_resource
def warm_up_cache() -> bool:
    """系统预热"""
    try:
        with st.spinner('正在初始化...'):
            _ = emb_text("测试")
            return True
    except Exception as e:
        st.error(f"初始化失败: {str(e)}")
        return False

@st.cache_resource
def get_cached_knowledge_graph():
    """缓存知识图谱数据"""
    return []

def display_content_with_images(text: str) -> None:
    """显示包含图片的文本内容"""
    pattern = r'!\[(.*?)\]\((images/.*?)\)'
    parts = re.split(pattern, text)
    
    for i in range(0, len(parts), 3):
        if parts[i]:
            st.write(parts[i])
        if i + 2 < len(parts):
            image_path = parts[i + 2]
            if os.path.exists(image_path):
                st.image(image_path, caption=parts[i + 1])

def update_knowledge_graph(kg_data: dict):
    """更新知识图谱数据"""
    if 'kg_data' not in st.session_state:
        st.session_state.kg_data = {"entities": [], "relations": []}
    
    for entity in kg_data["entities"]:
        if entity not in st.session_state.kg_data["entities"]:
            st.session_state.kg_data["entities"].append(entity)
    
    for relation in kg_data["relations"]:
        if relation not in st.session_state.kg_data["relations"]:
            st.session_state.kg_data["relations"].append(relation)

def display_knowledge_graph():
    """显示知识图谱"""
    if 'kg_data' not in st.session_state:
        st.warning("暂无知识图谱数据")
        return
    
    try:
        entity_count = len(st.session_state.kg_data["entities"])
        relation_count = len(st.session_state.kg_data["relations"])
        
        net = Network(
            height="500px", 
            width="100%", 
            notebook=False, 
            directed=True,
            bgcolor="#ffffff",
            font_color="#000000"
        )
        
        for entity in st.session_state.kg_data["entities"]:
            if not isinstance(entity, dict):
                continue
            net.add_node(
                entity.get("id", ""),
                label=entity.get("label", "未命名实体"),
                title=f"类型: {entity.get('type', '未知类型')}",
                font={"size": 14}
            )
        
        for relation in st.session_state.kg_data["relations"]:
            if not isinstance(relation, dict):
                continue
            if relation.get("from") and relation.get("to"):
                net.add_edge(
                    relation["from"],
                    relation["to"],
                    label=relation.get("label", "未知关系"),
                    arrows="to",
                    font={"size": 12}
                )
        
        net.set_options("""
        {
            "nodes": {
                "shape": "dot",
                "size": 20,
                "font": {
                    "size": 14
                },
                "borderWidth": 2
            },
            "edges": {
                "arrows": "to",
                "width": 2,
                "font": {
                    "size": 12
                },
                "smooth": {
                    "type": "cubicBezier",
                    "forceDirection": "horizontal"
                }
            },
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 100,
                    "updateInterval": 10
                },
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.1
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "navigationButtons": false
            }
        }
        """)
        
        html_content = net.generate_html()
        components.html(html_content, height=550, scrolling=False)
        
        st.markdown(f"""
        <div style="text-align: center;">
            <p>实体数量: {entity_count} | 关系数量: {relation_count}</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"渲染知识图谱时出错: {str(e)}")

async def async_process_query(question: str, progress_bar) -> List[Dict]:
    """异步处理查询"""
    loop = asyncio.get_event_loop()
    
    progress_bar.progress(0.2, text="正在进行语义向量化...")
    query_vector = await loop.run_in_executor(executor, emb_text, question)
    
    progress_bar.progress(0.4, text="正在检索相关内容...")
    results = await loop.run_in_executor(
        executor,
        lambda: get_search_results(milvus_client, COLLECTION_NAME, query_vector, ["text"])
    )
    
    progress_bar.progress(0.6, text="检索完成，正在生成回答...")
    return results

def load_css():
    with open('static/styles.css', 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()
warm_up_cache()

header = st.container()
with header:
    cols = st.columns([1.2, 2, 1])
    with cols[0]:
        st.image("./pics/logo.png", width=200)
    with cols[1]:
        st.markdown("""
            <div style="text-align: center;">
                <h1 style="text-align: center;">三维动画智能问答系统</h1>
                <p style="text-align: center; font-size: 16px; color: gray;">基于Milvus向量数据库 + BGE嵌入模型 + Qwen2.5开源模型构建</p>
            </div>
        """, unsafe_allow_html=True)

left_col, middle_col, right_col = st.columns([1,2,1])

with left_col:
    retrieval_title = st.empty()
    retrieval_container = st.container()

with middle_col:
    with st.form("my_form"):
        question = st.text_area("请输入您的问题:")
        cols = st.columns([7, 2])
        with cols[1]:
            submitted = st.form_submit_button(
                "提交",
                use_container_width=True
            )
    progress_placeholder = st.empty()
    chat_container = st.container()

with right_col:
    graph_title = st.empty()
    graph_container = st.container()

def log_user_query(question, kg_data):
    """记录用户查询和知识图谱生成情况"""
    try:
        logging.info(f"用户问题: {question}")
        
        if kg_data and kg_data["entities"]:
            entity_count = len(kg_data["entities"])
            relation_count = len(kg_data["relations"])
            logging.info(f"知识图谱生成成功 - 实体数量: {entity_count}, 关系数量: {relation_count}")
        else:
            logging.warning("知识图谱生成失败，未提取到有效实体和关系")
    except Exception as e:
        logging.error(f"日志记录失败: {str(e)}")

if question and submitted:
    try:
        retrieval_title.markdown("<h3 style='text-align: center; font-size: 24px;'>向量检索</h3>", unsafe_allow_html=True)
        graph_title.markdown("<h3 style='text-align: center; font-size: 24px;'>知识图谱</h3>", unsafe_allow_html=True)
        
        progress_bar = progress_placeholder.progress(0, text="开始处理查询...")
        
        with graph_container:
            phase1 = st.info("等待回答完成...")
            
            results = asyncio.run(async_process_query(question, progress_bar))
            retrieved_lines_with_distances = [
                (res["entity"]["text"], res["distance"]) 
                for res in results[0]
            ]
            
            with retrieval_container:
                for idx, (text, distance) in enumerate(retrieved_lines_with_distances, 1):
                    st.markdown("---")
                    st.markdown(f"**结果 {idx}:**")
                    display_content_with_images(text)
                    st.markdown(f"*相似度: {1-distance:.4f}*")
            
            with chat_container:
                st.chat_message("user").write(question)
                assistant_msg = st.chat_message("assistant")
                message_placeholder = assistant_msg.empty()
                
                context = "\n\n".join([
                    f"相关内容 {i+1}：\n{line[0]}" 
                    for i, line in enumerate(retrieved_lines_with_distances)
                ])
                
                full_response = ""
                for line in stream_llm_answer(ollama_client, context, question):
                    try:
                        response_data = json.loads(line)
                        if "message" in response_data:
                            content = response_data["message"]["content"]
                            if content:
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")
                    except Exception as e:
                        continue
                
                message_placeholder.markdown(full_response)

            progress_bar.progress(0.8, text="回答完成，正在生成知识图谱...")
            phase1.empty()
            
            phase2 = st.info("图谱生成中...")
            
            kg_data = extract_kg_from_text(ollama_client, full_response)
            update_knowledge_graph(kg_data)

            log_user_query(question, kg_data)
            phase2.empty()
            
            if kg_data["entities"]:
                st.success("图谱生成完成")
                display_knowledge_graph()
            else:
                st.warning("图谱生成失败，未提取到有效实体和关系")
            st.empty()
        
        progress_bar.progress(1.0, text="处理完成")
        time.sleep(0.5)
        progress_placeholder.empty()
        
    except Exception as e:
        logging.error(f"处理查询时出错: {str(e)}")
        progress_placeholder.empty()
        st.error(f"处理查询时出错: {str(e)}")
