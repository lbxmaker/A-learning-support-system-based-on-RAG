import requests
import json
import streamlit as st
import re
import logging
import time
from typing import List, Dict

class OllamaAPI:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def chat(self, messages: List[Dict], stream: bool = False) -> Dict:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": "qwen2.5",
            "messages": messages,
            "stream": stream
        }
        
        response = requests.post(url, json=payload, stream=stream)
        return response.iter_lines() if stream else response.json()

DEFAULT_MODEL = "qwen2.5"

SYSTEM_PROMPT = """
你是一个专业的文档问答助手。请遵循以下规则：
1. 仔细分析上下文中的所有信息
2. 提供详尽的答案，包含具体的例子、时间、人物和事件
3. 按时间顺序或逻辑顺序组织内容
4. 使用专业术语，但要确保通俗易懂
5. 如果上下文包含数字、日期、人名等具体信息，请务必在回答中体现
6. 如果上下文中的信息不完整或缺失，请基于你的专业知识进行补充和扩展

回答要求：
1. 内容完整、准确
2. 结构清晰，分段合理
3. 语言流畅，易于理解
4. 重要信息需要突出
5. 适当融入你的专业见解
"""

KG_SYSTEM_PROMPT = """
你是一个专业的实体关系提取助手。请遵循以下规则：
1. 仔细分析文本内容，识别出所有实体和它们之间的关系
2. 实体包括：人物、地点、组织、概念、事件等
3. 关系包括：包含、属于、导致、参与、使用等
4. 返回格式必须为严格的JSON格式
5. 每个实体必须有唯一的id和label，id格式为"entity_数字"
6. 每个关系必须包含from和to，指向实体的id
7. 如果文本中没有可提取的实体和关系，返回空列表

返回格式示例：
{
    "entities": [
        {"id": "entity_1", "label": "实体1", "type": "类型"},
        {"id": "entity_2", "label": "实体2", "type": "类型"}
    ],
    "relations": [
        {"from": "entity_1", "to": "entity_2", "label": "关系描述"}
    ]
}
"""

def _prepare_messages(context: str, question: str) -> List[Dict]:
    """准备LLM对话消息"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"基于以下上下文信息回答用户问题。\n上下文信息：{context}\n用户问题：{question}"}
    ]

def get_llm_answer(client: OllamaAPI, context: str, question: str) -> str:
    try:
        messages = _prepare_messages(context, question)
        response = client.chat(messages=messages)
        return response["message"]["content"]
    except Exception as e:
        logging.error(f"Ollama API 调用失败: {str(e)}")
        raise

def stream_llm_answer(client: OllamaAPI, context: str, question: str):
    """
    流式生成LLM回答
    
    Args:
        client: OllamaAPI客户端实例
        context: 上下文信息
        question: 用户问题
    
    Yields:
        生成的回答流
    """
    try:
        messages = _prepare_messages(context, question)
        return client.chat(messages=messages, stream=True)
    except Exception as e:
        raise Exception(f"Ollama API 流式调用失败: {str(e)}")

def extract_kg_from_text(client: OllamaAPI, text: str) -> dict:
    """
    从文本中提取知识图谱数据
    
    Args:
        client: OllamaAPI客户端实例
        text: 要分析的文本
    
    Returns:
        包含实体和关系的字典
    """
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": KG_SYSTEM_PROMPT},
                {"role": "user", "content": f"请从以下文本中提取实体和关系：\n\n{text}"}
            ]
            response = client.chat(messages=messages)
            
            if not response or "message" not in response:
                raise ValueError("API返回数据格式错误")
                
            content = response["message"]["content"]
            if not content:
                raise ValueError("API返回内容为空")
            
            try:
                kg_data = json.loads(content)
            except json.JSONDecodeError:
                cleaned_content = re.sub(r'```json\s*|\s*```', '', content)
                cleaned_content = re.sub(r'(\]|\})(\s*)(\{|\[)', r'\1,\2\3', cleaned_content)
                cleaned_content = re.sub(r'("[^"]+")(\s*)([^"\s{]+)', r'\1:\3', cleaned_content)
                kg_data = json.loads(cleaned_content)
            
            if not isinstance(kg_data, dict):
                raise ValueError("解析后的数据不是字典格式")
                
            kg_data.setdefault("entities", [])
            kg_data.setdefault("relations", [])
            
            for i, entity in enumerate(kg_data["entities"]):
                if not isinstance(entity, dict):
                    continue
                entity.setdefault("id", f"entity_{i+1}")
                entity.setdefault("label", f"未命名实体_{i+1}")
                entity.setdefault("type", "未知类型")

            valid_relations = []
            for relation in kg_data["relations"]:
                if not isinstance(relation, dict):
                    continue
                if "from" in relation and "to" in relation:
                    relation.setdefault("label", "未知关系")
                    valid_relations.append(relation)
            kg_data["relations"] = valid_relations
                    
            return kg_data
            
        except Exception as e:
            logging.error(f"知识图谱提取尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                st.error(f"知识图谱提取失败: {str(e)}")
                return {"entities": [], "relations": []}

get_llm_answer.SYSTEM_PROMPT = SYSTEM_PROMPT

__all__ = ['OllamaAPI', 'get_llm_answer', 'stream_llm_answer', 'DEFAULT_MODEL']
