import streamlit as st
import json
from typing import Dict, List
import os
import re

# 初始化
st.set_page_config(layout="wide", page_title="knowledge_point")

# 加载图谱数据
@st.cache_resource
def load_kg_data() -> Dict:
    kg_path = os.path.join(os.path.dirname(__file__), "kg_data.json")
    with open(kg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 加载原文内容
@st.cache_resource
def load_wz_content() -> str:
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    wz_path = os.path.join(parent_dir, "wz.md")
    with open(wz_path, "r", encoding="utf-8") as f:
        return f.read()

# 知识点搜索
def search_knowledge_points(kg_data: Dict, query: str) -> List[Dict]:
    return [entity for entity in kg_data["entities"] 
            if entity["type"] == "知识点" and query.lower() in entity["label"].lower()]

# 获取知识点内容
def get_knowledge_point_content(wz_content: str, point_label: str) -> str:
    pattern = rf"#### {re.escape(point_label)}(.*?)(?=\n#### |\Z)"
    match = re.search(pattern, wz_content, re.DOTALL)
    return match.group(1).strip() if match else "未找到相关内容"

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

# 主页面
def main():
    kg_data = load_kg_data()
    wz_content = load_wz_content()
    
    # 页面布局
    st.markdown("""
        <h3 style="text-align: center;">三维动画核心知识点</h3>
    """, unsafe_allow_html=True)
    
    # 左右两列
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        # 搜索栏
        search_query = st.text_input("搜索知识点", "", placeholder="输入知识点名称...")
        
        # 搜索建议
        if search_query:
            suggestions = [p["label"] for p in search_knowledge_points(kg_data, search_query)]
            if suggestions:
                st.markdown("**搜索建议：**")
                for suggestion in suggestions[:5]:
                    if st.button(suggestion, key=suggestion):
                        search_query = suggestion
    
    with right_col:
        # 显示选中的知识点
        if search_query:
            points = search_knowledge_points(kg_data, search_query)
            if points:
                point = points[0]
                # 展示内容
                with st.expander(f"知识点详情：{point['label']}", expanded=True):
                    content = get_knowledge_point_content(wz_content, point["label"])
                    display_content_with_images(content)
            else:
                st.info("未找到相关知识点")

if __name__ == "__main__":
    main()
