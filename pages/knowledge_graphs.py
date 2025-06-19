import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
import json
from typing import Dict, List, Optional, Tuple, Set
import os

# 常量定义
COLORS = {
    "课程": "#9467bd",
    "部分": "#ff7f0e",
    "章节": "#1f77b4",
    "小节": "#2ca02c",
    "知识点": "#d62728"
}

DEFAULT_EDGE_STYLE = {
    "color": "#cccccc",
    "highlight": "#ff0000",
    "width": 1,
    "physics": True,
    "length": 200
}

HIGHLIGHTED_EDGE_STYLE = {
    "color": "#ff0000",
    "highlight": "#ff0000",
    "width": 3
}

class KnowledgeGraph:
    def __init__(self):
        self.initialize_page()
        
    @staticmethod
    def initialize_page():
        st.set_page_config(layout="wide", page_title="knowledge_graphs")
        st.markdown("""
            <h3 style="text-align: center;">三维动画课程知识图谱</h3>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center; margin-bottom: 20px;">
                <span style="color: #9467bd; font-weight: bold;">■</span> 课程 &nbsp;&nbsp;
                <span style="color: #ff7f0e; font-weight: bold;">■</span> 部分 &nbsp;&nbsp;
                <span style="color: #1f77b4; font-weight: bold;">■</span> 章节 &nbsp;&nbsp;
                <span style="color: #2ca02c; font-weight: bold;">■</span> 小节 &nbsp;&nbsp;
                <span style="color: #d62728; font-weight: bold;">■</span> 知识点
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def load_content() -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        index_path = os.path.join(parent_dir, "index.md")
        wz_path = os.path.join(parent_dir, "wz.md")
        kg_path = os.path.join(os.path.dirname(__file__), "kg_data.json")
        
        try:
            if os.path.exists(kg_path):
                with open(kg_path, "r", encoding="utf-8") as f:
                    return None, None, json.load(f)
            
            with open(index_path, "r", encoding="utf-8") as f:
                index_content = f.read()
            with open(wz_path, "r", encoding="utf-8") as f:
                wz_content = f.read()
            return index_content, wz_content, None
            
        except Exception as e:
            st.error(f"文件读取错误: {str(e)}")
            return None, None, None

    @staticmethod
    def parse_index(index_content: str) -> Dict:
        sections = []
        current_part = None
        current_chapter = None
        
        for line in index_content.split('\n'):
            if line.startswith('# '):
                if current_part:
                    sections.append(current_part)
                current_part = {"title": line.replace('# ', '').strip(), "chapters": []}
            
            elif line.startswith('## '):
                if current_chapter:
                    current_part["chapters"].append(current_chapter)
                current_chapter = {"title": line.replace('## ', '').strip(), "sections": []}
            
            elif line.startswith('### '):
                section = line.replace('### ', '').strip()
                current_chapter["sections"].append(section)
        
        # 处理最后的部分和章节
        if current_chapter:
            current_part["chapters"].append(current_chapter)
        if current_part:
            sections.append(current_part)
        
        return {"title": "三维动画设计原理", "parts": sections}

    @staticmethod
    def build_knowledge_graph(course: Dict, wz_content: str) -> Dict:
        graph = {"entities": [], "relations": []}
        added_entities: Set[str] = set()
        
        def add_entity(entity_id: str, label: str, entity_type: str) -> None:
            if entity_id not in added_entities:
                graph["entities"].append({
                    "id": entity_id,
                    "label": label,
                    "type": entity_type
                })
                added_entities.add(entity_id)
        
        # 添加课程结构
        course_id = f"course_{course['title']}"
        add_entity(course_id, course["title"], "课程")
        
        for part in course["parts"]:
            part_id = f"part_{part['title']}"
            add_entity(part_id, part["title"], "部分")
            graph["relations"].append({"from": course_id, "to": part_id, "label": "包含"})
            
            for chapter in part["chapters"]:
                chapter_id = f"chapter_{chapter['title']}"
                add_entity(chapter_id, chapter["title"], "章节")
                graph["relations"].append({"from": part_id, "to": chapter_id, "label": "包含"})
                
                for section in chapter["sections"]:
                    section_id = f"section_{section}"
                    add_entity(section_id, section, "小节")
                    graph["relations"].append({"from": chapter_id, "to": section_id, "label": "包含"})
        
        # 添加知识点
        current_section = None
        for line in wz_content.split('\n'):
            if line.startswith('#### '):
                topic = line.replace('#### ', '').strip()
                topic_id = f"topic_{topic}"
                add_entity(topic_id, topic, "知识点")
                if current_section:
                    graph["relations"].append({"from": current_section, "to": topic_id, "label": "包含"})
            elif line.startswith('### '):
                current_section = f"section_{line.replace('### ', '').strip()}"
        
        return graph

    @staticmethod
    def find_path(graph: Dict, start: str, end: str, path: Optional[List] = None) -> Optional[List]:
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return path
        if start not in graph:
            return None
        for node in graph[start]:
            if node not in path:
                newpath = KnowledgeGraph.find_path(graph, node, end, path)
                if newpath:
                    return newpath
        return None

    @staticmethod
    def create_network() -> Network:
        net = Network(
            height="600px",
            width="100%",
            notebook=False,
            directed=True,
            bgcolor="#ffffff",
            font_color="#000000"
        )
        
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "stabilization": {
                    "enabled": true,
                    "iterations": 75,
                    "updateInterval": 10
                }
            }
        }
        """)
        
        return net

    @staticmethod
    def highlight_path(net: Network, path: List[str]) -> None:
        for i in range(len(path)-1):
            from_node = path[i]
            to_node = path[i+1]
            for edge in net.edges:
                if (edge["from"] == from_node and edge["to"] == to_node) or \
                   (edge["from"] == to_node and edge["to"] == from_node):
                    edge.update(HIGHLIGHTED_EDGE_STYLE)

    def visualize_knowledge_graph(self, kg_data: Dict) -> None:
        # 初始化session state
        if 'initialized' not in st.session_state:
            st.session_state['initialized'] = False
            st.session_state['stabilized'] = False
            st.session_state['net'] = None
            st.session_state['html_content'] = None

        # 构建图结构
        graph = {}
        for relation in kg_data["relations"]:
            if relation["from"] not in graph:
                graph[relation["from"]] = []
            graph[relation["from"]].append(relation["to"])

        # 路径查找函数
        def find_path(graph, start, end, path=None):
            if path is None:
                path = []
            path = path + [start]
            if start == end:
                return path
            if start not in graph:
                return None
            # 广度优先搜索（BFS）
            for node in graph[start]:
                if node not in path:
                    newpath = find_path(graph, node, end, path)
                    if newpath:
                        return newpath
            return None

        # 初始化网络图
        if not st.session_state['initialized']:
            try:
                net = Network(
                    height="600px",
                    width="100%",
                    notebook=False,
                    directed=True,
                    bgcolor="#ffffff",
                    font_color="#000000"
                )
                
                # 添加节点
                for entity in kg_data["entities"]:
                    net.add_node(
                        entity["id"],
                        label=entity["label"],
                        title=f"类型: {entity['type']}",
                        font={"size": 14},
                        color={
                            "课程": "#9467bd",
                            "部分": "#ff7f0e",
                            "章节": "#1f77b4",
                            "小节": "#2ca02c",
                            "知识点": "#d62728"
                        }[entity["type"]],
                        margin=20
                    )
                
                # 添加边
                for relation in kg_data["relations"]:
                    net.add_edge(
                        relation["from"],
                        relation["to"],
                        label=relation["label"],
                        arrows="to",
                        font={"size": 12},
                        color={"color": "#cccccc", "highlight": "#ff0000"}
                    )
                
                # 设置网络选项
                net.set_options("""
                {
                    "physics": {
                        "enabled": true,
                        "stabilization": {
                            "enabled": true,
                            "iterations": 75,
                            "updateInterval": 10
                        }
                    }
                }
                """)
                
                # 将网络图存入session state
                st.session_state['net'] = net
                st.session_state['html_content'] = net.generate_html()
                st.session_state['initialized'] = True
                st.session_state['stabilized'] = True
                
            except Exception as e:
                st.error(f"初始化错误: {str(e)}")
                return

        # 创建节点选择器
        node_options = {entity["id"]: entity["label"] for entity in kg_data["entities"]}
        selected_nodes = st.multiselect(
            "选择节点（可多选）", 
            options=list(node_options.keys()), 
            format_func=lambda x: node_options[x]
        )
        
        # 查找路径并高亮
        if selected_nodes:
            net = st.session_state['net']
            for edge in net.edges:
                edge["color"] = {"color": "#cccccc", "highlight": "#ff0000"}
                edge["width"] = 1
                edge["physics"] = True
                edge["length"] = 200
            
            # 找到根节点
            root_node = next(entity["id"] for entity in kg_data["entities"] if entity["type"] == "课程")
            
            # 处理多个节点
            if len(selected_nodes) > 1:
                complete_path = []
                
                # 获取所有节点对之间的路径
                for i in range(len(selected_nodes)):
                    for j in range(i + 1, len(selected_nodes)):
                        node1 = selected_nodes[i]
                        node2 = selected_nodes[j]
                        
                        # 找到两个节点的最近公共祖先
                        path1 = find_path(graph, root_node, node1)
                        path2 = find_path(graph, root_node, node2)
                        
                        if path1 and path2:
                            # 找到最近公共祖先
                            common_ancestor = None
                            for n1 in path1:
                                if n1 in path2:
                                    common_ancestor = n1
                                    break
                            
                            if common_ancestor:
                                # 构建从公共祖先到两个节点的路径
                                idx1 = path1.index(common_ancestor)
                                idx2 = path2.index(common_ancestor)
                                path_to_node1 = path1[idx1:]
                                path_to_node2 = path2[idx2:]
                                
                                # 添加路径到完整路径中
                                complete_path.extend(path_to_node1)
                                complete_path.extend(path_to_node2)
                
                # 高亮路径
                if complete_path:
                    self.highlight_path(net, complete_path)
            # 处理单个节点
            elif len(selected_nodes) == 1:
                # 获取从根节点到选中节点的路径
                path = find_path(graph, root_node, selected_nodes[0])
                if path:
                    self.highlight_path(net, path)
            
            # 更新HTML内容
            st.session_state['html_content'] = net.generate_html()
        else:
            # 当没有选择任何节点时，重置所有边的样式
            net = st.session_state['net']
            for edge in net.edges:
                edge["color"] = {"color": "#cccccc", "highlight": "#ff0000"}
                edge["width"] = 1
                edge["physics"] = True
                edge["length"] = 200
            st.session_state['html_content'] = net.generate_html()
        
        # 显示图谱
        if st.session_state.get('stabilized', False):
            components.html(st.session_state['html_content'], height=600, scrolling=False)
        else:
            st.info("正在初始化知识图谱，请稍候...")

def main():
    kg = KnowledgeGraph()
    index_content, wz_content, kg_data = kg.load_content()
    
    if kg_data is None:
        course = kg.parse_index(index_content)
        kg_data = kg.build_knowledge_graph(course, wz_content)
        
        kg_path = os.path.join(os.path.dirname(__file__), "kg_data.json")
        with open(kg_path, "w", encoding="utf-8") as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
    
    kg.visualize_knowledge_graph(kg_data)
    
    # 显示统计信息
    entity_counts = {
        entity_type: len([e for e in kg_data["entities"] if e["type"] == entity_type])
        for entity_type in COLORS.keys()
    }
    
    st.markdown(f"""
        <div style="text-align: center; margin-top: 20px;">
            <p>课程数量: {entity_counts['课程']} | 
            部分数量: {entity_counts['部分']} | 
            章节数量: {entity_counts['章节']} | 
            小节数量: {entity_counts['小节']} | 
            知识点数量: {entity_counts['知识点']}</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
