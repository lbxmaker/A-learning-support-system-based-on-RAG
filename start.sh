#!/bin/bash
# 启动 Docker 服务
docker-compose up -d

# 等待 Milvus 完全启动
sleep 30

# 初始化数据库
python insert.py

# 启动 Streamlit 应用
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 