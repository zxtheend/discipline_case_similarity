# zxtheend GitHub 项目汇总

本文件汇总了 4 个主要仓库的概览，包括 `discipline_case_similarity`、`barcode`、`dify-case-extraction` 和 `asr_online`，方便整体查看。

## 1. discipline_case_similarity
[GitHub 链接](https://github.com/zxtheend/discipline_case_similarity)
- 默认分支: main
- 仓库大小: 182 KB
- 权限: admin / maintain / pull / push / triage

### 项目简介
山西纪委信访案件智能检索系统 V1，后端服务为主，提供历史相似案件检索、重复件判断、新线索挖掘。

### 功能与接口
- POST /api/v1/identify
- POST /api/v1/clues
- POST /api/v1/admin/sync/full
- POST /api/v1/admin/sync/rebuild-row
- GET /health, /ready, /ready/sync

### 系统架构
- 数据链路: 达梦 -> StreamSets -> MySQL -> Qdrant -> Identify API
- 识别链路: 提交案件 -> 过滤 -> dense/sparse -> RRF -> rerank -> 返回结果
- 同步链路: 读取 MySQL -> 解密标准化 -> embedding -> upsert Qdrant

### 部署
```bash
docker compose up -d --build
```

## 2. barcode
[GitHub 链接](https://github.com/zxtheend/barcode)
- 默认分支: main
- 仓库大小: 17 KB

### 项目简介
基于 FastAPI + PyMuPDF + zxing-cpp 的轻量服务，识别 PDF 文件条形码/二维码。

### 功能
- 上传 PDF 并逐页渲染识别
- 返回码制、内容、页码、方向、定位坐标
- 支持 Docker 部署和离线环境

### 部署
```bash
docker build -t pdf-barcode-api:local .
docker run -d --name pdf-barcode-api -p 8000:8000 pdf-barcode-api:local
``` 
或使用 docker-compose

## 3. dify-case-extraction
[GitHub 链接](https://github.com/zxtheend/dify-case-extraction)
- 默认分支: main
- 仓库大小: 24 KB

### 项目简介
Dify DSL 文件，用于纪检监察信访场景案件信息提取，包含 `案件信息提取.yml` 工作流。

### 使用方式
1. 在 Dify 导入 DSL 文件
2. 配置模型服务
3. 兼容 Qwen3.5-9B 模型模板

## 4. asr_online
[GitHub 链接](https://github.com/zxtheend/asr_online)
- 默认分支: main
- 仓库大小: 185 KB

### 项目简介
实时语音转写系统，浏览器采集音频，FastAPI 控制会话，FunASR Runtime 识别。

### 系统特点
- 单机内网部署
- 默认 2pass 模式
- WebSocket 实时通信
- 健康检查接口
- 离线部署支持

### 部署
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```
或使用 Docker Compose
```bash
docker compose up -d --build
```