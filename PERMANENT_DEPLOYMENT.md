# AIsthma Forge - 永久部署指南

## 🌐 永久部署方案

本文档提供三种永久部署 AIsthma Forge 的方案,从最简单到最灵活排序。

---

## 方案 1: Streamlit Community Cloud (推荐 ⭐)

### 优势
- ✅ **完全免费**
- ✅ **零配置部署** - 几分钟内上线
- ✅ **自动 HTTPS** 和 SSL 证书
- ✅ **自动更新** - Git push 即部署
- ✅ **无需服务器管理**
- ✅ **公共访问** URL (例如: `https://aisthma-forge.streamlit.app`)

### 部署步骤

#### 1. 准备 GitHub 仓库

```bash
# 初始化 Git 仓库
cd /home/ubuntu/aisthma_forge
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: AIsthma Forge v1.0"

# 创建 GitHub 仓库 (在 GitHub 网站上操作)
# 然后关联远程仓库
git remote add origin https://github.com/YOUR_USERNAME/aisthma-forge.git

# 推送代码
git branch -M main
git push -u origin main
```

#### 2. 部署到 Streamlit Cloud

1. 访问 **https://share.streamlit.io**
2. 使用 GitHub 账号登录
3. 点击 **"New app"**
4. 选择配置:
   - **Repository**: `YOUR_USERNAME/aisthma-forge`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. 点击 **"Deploy"**

#### 3. 等待部署完成 (约 2-5 分钟)

部署完成后,您将获得永久 URL:
```
https://aisthma-forge.streamlit.app
```

### 配置说明

Streamlit Cloud 会自动识别:
- `requirements.txt` - Python 依赖
- `packages.txt` - 系统依赖
- `.streamlit/config.toml` - 应用配置

### 限制
- 资源限制: 1 CPU, 800MB RAM (免费版)
- 适合中小型数据集 (<500 samples)
- 大数据集建议使用方案 2 或 3

---

## 方案 2: Hugging Face Spaces

### 优势
- ✅ **免费**
- ✅ **更高资源配额** (2 CPU, 16GB RAM)
- ✅ **GPU 支持** (付费)
- ✅ **易于分享**
- ✅ **社区可见性**

### 部署步骤

#### 1. 创建 Hugging Face Space

1. 访问 **https://huggingface.co/spaces**
2. 点击 **"Create new Space"**
3. 选择:
   - **Space name**: `aisthma-forge`
   - **SDK**: `Streamlit`
   - **Visibility**: `Public` 或 `Private`

#### 2. 推送代码

```bash
# 克隆 Space 仓库
git clone https://huggingface.co/spaces/YOUR_USERNAME/aisthma-forge
cd aisthma-forge

# 复制应用文件
cp -r /home/ubuntu/aisthma_forge/* .

# 提交并推送
git add .
git commit -m "Deploy AIsthma Forge"
git push
```

#### 3. 访问应用

URL: `https://huggingface.co/spaces/YOUR_USERNAME/aisthma-forge`

### 配置文件

需要创建 `README.md` (Space 配置):

```yaml
---
title: AIsthma Forge
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# AIsthma Forge

Microbiome Analysis Platform for Asthma Research
```

---

## 方案 3: 自托管云服务器

### 适用场景
- 需要完全控制
- 处理大型数据集
- 企业级部署
- 自定义域名

### 3.1 使用 Railway (最简单的自托管)

#### 优势
- ✅ 免费额度 ($5/月)
- ✅ 自动 HTTPS
- ✅ 从 GitHub 自动部署
- ✅ 简单的环境变量管理

#### 部署步骤

1. 访问 **https://railway.app**
2. 连接 GitHub 账号
3. 点击 **"New Project"** → **"Deploy from GitHub repo"**
4. 选择 `aisthma-forge` 仓库
5. Railway 自动检测 Python 应用并部署

#### 配置

在 Railway 设置中添加:
- **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- **Environment Variables**: (如需要)

### 3.2 使用 Render

#### 优势
- ✅ 免费层可用
- ✅ 自动 SSL
- ✅ 持续部署

#### 部署步骤

1. 访问 **https://render.com**
2. 创建 **"New Web Service"**
3. 连接 GitHub 仓库
4. 配置:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. 点击 **"Create Web Service"**

### 3.3 使用 Google Cloud Run

#### 优势
- ✅ 按使用付费
- ✅ 自动扩展
- ✅ 高性能

#### 部署步骤

1. 创建 `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0
```

2. 部署:

```bash
# 构建并推送镜像
gcloud builds submit --tag gcr.io/PROJECT_ID/aisthma-forge

# 部署到 Cloud Run
gcloud run deploy aisthma-forge \
  --image gcr.io/PROJECT_ID/aisthma-forge \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi
```

---

## 推荐方案对比

| 方案 | 成本 | 难度 | 资源 | 适用场景 |
|------|------|------|------|----------|
| **Streamlit Cloud** | 免费 | ⭐ 最简单 | 800MB RAM | 演示、小型研究 |
| **Hugging Face** | 免费 | ⭐⭐ 简单 | 16GB RAM | 中型研究、社区分享 |
| **Railway** | $5/月 | ⭐⭐ 简单 | 可配置 | 个人项目 |
| **Render** | 免费/付费 | ⭐⭐⭐ 中等 | 可配置 | 专业项目 |
| **Google Cloud Run** | 按用量 | ⭐⭐⭐⭐ 复杂 | 高性能 | 企业级 |

---

## 最佳实践建议

### 对于大多数用户 (推荐)
**使用 Streamlit Community Cloud**
- 最快上线 (5 分钟)
- 零成本
- 适合演示和中小型数据集

### 对于研究团队
**使用 Hugging Face Spaces**
- 更高资源配额
- 更好的社区可见性
- 易于协作

### 对于企业用户
**使用 Google Cloud Run 或 AWS**
- 完全控制
- 高性能
- 可扩展性

---

## 部署后配置

### 1. 自定义域名 (可选)

**Streamlit Cloud:**
- 升级到付费计划
- 在设置中添加自定义域名

**其他平台:**
- 在 DNS 设置中添加 CNAME 记录
- 指向平台提供的 URL

### 2. 环境变量

如果需要 API 密钥或敏感配置:

**Streamlit Cloud:**
```toml
# .streamlit/secrets.toml (不要提交到 Git)
[api_keys]
openai = "sk-..."
```

**其他平台:**
在平台的环境变量设置中添加

### 3. 监控和日志

**Streamlit Cloud:**
- 内置日志查看器
- 应用状态监控

**自托管:**
- 配置日志聚合 (如 Sentry)
- 设置性能监控

---

## 故障排除

### 部署失败

**检查:**
1. `requirements.txt` 是否完整
2. Python 版本兼容性 (使用 3.11)
3. 依赖包冲突

**解决:**
```bash
# 测试本地构建
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 内存不足

**Streamlit Cloud:**
- 减少数据集大小
- 应用更严格的过滤
- 升级到付费计划

**自托管:**
- 增加内存配置
- 使用更大的实例

### 应用缓慢

**优化:**
- 使用 `@st.cache_data` 缓存计算
- 减少实时计算
- 优化数据加载

---

## 维护和更新

### 更新应用

```bash
# 本地修改代码
cd /home/ubuntu/aisthma_forge

# 提交更改
git add .
git commit -m "Update: description of changes"
git push

# Streamlit Cloud 和 Hugging Face 会自动重新部署
```

### 版本管理

```bash
# 创建版本标签
git tag -a v1.1 -m "Version 1.1: Added new features"
git push origin v1.1
```

---

## 安全建议

### 1. 数据隐私
- 不要在公共仓库中包含敏感数据
- 使用 `.gitignore` 排除数据文件
- 考虑私有部署用于敏感研究

### 2. 访问控制
- Streamlit Cloud 支持密码保护 (付费)
- 自托管可配置认证层

### 3. HTTPS
- 所有推荐平台都提供自动 HTTPS
- 确保不禁用 SSL

---

## 成本估算

### 免费方案
- **Streamlit Cloud**: $0/月 (有限资源)
- **Hugging Face**: $0/月 (更好资源)
- **Render**: $0/月 (有限资源)

### 付费方案
- **Streamlit Cloud Pro**: $20/月 (更多资源)
- **Railway**: ~$5-20/月 (按使用)
- **Google Cloud Run**: ~$10-50/月 (按使用)
- **AWS EC2**: ~$20-100/月 (固定实例)

---

## 立即开始部署!

### 快速部署命令 (Streamlit Cloud)

```bash
# 1. 初始化 Git
cd /home/ubuntu/aisthma_forge
git init
git add .
git commit -m "Initial commit"

# 2. 创建 GitHub 仓库 (在网页上)
# https://github.com/new

# 3. 推送代码
git remote add origin https://github.com/YOUR_USERNAME/aisthma-forge.git
git branch -M main
git push -u origin main

# 4. 访问 Streamlit Cloud 部署
# https://share.streamlit.io
```

### 需要帮助?

- 📖 查看 Streamlit 文档: https://docs.streamlit.io/streamlit-community-cloud
- 💬 加入 Streamlit 社区: https://discuss.streamlit.io
- 🐛 报告问题: GitHub Issues

---

🫁 **准备好永久部署 AIsthma Forge,让全球研究者受益!**
