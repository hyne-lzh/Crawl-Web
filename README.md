# 搜索引擎聚合爬虫

一键对 **69+ 个搜索引擎** 同时发起搜索，自动提取、清洗、去重、排序，最终输出 JSON + CSV 结果。

## 工作流程

```
webs.json ──→ 爬取 (Playwright) ──→ 提取 (bs4 + sklearn) ──→ 清洗 (sklearn) ──→ 保存 (JSON + CSV)
```

| 阶段 | 模块 | 功能 |
|------|------|------|
| 1. 加载 | `request_webs.py` | 读取 `webs.json`，按 speed 过滤引擎 |
| 2. 爬取 | `request_webs.py` | Playwright 无头浏览器访问各搜索引擎，获取 HTML |
| 3. 提取 | `extract_data.py` | 双引擎提取：CSS 选择器 + sklearn 文本模式识别 |
| 4. 清洗 | `wash_data.py` | 广告检测、跨引擎去重、相关性排序 |
| 5. 保存 | `wash_data.py` | 终端 Rich 表格预览 + 导出 JSON/CSV |

## 项目结构

```
Crawl Web/
├── request_webs.py                    # 主入口：加载配置 → 爬取 → 串联各模块
├── extract_data.py                    # 数据提取：bs4 CSS选择器 + sklearn TF-IDF 双路提取
├── wash_data.py                       # 数据清洗：广告检测 → 去重 → 排序 → 展示 → 保存
├── webs.json                          # 搜索引擎配置文件（69 个引擎）
├── requirements.txt                   # Python 依赖
│
├── windows_setup_install_cn.py        # Windows 一键安装脚本
├── macos_setup_install_cn.py          # macOS 一键安装脚本
├── linux_setup_install_cn.py          # Linux 一键安装脚本
│
└── results/                           # 输出目录（JSON + CSV）
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- Windows / macOS / Linux

### 2. 安装依赖

根据操作系统选择对应的一键安装脚本（国内镜像加速）：

**Windows：**

```bash
python windows_setup_install_cn.py
```

**macOS：**

```bash
python macos_setup_install_cn.py
```

**Linux（Debian/Ubuntu/Fedora/Arch）：**

```bash
python linux_setup_install_cn.py
```

> 各脚本自动使用国内镜像加速 playwright 浏览器下载。

**手动安装：**

```bash
pip install -r requirements.txt
playwright install chromium
```

> Linux 用户如遇 Chromium 依赖缺失，运行：`playwright install --with-deps chromium`
>
> 国内Windows用户可设置镜像加速：`set PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright`

### 3. 运行

```bash
python request_webs.py
```

按提示输入搜索关键词和最低速度阈值（0-10，默认 5）：

```
请输入搜索关键词: Python 教程
最低速度阈值 (0-10, 默认 5): 7
```

程序将自动完成：加载配置 → 爬取 → 提取 → 清洗 → 展示 → 保存。

### 4. 输出

结果保存在 `results/` 目录下：

| 文件 | 说明 |
|------|------|
| `results_YYYYMMDD_HHMMSS.json` | 完整结构化数据 |
| `results_YYYYMMDD_HHMMSS.csv` | 表格数据，可用 Excel/WPS 打开 |

## 配置文件 `webs.json`

格式为 `{名称: {url, speed}}` 字典：

```json
{
    "百度": {
        "url": "https://www.baidu.com/s?wd={query}",
        "speed": 10
    },
    "Google": {
        "url": "https://www.google.com/search?q={query}&hl=zh-CN",
        "speed": 1
    }
}
```

| 字段 | 说明 |
|------|------|
| 名称 | 搜索引擎名称（需与 `extract_data.py` 中的选择器键名一致） |
| url | 搜索 URL，`{query}` 占位符会被替换为关键词 |
| speed | 1-10 速度评级，10 最快。运行时可设阈值过滤慢速站 |

### 当前收录的引擎（69 个）

**通用搜索：** Google、百度、必应搜索、DuckDuckGo、Yandex、Qwant、YOU、Perplexity、Brave Search、Mojeek、Ecosia、Yahoo 搜索、搜狗全网搜索、360 搜索、多吉搜索

**学术搜索：** Google 学术、Semantic Scholar、arXiv 预印本、AMiner 学术、知网（CNKI）、CNKI 研学平台、CNKI 外文文献

**开发者搜索：** 开发者搜索、Github、GitLab、Stack Overflow、HuggingFace 模型搜索

**百科知识：** 百度百科、中文维基百科、Wikipedia、WikiHow、Wolfram Alpha、Urban Dictionary、The Free Dictionary、华为 IP 知识百科、Stanford Encyclopedia of Philosophy

**图片视频：** Google 图片、百度图片、必应图片、Unsplash、Pexels、Pixabay、Flickr、CC Search、Iconfinder、FindIcons、LibreStock

**书籍文档：** Google 图书搜索、鸠摩搜书、Ebooke

**社区媒体：** 知乎搜索、头条搜索、搜狗微信搜索、B 站搜索、Reddit 搜索、IMDb 影视搜索、Bandcamp 音乐、Wayback Machine 存档检索、全历史、TinEye、Tunefind

**工具站点：** SimilarSites、SimilarWeb、visualcapitalist、ProSettings、BetaWiki、TOP 500、The Pudding

## 各模块说明

### `request_webs.py` — 爬取主控

- `WebCrawler` 类：加载配置、构建搜索 URL、管理 Playwright 浏览器
- 自动识别 `webs.json` 新旧格式（数组 / 字典）
- 反检测：自定义 User-Agent、隐藏 `navigator.webdriver`
- 引擎间延迟 1.5s 防限流
- 30s 超时 + 懒加载滚动等待

### `extract_data.py` — 双引擎提取

**路径 1 — bs4 CSS 选择器：** 为百度、Google、必应等主流引擎预定义容器/标题/链接/摘要选择器，精确提取。选择器失效时自动降级为通用提取。

**路径 2 — sklearn 文本模式：** 不依赖 CSS，用 TF-IDF 向量化所有带链接的文本块，计算与关键词的余弦相似度，过滤低分候选和噪声内容。

两路结果自动合并（相同 URL 保留摘要更丰富的那条），再经全局后过滤（URL 质量、噪声词、相关性二次检验）。

### `wash_data.py` — 数据清洗

1. **广告检测** — 规则 + sklearn LogisticRegression 双模检测，过滤广告/推广结果
2. **跨引擎去重** — TF-IDF + 余弦相似度，相似度 ≥ 85% 视为重复
3. **相关性排序** — 按与 query 的 TF-IDF 余弦相似度降序排列
4. **终端展示** — Rich 彩色表格，每个引擎独立展示
5. **保存** — JSON（完整数据）+ CSV（Excel 兼容，UTF-8 BOM）

## 自定义

### 添加新搜索引擎

在 `webs.json` 中添加条目：

```json
"新引擎名": {
    "url": "https://example.com/search?q={query}",
    "speed": 6
}
```

如果希望获得更精准的提取结果，可在 `extract_data.py` 的 `ENGINE_SELECTORS` 字典中为它添加 CSS 选择器。

### 编程调用

```python
from request_webs import WebCrawler

crawler = WebCrawler()
crawler.run(query="Python 教程", min_speed=5)
```

## 依赖

| 包 | 用途 |
|----|------|
| Playwright | 无头浏览器爬取 |
| scikit-learn | TF-IDF 向量化、广告分类、相关性排序 |
| beautifulsoup4 | HTML 解析与 CSS 选择器提取 |
| rich | 终端彩色进度条与表格渲染 |

## License

MIT
