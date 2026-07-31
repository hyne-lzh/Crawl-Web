# Search Engine Aggregator

Search **69+ search engines** simultaneously, automatically extract, clean, deduplicate, rank, and export results as JSON + CSV.

## Pipeline

```
webs.json ──→ Crawl (Playwright) ──→ Extract (bs4 + sklearn) ──→ Clean (sklearn) ──→ Save (JSON + CSV)
```

| Stage | Module | Function |
|-------|--------|----------|
| 1. Load | `request_webs.py` | Read `webs.json`, filter engines by speed rating |
| 2. Crawl | `request_webs.py` | Headless Chromium visits each search engine, grabs HTML |
| 3. Extract | `extract_data.py` | Dual-engine extraction: CSS selectors + sklearn text patterns |
| 4. Clean | `wash_data.py` | Ad detection, cross-engine dedup, relevance ranking |
| 5. Save | `wash_data.py` | Terminal Rich table preview + JSON/CSV export |

## Project Structure

```
Crawl Web/
├── request_webs.py                    # Main entry: load config → crawl → orchestrate modules
├── extract_data.py                    # Extraction: bs4 CSS selectors + sklearn TF-IDF dual-path
├── wash_data.py                       # Cleaning: ad detection → dedup → ranking → display → save
├── webs.json                          # Search engine configuration (69 engines)
├── requirements.txt                   # Python dependencies
│
├── windows_setup_install.py           # Windows one-click setup
├── macos_setup_install.py             # macOS one-click setup
├── linux_setup_install.py             # Linux one-click setup
│
└── results/                           # Output directory (JSON + CSV)
```

## Quick Start

### 1. Requirements

- Python 3.10+
- Windows / macOS / Linux

### 2. Install Dependencies

Choose the setup script for your operating system:

**Windows:**

```bash
python windows_setup_install.py
```

**macOS:**

```bash
python macos_setup_install.py
```

**Linux (Debian/Ubuntu/Fedora/Arch):**

```bash
python linux_setup_install.py
```

**Manual install:**

```bash
pip install -r requirements.txt
playwright install chromium
```

> On Linux, if Chromium dependency errors occur, run: `playwright install --with-deps chromium`

### 3. Run

```bash
python request_webs.py
```

Enter a search keyword and a minimum speed threshold (0-10, default 5):

```
Enter search keyword: Python tutorial
Minimum speed threshold (0-10, default 5): 7
```

The program will automatically: load config → crawl → extract → clean → display → save.

### 4. Output

Results are saved in the `results/` directory:

| File | Description |
|------|-------------|
| `results_YYYYMMDD_HHMMSS.json` | Full structured data |
| `results_YYYYMMDD_HHMMSS.csv` | Tabular data, openable in Excel |

## Configuration: `webs.json`

Format: `{name: {url, speed}}` dictionary:

```json
{
    "Google": {
        "url": "https://www.google.com/search?q={query}&hl=zh-CN",
        "speed": 1
    },
    "Baidu": {
        "url": "https://www.baidu.com/s?wd={query}",
        "speed": 10
    }
}
```

| Field | Description |
|-------|-------------|
| name | Search engine name (must match selector keys in `extract_data.py`) |
| url | Search URL with `{query}` placeholder substituted at runtime |
| speed | 1-10 speed rating, 10 = fastest. Set threshold at runtime to filter slow sites |

### Currently Included Engines (69)

**General Search:** Google, Baidu, Bing, DuckDuckGo, Yandex, Qwant, YOU, Perplexity, Brave Search, Mojeek, Ecosia, Yahoo, Sogou Web, 360 Search, DogeDoge

**Academic:** Google Scholar, Semantic Scholar, arXiv, AMiner, CNKI, CNKI Research Platform, CNKI Foreign Literature

**Developer:** Developer Search, GitHub, GitLab, Stack Overflow, HuggingFace Models

**Encyclopedias / Reference:** Baidu Baike, Chinese Wikipedia, Wikipedia, WikiHow, Wolfram Alpha, Urban Dictionary, The Free Dictionary, Huawei IP Encyclopedia, Stanford Encyclopedia of Philosophy

**Images:** Google Images, Baidu Images, Bing Images, Unsplash, Pexels, Pixabay, Flickr, CC Search, Iconfinder, FindIcons, LibreStock

**Books / Documents:** Google Books, Jiumo E-Book Search, Ebooke

**Community / Media:** Zhihu, Toutiao Search, Sogou WeChat, Bilibili, Reddit, IMDb, Bandcamp, Wayback Machine, AllHistory, TinEye, Tunefind

**Tools / Data:** SimilarSites, SimilarWeb, Visual Capitalist, ProSettings, BetaWiki, TOP 500, The Pudding

## Module Details

### `request_webs.py` — Crawl Controller

- `WebCrawler` class: loads config, builds search URLs, manages Playwright browser
- Auto-detects old/new `webs.json` format (array / dict)
- Anti-detection: custom User-Agent, hides `navigator.webdriver`
- 1.5s inter-engine delay for rate-limiting protection
- 30s timeout + lazy-load scroll waiting

### `extract_data.py` — Dual-Engine Extraction

**Path 1 — bs4 CSS selectors:** Predefined container/title/url/snippet selectors for major engines (Baidu, Google, Bing, etc.) for precise extraction. Falls back to generic extraction if selectors don't match.

**Path 2 — sklearn text patterns:** No CSS dependency. TF-IDF vectorizes all linked text blocks, computes cosine similarity against the query, filters low-scoring and noisy candidates.

Both paths are auto-merged (same URL → keep richer snippet), then go through a global post-filter (URL quality, noise words, secondary relevance check).

### `wash_data.py` — Data Cleaning

1. **Ad detection** — Rule-based + sklearn LogisticRegression dual-mode, filters sponsored/promotional results
2. **Cross-engine dedup** — TF-IDF + cosine similarity, similarity ≥ 85% is considered duplicate
3. **Relevance ranking** — Sorted by TF-IDF cosine similarity to the query, descending
4. **Terminal display** — Rich colored tables, one table per engine
5. **Save** — JSON (full data) + CSV (Excel compatible, UTF-8 BOM)

## Customization

### Adding a New Search Engine

Add an entry to `webs.json`:

```json
"New Engine": {
    "url": "https://example.com/search?q={query}",
    "speed": 6
}
```

For more accurate extraction, add CSS selectors to `ENGINE_SELECTORS` in `extract_data.py`.

### Programmatic Usage

```python
from request_webs import WebCrawler

crawler = WebCrawler()
crawler.run(query="Python tutorial", min_speed=5)
```

## Dependencies

| Package | Purpose |
|---------|---------|
| Playwright | Headless browser crawling |
| scikit-learn | TF-IDF vectorization, ad classification, relevance ranking |
| beautifulsoup4 | HTML parsing & CSS selector extraction |
| rich | Terminal colored progress bars & table rendering |

## License

MIT
