"""
extract_data.py — 数据提取模块
双引擎提取：bs4 (CSS选择器) + sklearn (文本模式识别)
防止目标网站更新 CSS 样式后选择器失效
接收 request_webs.py 的爬取结果 -> 提取 -> 交给 wash_data.py
"""

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn


class DataExtractor:
    """HTML 数据提取器 — 从各搜索引擎结果页中提取结构化数据"""

    # ============ 各搜索引擎的 CSS 选择器策略 ============
    # 每条策略: { container, title, url, snippet }
    ENGINE_SELECTORS: dict[str, dict] = {
        "百度": {
            "container": "div.result, div.c-container, div.result-op",
            "title":     "h3 a, h3[class] a",
            "url":       "h3 a",
            "snippet":   "span.content-right_8Zs40, div.c-abstract, span.c-abstract, "
                         "div.c-span-last, div.c-row",
        },
        "必应搜索": {
            "container": "li.b_algo",
            "title":     "h2 a",
            "url":       "h2 a",
            "snippet":   "div.b_caption p, p.b_lineclamp2, p.b_lineclamp4",
        },
        "Google": {
            "container": "div.g, div[data-sokoban-container], div.MjjYud",
            "title":     "h3",
            "url":       "a[jsname], div.yuRUbf a, a[data-ved]",
            "snippet":   "div.VwiC3b, span.aCOpRe, div[data-sncf], span.st",
        },
        "知乎搜索": {
            "container": "div.SearchResult-Card, div.List-item",
            "title":     "h2 a, a[data-za-detail-view-element_name='Title']",
            "url":       "a[data-za-detail-view-element_name='Title'], h2 a",
            "snippet":   "div.RichText, div.SearchItem-excerpt, div[class*='content']",
        },
        "搜狗微信搜索": {
            "container": "div.wx-news-item, div.news-box, ul.news-list li",
            "title":     "h3 a, h4 a",
            "url":       "h3 a, h4 a",
            "snippet":   "p.txt-info, div.news-txt, p[class*='desc']",
        },
        "头条搜索": {
            "container": "div.result-item, div.s-result, div.soResult",
            "title":     "h2 a, h3 a, a[class*='title']",
            "url":       "h2 a, h3 a",
            "snippet":   "div.content, p.abstract, div[class*='abstract'], "
                         "div[class*='desc']",
        },
        "开发者搜索": {
            "container": "div.result, div.search-result",
            "title":     "h3 a, h2 a",
            "url":       "h3 a, h2 a",
            "snippet":   "div.abstract, p.desc, div.summary",
        },
        "DuckDuckGo": {
            "container": "li[data-layout='organic'], article[data-testid='result']",
            "title":     "h2 a, a[data-testid='result-title-a']",
            "url":       "h2 a, a[data-testid='result-title-a']",
            "snippet":   "span[data-testid='result-snippet'], "
                         "div[data-result='snippet'], "
                         "span.line-clamp-3",
        },
        "Yandex": {
            "container": "li.serp-item",
            "title":     "h2 a, a.link_theme_outer",
            "url":       "h2 a, a.link_theme_outer",
            "snippet":   "div.text-container, span.extended-text__short",
        },
        "Qwant": {
            "container": "div[data-testid='result'], div.result",
            "title":     "a[data-testid='result-title'], h3 a",
            "url":       "a[data-testid='result-title'], h3 a",
            "snippet":   "span[data-testid='result-description'], p.desc",
        },
    }

    def __init__(self):
        """初始化提取器"""
        self.raw_data: list[dict] = []
        self.extracted_data: list[dict] = []
        self.console = Console()

    # ==================== 接收数据 ====================

    def receive(self, raw_data: list[dict]) -> None:
        """
        接收 request_webs.py 传来的原始爬取数据
        :param raw_data: [{"engine": "百度", "url": "...", "query": "...", "html": "..."}, ...]
        """
        self.raw_data = raw_data
        self.extracted_data = []

    # ==================== 基础提取工具 ====================

    @staticmethod
    def _make_soup(html: str) -> BeautifulSoup:
        """创建 BeautifulSoup 对象"""
        return BeautifulSoup(html, "html.parser")

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理文本：去多余空白、去控制字符"""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\u3000", " ")  # 全角空格
        return text.strip()

    @staticmethod
    def _resolve_url(href: str, base_url: str) -> str:
        """将相对链接转为绝对链接"""
        if not href:
            return ""
        return urljoin(base_url, href)

    def extract_text(self, html: str) -> str:
        """从 HTML 中提取页面纯文本"""
        soup = self._make_soup(html)
        # 移除 script、style 标签
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return self._clean_text(soup.get_text())

    def extract_links(self, html: str, base_url: str = "") -> list[str]:
        """提取所有链接"""
        soup = self._make_soup(html)
        links = []
        for a in soup.find_all("a", href=True):
            url = self._resolve_url(a.get("href", ""), base_url)
            if url and url.startswith(("http://", "https://")):
                links.append(url)
        return links

    def extract_titles(self, html: str) -> list[str]:
        """提取页面中所有标题 (h1-h6)"""
        soup = self._make_soup(html)
        return [self._clean_text(tag.get_text()) for tag in soup.find_all(["h1", "h2", "h3", "h4"])
                if self._clean_text(tag.get_text())]

    def extract_snippets(self, html: str) -> list[str]:
        """提取页面中所有段落/摘要文本"""
        soup = self._make_soup(html)
        snippets = []
        for tag in soup.find_all(["p", "span", "div"]):
            text = self._clean_text(tag.get_text())
            # 过滤太短或太长的文本
            if 10 < len(text) < 500:
                # 排除导航、页脚等常见干扰
                parent_classes = " ".join(tag.get("class", [])) if tag.get("class") else ""
                if any(kw in parent_classes.lower() for kw in ("nav", "footer", "header", "menu")):
                    continue
                snippets.append(text)
        return snippets

    # ==================== 搜索引擎专用提取 ====================

    def _extract_engine_results(
        self, html: str, engine_name: str, base_url: str
    ) -> list[dict]:
        """
        使用引擎专用选择器提取搜索结果
        :return: [{"title": "...", "url": "...", "snippet": "..."}, ...]
        """
        soup = self._make_soup(html)
        selectors = self.ENGINE_SELECTORS.get(engine_name)

        if not selectors:
            # 无专用选择器，使用通用提取
            return self._extract_generic_results(soup, base_url)

        containers = soup.select(selectors["container"])
        if not containers:
            # 选择器未命中，降级为通用提取
            return self._extract_generic_results(soup, base_url)

        results = []
        seen_urls = set()

        for container in containers:
            # 提取标题
            title_tag = container.select_one(selectors["title"])
            title = self._clean_text(title_tag.get_text()) if title_tag else ""

            # 提取链接
            url_tag = container.select_one(selectors["url"])
            url = ""
            if url_tag and url_tag.get("href"):
                url = self._resolve_url(url_tag["href"], base_url)

            # 提取摘要
            snippet_tag = container.select_one(selectors["snippet"])
            snippet = self._clean_text(snippet_tag.get_text()) if snippet_tag else ""

            # 至少有标题或链接才视为有效结果
            if not title and not url:
                continue

            # 去重
            if url and url in seen_urls:
                continue
            seen_urls.add(url)

            results.append({"title": title, "url": url, "snippet": snippet})

        return results

    @staticmethod
    def _extract_generic_results(soup: BeautifulSoup, base_url: str) -> list[dict]:
        """
        通用提取方法（无专用选择器时使用）
        提取页面中所有带链接的文本块
        """
        results = []
        seen_urls = set()

        # 策略1: 找包含 h3/a + 段落 的容器
        for tag in soup.find_all(["article", "section", "li", "div"]):
            a_tag = tag.find("a", href=True)
            if not a_tag:
                continue

            url = urljoin(base_url, a_tag.get("href", ""))
            if not url.startswith(("http://", "https://")):
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = DataExtractor._clean_text(a_tag.get_text())
            if not title:
                title = a_tag.get("title", "") or a_tag.get("aria-label", "")

            # 在同级或父级容器中找摘要文本
            p_tag = tag.find("p") or tag.find("span")
            snippet = DataExtractor._clean_text(p_tag.get_text()) if p_tag else ""

            # 跳过太短的（可能是页脚链接）
            if len(title) < 3:
                continue

            results.append({"title": title, "url": url, "snippet": snippet})

        # 策略2: 如果策略1没找到结果，用简单方法提取所有链接
        if not results:
            for a in soup.find_all("a", href=True):
                url = urljoin(base_url, a["href"])
                if not url.startswith(("http://", "https://")):
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                title = DataExtractor._clean_text(a.get_text())
                if len(title) < 3:
                    continue
                results.append({"title": title, "url": url, "snippet": ""})

        return results[:50]  # 限制最多50条

    # ==================== sklearn 文本模式提取 ====================

    @staticmethod
    def _find_nearby_text(a_tag: Tag, max_distance: int = 4) -> str:
        """
        查找链接附近的描述文本（不依赖 CSS 类名）
        逐层向上遍历父节点，找到足够长的非链接文本
        :param a_tag: 目标 <a> 标签
        :param max_distance: 最大向上层级
        :return: 找到的附近文本
        """
        link_text = DataExtractor._clean_text(a_tag.get_text())

        # 策略1: 在父节点中查找所有文本，排除链接自身的文本
        parent = a_tag.parent
        for _ in range(max_distance):
            if parent is None:
                break
            full_text = DataExtractor._clean_text(parent.get_text())
            # 移除链接文本后看剩余内容
            remaining = full_text
            if link_text and link_text in remaining:
                remaining = remaining.replace(link_text, "", 1).strip()
            if len(remaining) > 15:
                return remaining[:300]
            parent = parent.parent

        # 策略2: 检查后续兄弟节点
        sibling = a_tag.find_next_sibling()
        if sibling:
            text = DataExtractor._clean_text(sibling.get_text())
            if len(text) > 10:
                return text[:300]

        # 策略3: 检查父节点后的兄弟节点
        if a_tag.parent:
            parent_sibling = a_tag.parent.find_next_sibling()
            if parent_sibling:
                text = DataExtractor._clean_text(parent_sibling.get_text())
                if len(text) > 10:
                    return text[:300]

        return ""

    @staticmethod
    def _is_result_like(item: dict, query: str, score: float) -> bool:
        """
        判断一个候选项是否像是有效搜索结果
        结合文本特征和 query 相关性综合评估
        """
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        combined = (title + " " + snippet).lower()

        # 过滤条件
        if len(title) < 3:
            return False

        # 排除明显的非结果内容
        noise_words = [
            "下一页", "上一页", "更多", "首页", "登录", "注册", "设置",
            "关于我们", "联系我们", "隐私政策", "服务条款", "帮助中心",
            "next", "previous", "home", "login", "sign up", "settings",
            "about us", "contact us", "privacy", "terms of service",
            "copyright", "cookie", "feedback", "report",
        ]
        if any(nw in combined for nw in noise_words):
            return False

        # 太短的 snippet 也要有一定相关性
        if len(snippet) < 8 and score < 0.1:
            return False

        return True

    def _extract_sklearn_results(
        self, html: str, base_url: str, query: str
    ) -> list[dict]:
        """
        sklearn 文本模式提取（不依赖 CSS 选择器）
        1. 从 HTML 中提取所有带链接的候选文本块
        2. 用 TF-IDF 向量化所有候选，计算与 query 的相似度
        3. 过滤低相关性候选，按相似度排序
        :param html: 原始 HTML
        :param base_url: 页面基础 URL
        :param query: 搜索关键词
        :return: [{"title": "...", "url": "...", "snippet": "..."}, ...]
        """
        soup = self._make_soup(html)

        # 移除干扰元素
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()

        # Step 1: 提取所有候选块
        candidates: list[dict] = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "")
            url = self._resolve_url(href, base_url)

            # 只保留有效的外部链接
            if not url.startswith(("http://", "https://")):
                continue
            # 排除页面内锚点、搜索引擎自身链接
            parsed_base = urlparse(base_url)
            parsed_url = urlparse(url)
            if parsed_url.netloc == parsed_base.netloc:
                continue

            title = self._clean_text(a_tag.get_text())
            if len(title) < 3:
                continue

            snippet = self._find_nearby_text(a_tag)
            candidates.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })

        if not candidates:
            return []

        # Step 2: TF-IDF 向量化 + 相似度计算
        candidate_texts = [f"{c['title']} {c['snippet']}" for c in candidates]

        try:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                max_features=1000,
                sublinear_tf=True,
            )
            all_vectors = vectorizer.fit_transform([query] + candidate_texts)
            query_vec = all_vectors[0:1]
            candidate_vecs = all_vectors[1:]
            scores = cosine_similarity(query_vec, candidate_vecs).flatten()
        except ValueError:
            # 候选太少时回退
            scores = np.ones(len(candidates)) * 0.5

        # Step 3: 过滤 + 去重 + 排序
        results: list[dict] = []
        seen_urls: set[str] = set()

        # 按相似度降序排列
        scored_candidates = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )

        for candidate, score in scored_candidates:
            url = candidate["url"]
            if url in seen_urls:
                continue
            if not self._is_result_like(candidate, query, score):
                continue

            seen_urls.add(url)
            results.append({
                "title": candidate["title"],
                "url": url,
                "snippet": candidate["snippet"],
            })

        return results[:50]

    # ==================== 双路合并 ====================

    @staticmethod
    def _merge_dual_results(
        bs4_results: list[dict],
        sklearn_results: list[dict],
    ) -> list[dict]:
        """
        合并 bs4 和 sklearn 两路提取结果
        - URL 相同 → 保留摘要更长的那条
        - 只有一路命中 → 直接加入
        :return: 合并去重后的结果列表
        """
        merged: dict[str, dict] = {}  # key=url 或 title, value=最佳结果

        for r in bs4_results:
            key = r.get("url") or r.get("title", "")
            if key:
                merged[key] = r

        for r in sklearn_results:
            key = r.get("url") or r.get("title", "")
            if not key:
                continue
            if key in merged:
                # 保留摘要更丰富的那条
                existing_len = len(merged[key].get("snippet", ""))
                new_len = len(r.get("snippet", ""))
                if new_len > existing_len:
                    merged[key] = r
            else:
                merged[key] = r

        # 去重后还原为列表
        return list(merged.values())

    # ==================== 批量处理 ====================

    def process_all(self) -> list[dict]:
        """
        对所有原始数据进行结构化提取（双引擎：bs4 + sklearn）
        :return: [
            {"engine": "百度", "query": "...", "results": [...]},
            ...
        ]
        """
        if not self.raw_data:
            self.console.print(Panel("[yellow]无原始数据可提取[/yellow]", border_style="yellow"))
            return []

        self.console.print(Panel.fit(
            f"[bold cyan]开始双引擎提取[/bold cyan] "
            f"(bs4 CSS选择器 + sklearn 文本模式)\n"
            f"共 [yellow]{len(self.raw_data)}[/yellow] 个引擎的结果",
            border_style="cyan",
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("[cyan]双路提取中...", total=len(self.raw_data))

            for item in self.raw_data:
                engine = item.get("engine", "未知引擎")
                html = item.get("html", "")
                base_url = item.get("url", "")
                query = item.get("query", "")

                if not html:
                    progress.advance(task)
                    continue

                # 路径1: bs4 CSS 选择器提取
                bs4_results = self._extract_engine_results(html, engine, base_url)

                # 路径2: sklearn 文本模式提取
                sklearn_results = self._extract_sklearn_results(html, base_url, query)

                # 双路合并
                merged_results = self._merge_dual_results(bs4_results, sklearn_results)

                self.extracted_data.append({
                    "engine": engine,
                    "query": query,
                    "results": merged_results,
                })
                progress.advance(task)

        # 统计
        total_results = sum(len(d["results"]) for d in self.extracted_data)
        self.console.print(Panel(
            f"[green]提取完成[/green] — "
            f"[cyan]{len(self.extracted_data)}[/cyan] 个引擎, "
            f"共 [cyan]{total_results}[/cyan] 条结果",
            border_style="green",
        ))

        return self.extracted_data

    def export(self) -> list[dict]:
        """导出提取结果，交给 wash_data.py"""
        return self.extracted_data

    def run(self, raw_data: list[dict]) -> list[dict]:
        """
        提取模块主流程
        1. 接收原始数据
        2. 批量提取
        3. 导出
        """
        self.receive(raw_data)
        return self.process_all()


# ==================== 独立测试 ====================

if __name__ == "__main__":
    # 模拟不同搜索引擎返回的 HTML
    mock_baidu_html = """
    <html><body>
    <div class="result c-container">
        <h3><a href="https://www.runoob.com/python/">Python 入门 - 菜鸟教程</a></h3>
        <div class="c-abstract">Python 基础教程，包含 Python 语法、数据类型、条件语句、函数等知识。</div>
    </div>
    <div class="result c-container">
        <h3><a href="https://www.liaoxuefeng.com/wiki/python">Python 教程 - 廖雪峰</a></h3>
        <div class="c-abstract">廖雪峰 Python 教程，涵盖基础、函数、模块、面向对象、IO、进程线程等内容。</div>
    </div>
    <div class="result c-container">
        <h3><a href="https://www.python.org/doc/">Python 官方文档</a></h3>
        <div class="c-row">Python 官方文档，包含完整的语言参考、标准库文档及教程指南。</div>
    </div>
    <div class="result-op">
        <h3><a href="https://example-ad.com/python">【限时优惠】Python 编程课</a></h3>
        <span class="content-right_8Zs40">广告推广：限时优惠！名额有限，立即报名！</span>
    </div>
    </body></html>
    """

    mock_bing_html = """
    <html><body>
    <ol>
    <li class="b_algo">
        <h2><a href="https://www.runoob.com/python/">Python 基础教程 | 菜鸟教程</a></h2>
        <div class="b_caption">
            <p>Python 基础教程，包含 Python 语法、数据类型等知识。</p>
        </div>
    </li>
    <li class="b_algo">
        <h2><a href="https://realpython.com/">Python Tutorial - Real Python</a></h2>
        <div class="b_caption">
            <p>Learn Python online: Python tutorials for developers of all skill levels.</p>
        </div>
    </li>
    </ol>
    </body></html>
    """

    mock_generic_html = """
    <html><body>
    <article><h3><a href="https://example.com/article">An Interesting Article</a></h3><p>This is a summary.</p></article>
    <article><h3><a href="https://example.com/another">Another Article</a></h3><p>Another summary here.</p></article>
    </body></html>
    """

    raw_data = [
        {"engine": "百度", "url": "https://www.baidu.com/s?wd=Python",
         "query": "Python 教程", "html": mock_baidu_html},
        {"engine": "必应搜索", "url": "https://cn.bing.com/search?q=Python",
         "query": "Python 教程", "html": mock_bing_html},
        {"engine": "示例网站", "url": "https://example.com",
         "query": "test", "html": mock_generic_html},
    ]

    extractor = DataExtractor()
    results = extractor.run(raw_data)

    print("\n提取结果预览:")
    for engine_data in results:
        print(f"\n[{engine_data['engine']}] ({len(engine_data['results'])} 条)")
        for i, item in enumerate(engine_data["results"], 1):
            print(f"  {i}. {item['title']}")
            print(f"     {item['url']}")
            if item["snippet"]:
                print(f"     {item['snippet'][:80]}")
