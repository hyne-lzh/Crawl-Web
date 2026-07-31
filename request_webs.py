"""
request_webs.py — 爬取数据模块（主程序入口）
使用 playwright 爬取搜索引擎结果，加载 webs.json 配置
用户输入问题 -> 爬取 -> 交给 extract_data.py -> 交给 wash_data.py
"""

import asyncio
import json
from pathlib import Path
from urllib.parse import quote
from datetime import datetime

from playwright.async_api import async_playwright, Browser, Page

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)
from rich.table import Table
from rich.rule import Rule
from rich import box

from extract_data import DataExtractor
from wash_data import DataWasher


class WebCrawler:
    """搜索引擎爬虫 — 从 webs.json 加载引擎配置并进行爬取"""

    # ============ 引擎名称标准化（新 webs.json 已直接使用标准名，保留映射以防旧格式）============
    ENGINE_NAME_MAP: dict[str, str] = {}

    # ============ 搜索 URL 模板（兼容旧格式 webs.json 的兜底映射）============
    SEARCH_URLS: dict[str, str] = {
        "Google":              "https://www.google.com/search?q={query}&hl=zh-CN",
        "百度":                "https://www.baidu.com/s?wd={query}",
        "DuckDuckGo":          "https://duckduckgo.com/?q={query}&ia=web",
        "必应搜索":             "https://cn.bing.com/search?q={query}",
        "知乎搜索":             "https://www.zhihu.com/search?type=content&q={query}",
        "Yandex":              "https://yandex.com/search/?text={query}",
        "Qwant":               "https://www.qwant.com/?q={query}",
        "搜狗微信搜索":         "https://weixin.sogou.com/weixin?type=2&query={query}",
        "头条搜索":             "https://so.toutiao.com/search?keyword={query}",
        "开发者搜索":           "https://kaifa.baidu.com/search?wd={query}",
        "YOU":                 "https://you.com/search?q={query}",
        "Perplexity":          "https://www.perplexity.ai/search?q={query}",
        "Google 图书搜索":      "https://books.google.com/books?q={query}",
        "Semantic Scholar":    "https://www.semanticscholar.org/search?q={query}",
        "SimilarSites":        "https://www.similarsites.com/search?q={query}",
        "Ebooke":              "https://ebookee.com/?s={query}",
        "Wikipedia":           "https://en.wikipedia.org/w/index.php?search={query}",
        "WikiHow":             "https://www.wikihow.com/wikiHowTo?search={query}",
        "CC Search":           "https://search.creativecommons.org/search?q={query}",
        "知网":                "https://kns.cnki.net/kns8s/defaultresult/index?kwd={query}",
        "WordHippo":           "https://www.wordhippo.com/what-is/search?q={query}",
        "TinEye":              "https://tineye.com/search?url={query}",
        "全历史":              "https://www.allhistory.com/search?q={query}",
        "FindIcons":           "https://findicons.com/search/{query}",
        "Iconfinder":          "https://www.iconfinder.com/search?q={query}",
        "Github":              "https://github.com/search?q={query}",
        "Stanford Encyclopedia of Philosophy": "https://plato.stanford.edu/search/search?query={query}",
        "Unsplash":            "https://unsplash.com/s/photos/{query}",
        "Pexels":              "https://www.pexels.com/search/{query}/",
        "Tunefind":            "https://www.tunefind.com/search?q={query}",
        "visualcapitalist":    "https://www.visualcapitalist.com/?s={query}",
        "ProSettings":         "https://prosettings.net/?s={query}",
        "BetaWiki":            "https://betawiki.net/index.php?search={query}",
        "TOP 500":             "https://www.top500.org/search/?q={query}",
        "The Pudding":         "https://pudding.cool/?s={query}",
        "华为IP知识百科":       "https://info.support.huawei.com/info-finder/encyclopedia/zh/search?keyword={query}",
        "LibreStock":          "https://librestock.com/search/?q={query}",
        "SimilarWeb":          "https://www.similarweb.com/search?q={query}",
        "Brave Search":        "https://search.brave.com/search?q={query}",
        "Mojeek":              "https://www.mojeek.com/search?q={query}",
        "Ecosia":              "https://www.ecosia.org/search?q={query}",
        "Yahoo搜索":            "https://search.yahoo.com/search?p={query}",
        "搜狗全网搜索":         "https://sogou.com/web?query={query}",
        "360搜索":             "https://www.so.com/s?q={query}",
        "多吉搜索":             "https://www.dogedoge.com/s?q={query}",
        "Google学术搜索":       "https://scholar.google.com/scholar?q={query}&hl=zh-CN",
        "arXiv预印本":          "https://arxiv.org/search/?query={query}",
        "AMiner学术":           "https://www.aminer.cn/search/pub?t={query}",
        "鸠摩搜书":             "https://www.jiumodiary.com/s/{query}",
        "Stack Overflow":      "https://stackoverflow.com/search?q={query}",
        "HuggingFace模型搜索":  "https://huggingface.co/search/full-text?q={query}",
        "Pixabay图库":          "https://pixabay.com/images/search/{query}/",
        "Flickr图片搜索":        "https://www.flickr.com/search/?text={query}",
        "百度百科":             "https://baike.baidu.com/search/word?word={query}",
        "中文维基百科":         "https://zh.wikipedia.org/w/index.php?search={query}",
        "B站搜索":              "https://search.bilibili.com/all?keyword={query}",
        "IMDb影视搜索":          "https://www.imdb.com/find?q={query}",
        "Wolfram Alpha":       "https://www.wolframalpha.com/input?i={query}",
        "Urban Dictionary":    "https://www.urbandictionary.com/define.php?term={query}",
        "Bandcamp音乐":         "https://bandcamp.com/search?q={query}",
        "Wayback Machine存档检索": "https://web.archive.org/web/*/{query}",
        "Reddit搜索":           "https://www.reddit.com/search/?q={query}",
        "CNKI研学平台":          "https://x.cnki.net/search/searchresult?kw={query}",
        "GitLab搜索":           "https://gitlab.com/search?search={query}",
        "Google图片搜索":        "https://images.google.com/search?q={query}&hl=zh-CN",
        "百度图片":             "https://image.baidu.com/search/index?tn=baiduimage&word={query}",
        "必应图片":             "https://cn.bing.com/images/search?q={query}",
        "The Free Dictionary":  "https://encyclopedia.thefreedictionary.com/{query}",
        "CNKI外文文献":          "https://kns.cnki.net/kcms/detail/search.aspx?dbcode=WJDC&v={query}",
    }

    # 不适合文本搜索的站点（新版 webs.json 已剔除大部分，保留兜底）
    SKIP_ENGINES: set[str] = set()

    def __init__(self, config_path: str = "webs.json"):
        self.config_path = Path(config_path)
        self.engines: list[dict] = []
        self.raw_results: list[dict] = []
        self.console = Console()
        self.query: str = ""

    # ==================== 名称处理 ====================

    def _normalize_name(self, name: str) -> str:
        """将 webs.json 中的名称标准化为 extract_data 用的名称"""
        return self.ENGINE_NAME_MAP.get(name, name)

    def _build_search_url(self, engine: dict, query: str) -> str | None:
        """
        为给定引擎构建搜索 URL
        新版 webs.json 中 url 已包含 {query} 占位符，直接替换即可
        :return: 搜索 URL，若无法搜索返回 None
        """
        norm_name = self._normalize_name(engine["name"])
        raw_url = engine.get("url", "")

        # 需要跳过的站点
        if norm_name in self.SKIP_ENGINES:
            return None

        # 新格式：URL 已包含 {query}，直接替换
        if "{query}" in raw_url:
            return raw_url.replace("{query}", quote(query))

        # 兼容旧格式：通过 SEARCH_URLS 字典查找
        if norm_name in self.SEARCH_URLS:
            return self.SEARCH_URLS[norm_name].format(query=quote(query))

        # 最后兜底：尝试通用搜索路径
        base = raw_url.rstrip("/")
        return f"{base}/search?q={quote(query)}"

    # ==================== 加载配置 ====================

    def load_engines(self) -> list[dict]:
        """
        从 webs.json 加载搜索引擎列表
        新版格式为 {名称: {url, speed}} 字典，自动转为内部列表格式
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, list):
                # 旧版数组格式兼容
                self.engines = raw
            elif isinstance(raw, dict):
                # 新版字典格式：{名称: {url, speed}}
                self.engines = [
                    {"name": name, "url": cfg.get("url", ""), "speed": cfg.get("speed", 5)}
                    for name, cfg in raw.items()
                ]
            else:
                self.engines = []

            self.console.print(
                f"  [green][OK][/green] 已加载 [cyan]{len(self.engines)}[/cyan] 个网站"
            )
        except FileNotFoundError:
            self.console.print(
                f"  [red][ERROR][/red] 配置文件不存在: [bold]{self.config_path}[/bold]"
            )
            self.engines = []
        except json.JSONDecodeError as e:
            self.console.print(f"  [red][ERROR][/red] JSON 格式错误: {e}")
            self.engines = []
        return self.engines

    def filter_by_speed(self, min_speed: int = 0) -> list[dict]:
        """
        按 speed 过滤搜索引擎
        :param min_speed: 最低速度阈值（0-10，0 最慢）
        :return: 过滤后的引擎列表
        """
        filtered = [e for e in self.engines if e.get("speed", 0) >= min_speed]
        excluded = len(self.engines) - len(filtered)
        if excluded > 0:
            self.console.print(
                f"  [dim]speed ≥ {min_speed}: 保留 [cyan]{len(filtered)}[/cyan] 个, "
                f"排除 [yellow]{excluded}[/yellow] 个慢速站[/dim]"
            )
        else:
            self.console.print(
                f"  [dim]speed ≥ {min_speed}: 保留全部 [cyan]{len(filtered)}[/cyan] 个[/dim]"
            )
        return filtered

    # ==================== 浏览器管理 ====================

    @staticmethod
    async def _setup_browser(playwright) -> Browser:
        """启动 Chromium 浏览器"""
        return await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )

    @staticmethod
    async def _create_page(browser: Browser) -> Page:
        """创建预配置的浏览器页面（含反检测脚本）"""
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
        )
        page = await context.new_page()
        # 隐藏自动化特征
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
        """)
        return page

    # ==================== 爬取单页 ====================

    async def crawl_single(
        self, engine: dict, query: str, page: Page
    ) -> dict:
        """
        爬取单个搜索引擎的结果页
        :param engine: 引擎配置 {name, url, speed}
        :param query: 搜索关键词
        :param page: 复用的 Playwright Page
        :return: 爬取结果 {engine, search_url, query, html, status, error}
        """
        norm_name = self._normalize_name(engine["name"])
        search_url = self._build_search_url(engine, query)

        result = {
            "engine": norm_name,
            "url": search_url or engine["url"],
            "query": query,
            "html": "",
            "status": "pending",
            "error": "",
        }

        # 无法构建搜索 URL 则跳过
        if search_url is None:
            result["status"] = "skipped"
            return result

        try:
            response = await page.goto(
                search_url,
                wait_until="domcontentloaded",
                timeout=30_000,
            )

            # 等待页面初步渲染
            await asyncio.sleep(2)

            # 尝试等待任意链接出现（确认内容加载）
            try:
                await page.wait_for_selector("a[href]", timeout=5_000)
            except Exception:
                pass

            # 滚动以触发懒加载
            await page.evaluate(
                "window.scrollTo(0, document.body.scrollHeight / 3)"
            )
            await asyncio.sleep(1)

            result["html"] = await page.content()
            result["status"] = (
                "success" if response and response.ok else "http_error"
            )

        except asyncio.TimeoutError:
            result["status"] = "timeout"
            result["error"] = "页面加载超时 (30s)"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:200]

        return result

    # ==================== 批量爬取 ====================

    async def crawl_all(
        self,
        query: str,
        engines: list[dict],
        engine_delay: float = 1.5,
    ) -> list[dict]:
        """
        逐个爬取所有引擎（异步并发控制 + 间隔防封）
        :param query: 搜索关键词
        :param engines: 待爬取的引擎列表
        :param engine_delay: 引擎间等待秒数（防限流）
        :return: 所有爬取结果
        """
        self.query = query
        self.raw_results = []

        if not engines:
            self.console.print("[yellow]无可爬取的引擎[/yellow]")
            return []

        # 统计可搜索 vs 非搜索
        searchable = sum(
            1 for e in engines
            if self._build_search_url(e, query) is not None
        )
        non_search = len(engines) - searchable

        self.console.print(Panel.fit(
            f"[bold cyan]开始爬取[/bold cyan]\n"
            f"关键词: [yellow]{query}[/yellow]\n"
            f"目标: [cyan]{len(engines)}[/cyan] 个引擎 "
            f"([green]{searchable}[/green] 可搜索"
            + (f", [dim]{non_search} 跳过[/dim])" if non_search else "") + ")",
            border_style="cyan",
        ))

        async with async_playwright() as p:
            browser = await self._setup_browser(p)
            page = await self._create_page(browser)

            results: list[dict] = []
            total = len(engines)
            success_count = 0
            error_count = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("[cyan]爬取中...", total=total)

                for i, engine in enumerate(engines):
                    name = engine["name"]
                    speed = engine.get("speed", 0)

                    progress.update(
                        task,
                        description=f"[cyan]{name[:30]} [dim](speed={speed})[/dim][/cyan]",
                    )

                    result = await self.crawl_single(engine, query, page)
                    results.append(result)

                    if result["status"] == "success":
                        success_count += 1
                    elif result["status"] not in ("skipped",):
                        error_count += 1

                    progress.advance(task)

                    # 引擎间间隔，防止被限流 / 封 IP
                    if i < total - 1:
                        await asyncio.sleep(engine_delay)

            await browser.close()

        self.raw_results = results

        # 结果汇总
        skipped = sum(1 for r in results if r["status"] == "skipped")
        status_text = f"[green]✓ 成功: {success_count}[/green]"
        if error_count > 0:
            status_text += f"    [red]✗ 失败: {error_count}[/red]"
        if skipped > 0:
            status_text += f"    [dim]— 跳过: {skipped}[/dim]"

        self.console.print(Panel(
            status_text,
            border_style="green" if error_count == 0 else "yellow",
        ))

        # 打印失败/错误详情
        failed = [
            r for r in results
            if r["status"] not in ("success", "skipped")
        ]
        if failed:
            err_table = Table(
                title="失败详情",
                box=box.SIMPLE,
                border_style="red",
            )
            err_table.add_column("引擎", style="bold")
            err_table.add_column("状态", style="red")
            err_table.add_column("错误信息", style="dim")
            for r in failed:
                err_table.add_row(
                    r["engine"],
                    r["status"],
                    r["error"][:100] or "-",
                )
            self.console.print(err_table)

        return results

    # ==================== 主流程 ====================

    def run(
        self,
        query: str | None = None,
        min_speed: int = 5,
        do_extract: bool = True,
        do_wash: bool = True,
    ) -> None:
        """
        主流程入口
        1. 加载 webs.json
        2. 获取搜索关键词
        3. 按 speed 过滤引擎
        4. 并发爬取
        5. 交给 extract_data 提取
        6. 交给 wash_data 清洗 + 展示 + 保存
        """
        self.console.print(Rule(
            "[bold bright_blue]搜索引擎聚合爬虫[/bold bright_blue]",
            style="bright_blue",
        ))

        # ---- 1. 加载配置 ----
        self.console.print("\n[bold]1. 加载配置[/bold]")
        self.load_engines()
        if not self.engines:
            return

        # ---- 2. 获取关键词 ----
        if query is None:
            self.console.print("\n[bold]2. 输入搜索关键词[/bold]")
            query = input("  >>> ").strip()
        else:
            self.console.print(
                f"\n[bold]2. 搜索关键词:[/bold] [yellow]{query}[/yellow]"
            )

        if not query:
            self.console.print("[red]关键词不能为空，已取消。[/red]")
            return

        # ---- 3. 过滤 + 爬取 ----
        self.console.print(f"\n[bold]3. 按速度过滤 (speed ≥ {min_speed})[/bold]")
        engines_to_crawl = self.filter_by_speed(min_speed)

        self.console.print("\n[bold]4. 爬取中[/bold]")
        asyncio.run(self.crawl_all(query, engines_to_crawl))

        # 提取成功的 HTML 结果
        successful = [
            {
                "engine": r["engine"],
                "url": r["url"],
                "query": r["query"],
                "html": r["html"],
            }
            for r in self.raw_results
            if r["status"] == "success" and r["html"]
        ]

        if not successful:
            self.console.print(Panel(
                "[red]所有引擎爬取均失败，无法继续。[/red]\n"
                "请检查网络连接或降低 speed 阈值。",
                border_style="red",
            ))
            return

        self.console.print(
            f"  [green][OK][/green] {len(successful)} 个引擎 HTML 可提取"
        )

        # ---- 4. 提取数据 ----
        if do_extract:
            self.console.print(Rule("[bold]数据提取[/bold]", style="bright_cyan"))
            extractor = DataExtractor()
            extracted_data = extractor.run(successful)
        else:
            extracted_data = successful

        # ---- 5. 清洗 + 展示 + 保存 ----
        if do_wash and extracted_data:
            self.console.print(Rule("[bold]数据清洗[/bold]", style="bright_cyan"))
            washer = DataWasher()
            washer.run(extracted_data, query)

        self.console.print(
            Rule("[bold green]流程结束[/bold green]", style="green")
        )


# ==================== 独立测试 / 命令行入口 ====================

if __name__ == "__main__":
    console = Console()

    console.print(Panel.fit(
        "[bold bright_blue]🔍 搜索引擎聚合爬虫[/bold bright_blue]\n"
        "[dim]加载 webs.json → 爬取 → 提取 → 清洗 → 展示 → 保存[/dim]",
        border_style="bright_blue",
    ))

    # 交互输入
    keyword = input("\n请输入搜索关键词: ").strip()
    if not keyword:
        console.print("[red]关键词不能为空，已退出。[/red]")
        exit(1)

    try:
        speed_input = input("最低速度阈值 (0-10, 默认 5): ").strip()
        min_speed = int(speed_input) if speed_input else 5
    except ValueError:
        console.print("[yellow]输入无效，使用默认值 5[/yellow]")
        min_speed = 5

    crawler = WebCrawler()
    crawler.run(query=keyword, min_speed=min_speed)
