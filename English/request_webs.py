"""
request_webs.py — Web crawling module (main entry point)
Uses Playwright to crawl search engine results, loading configuration from webs.json
User query → crawl → hand off to extract_data.py → hand off to wash_data.py
Optimized for English / international search engines
"""

import asyncio
import json
from pathlib import Path
from urllib.parse import quote

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
    """Search engine crawler — loads engine configs from webs.json and performs crawling"""

    # ============ Search URL templates (fallback for legacy webs.json or engines without {query} in URL) ============
    SEARCH_URLS: dict[str, str] = {
        # Major search engines
        "Google":              "https://www.google.com/search?q={query}&hl=en",
        "Bing":                "https://www.bing.com/search?q={query}",
        "Yahoo":               "https://search.yahoo.com/search?p={query}",
        "DuckDuckGo":          "https://duckduckgo.com/?q={query}&ia=web",
        "Brave Search":        "https://search.brave.com/search?q={query}",
        "Ecosia":              "https://www.ecosia.org/search?q={query}",
        "Startpage":           "https://www.startpage.com/sp/search?q={query}",
        "Mojeek":              "https://www.mojeek.com/search?q={query}",
        "Swisscows":           "https://swisscows.com/web?query={query}",
        "Yep.com":             "https://yep.com/web?q={query}",
        "Kagi":                "https://kagi.com/search?q={query}",
        "KARMA Search":        "https://karmasearch.io/search?q={query}",
        # AI-powered search
        "ChatGPT Search":      "https://chatgpt.com/search?q={query}",
        "Google AI Mode":      "https://www.google.com/search?q={query}&udm=14",
        "Perplexity.ai":       "https://www.perplexity.ai/search?q={query}",
        "You.com":             "https://you.com/search?q={query}",
        # International engines
        "Yandex":              "https://yandex.com/search/?text={query}",
        "Baidu":               "https://www.baidu.com/s?wd={query}",
        "Sogou":               "https://sogou.com/web?query={query}",
        "Naver":               "https://search.naver.com/search.naver?query={query}",
        # Knowledge / reference
        "Wikipedia":           "https://en.wikipedia.org/w/index.php?search={query}",
        "WikiHow":             "https://www.wikihow.com/wikiHowTo?search={query}",
        "Wolfram Alpha":       "https://www.wolframalpha.com/input?i={query}",
        "Urban Dictionary":    "https://www.urbandictionary.com/define.php?term={query}",
        "The Free Dictionary": "https://encyclopedia.thefreedictionary.com/{query}",
        "WordHippo":           "https://www.wordhippo.com/what-is/search?q={query}",
        # Developer / tech
        "Stack Overflow":      "https://stackoverflow.com/search?q={query}",
        "Github":              "https://github.com/search?q={query}",
        "GitLab":              "https://gitlab.com/search?search={query}",
        # Academic / research
        "Google Scholar":      "https://scholar.google.com/scholar?q={query}&hl=en",
        "Semantic Scholar":    "https://www.semanticscholar.org/search?q={query}",
        "ArXiv":               "https://arxiv.org/search/?query={query}",
        # Media / content
        "Openverse":           "https://openverse.org/search/?q={query}",
        "Unsplash":            "https://unsplash.com/s/photos/{query}",
        "Pexels":              "https://www.pexels.com/search/{query}/",
        "IMDb":                "https://www.imdb.com/find?q={query}",
        # Social / community
        "Reddit":              "https://www.reddit.com/search/?q={query}",
        "X (Twitter)":         "https://x.com/search?q={query}",
        "LinkedIn":            "https://www.linkedin.com/search/results/all/?keywords={query}",
        "SlideShare":          "https://www.slideshare.net/search/slideshow?q={query}",
        # Archive
        "Wayback Machine":     "https://web.archive.org/web/*/{query}",
        # Analytics / tools
        "SimilarWeb":          "https://www.similarweb.com/search?q={query}",
        "LibreStock":          "https://librestock.com/search/?q={query}",
        "FindIcons":           "https://findicons.com/search/{query}",
        "Iconfinder":          "https://www.iconfinder.com/search?q={query}",
    }

    # Sites that are not suitable for text search
    SKIP_ENGINES: set[str] = set()

    def __init__(self, config_path: str = "webs.json"):
        self.config_path = Path(config_path)
        self.engines: list[dict] = []
        self.raw_results: list[dict] = []
        self.console = Console()
        self.query: str = ""

    # ==================== Name handling ====================

    def _normalize_name(self, name: str) -> str:
        """Normalize engine name from webs.json"""
        # No mapping needed — webs.json uses canonical names
        return name

    def _build_search_url(self, engine: dict, query: str) -> str | None:
        """
        Build a search URL for the given engine
        New webs.json format already includes {query} placeholder in URLs — just substitute
        :return: Search URL, or None if not searchable
        """
        norm_name = self._normalize_name(engine["name"])
        raw_url = engine.get("url", "")

        # Skip non-searchable sites
        if norm_name in self.SKIP_ENGINES:
            return None

        # New format: URL already contains {query}, substitute directly
        if "{query}" in raw_url:
            return raw_url.replace("{query}", quote(query))

        # Legacy compat: look up via SEARCH_URLS dictionary
        if norm_name in self.SEARCH_URLS:
            return self.SEARCH_URLS[norm_name].format(query=quote(query))

        # Final fallback: try a generic search path
        base = raw_url.rstrip("/")
        return f"{base}/search?q={quote(query)}"

    # ==================== Config loading ====================

    def load_engines(self) -> list[dict]:
        """
        Load search engine list from webs.json
        New format is a {name: {url, speed}} dict, auto-converted to internal list format
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, list):
                # Legacy array format compat
                self.engines = raw
            elif isinstance(raw, dict):
                # New dict format: {name: {url, speed}}
                self.engines = [
                    {"name": name, "url": cfg.get("url", ""), "speed": cfg.get("speed", 5)}
                    for name, cfg in raw.items()
                ]
                # Sort by speed descending (faster engines first)
                self.engines.sort(key=lambda e: e.get("speed", 0), reverse=True)
            else:
                self.engines = []

            self.console.print(
                f"  [green][OK][/green] Loaded [cyan]{len(self.engines)}[/cyan] engines"
            )
        except FileNotFoundError:
            self.console.print(
                f"  [red][ERROR][/red] Config file not found: [bold]{self.config_path}[/bold]"
            )
            self.engines = []
        except json.JSONDecodeError as e:
            self.console.print(f"  [red][ERROR][/red] JSON format error: {e}")
            self.engines = []
        return self.engines

    def filter_by_speed(self, min_speed: int = 0) -> list[dict]:
        """
        Filter search engines by speed threshold
        :param min_speed: Minimum speed rating (0-10, 10 = fastest / most reliable)
        :return: Filtered engine list
        """
        filtered = [e for e in self.engines if e.get("speed", 0) >= min_speed]
        excluded = len(self.engines) - len(filtered)
        if excluded > 0:
            self.console.print(
                f"  [dim]speed >= {min_speed}: kept [cyan]{len(filtered)}[/cyan], "
                f"excluded [yellow]{excluded}[/yellow] slower sites[/dim]"
            )
        else:
            self.console.print(
                f"  [dim]speed >= {min_speed}: kept all [cyan]{len(filtered)}[/cyan][/dim]"
            )
        return filtered

    # ==================== Browser management ====================

    @staticmethod
    async def _setup_browser(playwright) -> Browser:
        """Launch Chromium browser"""
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
        """Create a pre-configured browser page (with anti-detection script)"""
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
        )
        page = await context.new_page()
        # Hide automation fingerprints
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
        """)
        return page

    # ==================== Single-page crawl ====================

    async def crawl_single(
        self, engine: dict, query: str, page: Page
    ) -> dict:
        """
        Crawl a single search engine results page
        :param engine: Engine config {name, url, speed}
        :param query: Search keyword
        :param page: Reusable Playwright Page
        :return: Crawl result {engine, search_url, query, html, status, error}
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

        # Skip if search URL cannot be built
        if search_url is None:
            result["status"] = "skipped"
            return result

        try:
            response = await page.goto(
                search_url,
                wait_until="domcontentloaded",
                timeout=30_000,
            )

            # Wait for initial rendering
            await asyncio.sleep(2)

            # Try waiting for any link to appear (confirm content loaded)
            try:
                await page.wait_for_selector("a[href]", timeout=5_000)
            except Exception:
                pass

            # Scroll to trigger lazy loading
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
            result["error"] = "Page load timeout (30s)"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)[:200]

        return result

    # ==================== Batch crawl ====================

    async def crawl_all(
        self,
        query: str,
        engines: list[dict],
        engine_delay: float = 1.5,
    ) -> list[dict]:
        """
        Crawl all engines sequentially (async concurrency with inter-engine delay for anti-rate-limiting)
        :param query: Search keyword
        :param engines: List of engines to crawl
        :param engine_delay: Delay between engines in seconds (anti-rate-limiting)
        :return: All crawl results
        """
        self.query = query
        self.raw_results = []

        if not engines:
            self.console.print("[yellow]No engines to crawl[/yellow]")
            return []

        # Count searchable vs non-searchable
        searchable = sum(
            1 for e in engines
            if self._build_search_url(e, query) is not None
        )
        non_search = len(engines) - searchable

        self.console.print(Panel.fit(
            f"[bold cyan]Starting crawl[/bold cyan]\n"
            f"Keyword: [yellow]{query}[/yellow]\n"
            f"Target: [cyan]{len(engines)}[/cyan] engines "
            f"([green]{searchable}[/green] searchable"
            + (f", [dim]{non_search} skipped[/dim])" if non_search else "") + ")",
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
                task = progress.add_task("[cyan]Crawling...", total=total)

                for i, engine in enumerate(engines):
                    name = engine["name"]
                    speed = engine.get("speed", 0)

                    progress.update(
                        task,
                        description=f"[cyan]{name[:35]} [dim](speed={speed})[/dim][/cyan]",
                    )

                    result = await self.crawl_single(engine, query, page)
                    results.append(result)

                    if result["status"] == "success":
                        success_count += 1
                    elif result["status"] not in ("skipped",):
                        error_count += 1

                    progress.advance(task)

                    # Delay between engines to avoid rate limiting / IP bans
                    if i < total - 1:
                        await asyncio.sleep(engine_delay)

            await browser.close()

        self.raw_results = results

        # Summary
        skipped = sum(1 for r in results if r["status"] == "skipped")
        status_text = f"[green]OK: {success_count}[/green]"
        if error_count > 0:
            status_text += f"    [red]FAIL: {error_count}[/red]"
        if skipped > 0:
            status_text += f"    [dim]SKIP: {skipped}[/dim]"

        self.console.print(Panel(
            status_text,
            border_style="green" if error_count == 0 else "yellow",
        ))

        # Print failure / error details
        failed = [
            r for r in results
            if r["status"] not in ("success", "skipped")
        ]
        if failed:
            err_table = Table(
                title="Failure details",
                box=box.SIMPLE,
                border_style="red",
            )
            err_table.add_column("Engine", style="bold")
            err_table.add_column("Status", style="red")
            err_table.add_column("Error message", style="dim")
            for r in failed:
                err_table.add_row(
                    r["engine"],
                    r["status"],
                    r["error"][:100] or "-",
                )
            self.console.print(err_table)

        return results

    # ==================== Main workflow ====================

    def run(
        self,
        query: str | None = None,
        min_speed: int = 5,
        do_extract: bool = True,
        do_wash: bool = True,
    ) -> None:
        """
        Main workflow entry point
        1. Load webs.json
        2. Get search keyword
        3. Filter engines by speed
        4. Crawl concurrently
        5. Hand off to extract_data for extraction
        6. Hand off to wash_data for cleaning + display + save
        """
        self.console.print(Rule(
            "[bold bright_blue]Search Engine Aggregator[/bold bright_blue]",
            style="bright_blue",
        ))

        # ---- 1. Load config ----
        self.console.print("\n[bold]1. Load config[/bold]")
        self.load_engines()
        if not self.engines:
            return

        # ---- 2. Get keyword ----
        if query is None:
            self.console.print("\n[bold]2. Enter search keyword[/bold]")
            query = input("  >>> ").strip()
        else:
            self.console.print(
                f"\n[bold]2. Search keyword:[/bold] [yellow]{query}[/yellow]"
            )

        if not query:
            self.console.print("[red]Keyword cannot be empty. Cancelled.[/red]")
            return

        # ---- 3. Filter + crawl ----
        self.console.print(f"\n[bold]3. Filter by speed (speed >= {min_speed})[/bold]")
        engines_to_crawl = self.filter_by_speed(min_speed)

        self.console.print("\n[bold]4. Crawling[/bold]")
        asyncio.run(self.crawl_all(query, engines_to_crawl))

        # Extract successful HTML results
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
                "[red]All engines failed to crawl. Cannot continue.[/red]\n"
                "Please check your network connection or lower the speed threshold.",
                border_style="red",
            ))
            return

        self.console.print(
            f"  [green][OK][/green] {len(successful)} engines have HTML ready for extraction"
        )

        # ---- 4. Extract data ----
        if do_extract:
            self.console.print(Rule("[bold]Data Extraction[/bold]", style="bright_cyan"))
            extractor = DataExtractor()
            extracted_data = extractor.run(successful)
        else:
            extracted_data = successful

        # ---- 5. Clean + display + save ----
        if do_wash and extracted_data:
            self.console.print(Rule("[bold]Data Cleaning[/bold]", style="bright_cyan"))
            washer = DataWasher()
            washer.run(extracted_data, query)

        self.console.print(
            Rule("[bold green]Pipeline complete[/bold green]", style="green")
        )


# ==================== Standalone test / CLI entry ====================

if __name__ == "__main__":
    console = Console()

    console.print(Panel.fit(
        "[bold bright_blue]Search Engine Aggregator[/bold bright_blue]\n"
        "[dim]Load webs.json → Crawl → Extract → Clean → Display → Save[/dim]",
        border_style="bright_blue",
    ))

    # Interactive input
    keyword = input("\nEnter search keyword: ").strip()
    if not keyword:
        console.print("[red]Keyword cannot be empty. Exiting.[/red]")
        exit(1)

    try:
        speed_input = input("Minimum speed threshold (0-10, default 5): ").strip()
        min_speed = int(speed_input) if speed_input else 5
    except ValueError:
        console.print("[yellow]Invalid input, using default 5[/yellow]")
        min_speed = 5

    crawler = WebCrawler()
    crawler.run(query=keyword, min_speed=min_speed)
