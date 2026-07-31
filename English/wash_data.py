"""
wash_data.py — Data cleaning and filtering module
Uses sklearn for ad detection, deduplication, and relevance ranking
Receives extraction results from extract_data.py → filters → displays & saves
Optimized for English / international search engines
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskID
from rich.text import Text
from rich.rule import Rule
from rich import box


class DataWasher:
    """Data cleaner — ad filtering, deduplication, relevance ranking, display & save"""

    # Ad corpus (positive samples) — heavily weighted toward English ad patterns
    AD_CORPUS: list[str] = [
        # English ad phrases
        "sponsored ad promotion buy now discount deal offer",
        "advertisement paid result sponsored content brand promotion",
        "limited time offer special deal flash sale shop now",
        "best price guaranteed free shipping money back guarantee",
        "click here to buy order now exclusive deal",
        "save up to percent off clearance sale today only",
        "affordable professional services contact us today quote",
        "top rated recommended best seller don't miss out",
        "subscribe now sign up free trial premium membership",
        "download now instant access claim your spot limited",
        "ad 推广 sponsored promoted listing commercial result",
        # General ad patterns
        "sponsored by partner promoted content advertisement",
        "buy now discount coupon promo code deal of the day",
        "sale ends soon hurry before it's gone act fast",
        "exclusive offer vip deal premium access get started",
    ]

    # Normal search result corpus (negative samples) — English focused
    NORMAL_CORPUS: list[str] = [
        "wikipedia documentation tutorial guide reference manual",
        "official website government education research academic journal",
        "open source GitHub repository documentation developer guide",
        "API reference specification standard protocol RFC",
        "news article report analysis review commentary opinion",
        "history science culture geography medicine law economics",
        "how to guide tutorial beginner advanced expert walkthrough",
        "FAQ frequently asked questions help center support knowledge base",
        "definition meaning encyclopedia dictionary thesaurus glossary",
        "research paper study findings methodology results conclusion",
        "blog post article essay discussion forum community Q&A",
        "university college institute department faculty course syllabus",
        "conference proceedings presentation slides whitepaper technical report",
        "data statistics infographic visualization analysis trends",
        "cookbook recipe instructions ingredients preparation method",
    ]

    # Ad keywords (rule-based pre-filter) — English focused
    AD_KEYWORDS: list[str] = [
        # Direct ad indicators
        "sponsored", "advertisement", "paid result", "promoted",
        "ad", "promotion", "promotional",
        # Commercial intent
        "buy now", "shop now", "order now", "subscribe",
        "free trial", "free quote", "free consultation",
        "limited time", "flash sale", "clearance", "closeout",
        "best price", "lowest price", "cheap", "affordable",
        "discount", "coupon", "promo code", "voucher",
        "save up to", "percent off", "off today",
        "money back", "satisfaction guaranteed",
        "act now", "don't miss", "hurry", "ending soon",
        "exclusive deal", "special offer", "vip access",
        "click here", "learn more", "get started now",
        "sign up today", "register now", "enroll today",
        "premium", "pro plan", "upgrade", "unlock",
        "featured listing", "top pick", "recommended for you",
        "partner content", "brand partner", "affiliate link",
        "call now", "contact us today", "instant quote",
        # URL-based
        "utm_source", "utm_medium", "utm_campaign",
        "ref=", "affiliate", "partner_id",
    ]

    def __init__(self, output_dir: str = "results"):
        """
        Initialize the cleaner
        :param output_dir: Directory for saving results
        """
        self.extracted_data: list[dict] = []
        self.cleaned_data: list[dict] = []
        self.ad_model: LogisticRegression | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()

    # --------------------- Receive data ---------------------

    def receive(self, extracted_data: list[dict]) -> None:
        """
        Receive extracted data from extract_data.py
        :param extracted_data: Structured data after extraction
        """
        self.extracted_data = extracted_data
        self.cleaned_data = []

    # --------------------- Ad detection ---------------------

    @staticmethod
    def _join_item(item: dict) -> str:
        """Merge title/snippet of a single result into one text string"""
        parts = []
        if item.get("title"):
            parts.append(item["title"])
        if item.get("snippet"):
            parts.append(item["snippet"])
        return " ".join(parts)

    @staticmethod
    def _rule_based_ad_check(text: str) -> bool:
        """Rule-based ad detection (quick pre-filter)"""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in DataWasher.AD_KEYWORDS)

    def train_ad_model(self) -> None:
        """
        Train ad detection model using sklearn
        TF-IDF vectorization + LogisticRegression classifier
        Uses word-level n-grams (optimized for English with natural word boundaries)
        """
        # Build training data
        X_texts = self.AD_CORPUS + self.NORMAL_CORPUS
        y = [1] * len(self.AD_CORPUS) + [0] * len(self.NORMAL_CORPUS)

        # TF-IDF vectorization — word-level analyzer better for English
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            max_features=1000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            stop_words="english",
        )
        X = self.vectorizer.fit_transform(X_texts)

        # Train logistic regression classifier
        self.ad_model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            class_weight="balanced",
        )
        self.ad_model.fit(X, y)

    def detect_ads(self, item: dict) -> bool:
        """
        Detect whether a single result is an ad
        Rule-based pre-check → URL feature check → sklearn model confirmation
        :param item: Single search result {title, url, snippet}
        :return: True if it is an ad
        """
        text = self._join_item(item)
        if not text.strip():
            return False

        ad_score = 0

        # 1. Rule-based detection
        if self._rule_based_ad_check(text):
            ad_score += 3

        # 2. URL feature detection (ad/tracking query parameters)
        url = item.get("url", "")
        url_lower = url.lower()
        ad_url_keywords = [
            "ad", "sponsor", "promote", "promotion", "track",
            "affiliate", "utm_source", "utm_medium", "utm_campaign",
            "ref=", "partner_id", "click_id", "campaign_id",
            "gclid", "fbclid", "msclkid",  # Google/Facebook/Microsoft click tracking
        ]
        if url and any(kw in url_lower for kw in ad_url_keywords):
            ad_score += 2

        # 3. Title keyword check (ads often use urgency/sales language in titles)
        title = item.get("title", "").lower()
        urgency_sales_words = [
            "sale", "deal", "offer", "discount", "price",
            "buy", "shop", "best", "top", "cheap", "save",
            "review 202", "best of", "#1", "number one",
            "official site", "official website",
        ]
        urgency_hits = sum(1 for kw in urgency_sales_words if kw in title)
        if urgency_hits >= 2:
            ad_score += 1

        # 4. sklearn model prediction
        if self.ad_model is not None and self.vectorizer is not None:
            try:
                vec = self.vectorizer.transform([text])
                proba = self.ad_model.predict_proba(vec)[0, 1]
                if proba > 0.5:
                    ad_score += 3
                elif proba > 0.35:
                    ad_score += 1
            except Exception:
                pass  # Skip model judgment if vectorization fails

        # Combined judgment: score >= 4 is considered an ad
        return ad_score >= 4

    def filter_ads(self) -> None:
        """Filter out all ad results"""
        if self.ad_model is None:
            self.train_ad_model()

        total_removed = 0
        for engine_data in self.extracted_data:
            results = engine_data.get("results", [])
            before = len(results)
            engine_data["results"] = [
                item for item in results if not self.detect_ads(item)
            ]
            total_removed += before - len(engine_data["results"])
        self.cleaned_data = self.extracted_data

        if total_removed > 0:
            self.console.print(f"  [dim]Ad filter: removed [yellow]{total_removed}[/yellow] ad-like results[/dim]")

    # --------------------- Deduplication ---------------------

    def deduplicate(self, threshold: float = 0.85) -> None:
        """
        Deduplication: remove results with text similarity >= threshold
        Uses TF-IDF + cosine_similarity
        :param threshold: Similarity threshold (0-1), above which results are considered duplicates
        """
        if not self.cleaned_data:
            return

        # Collect all items (cross-engine dedup)
        all_items: list[dict] = []
        for engine_data in self.cleaned_data:
            all_items.extend(engine_data.get("results", []))

        if len(all_items) <= 1:
            return

        # Extract texts and vectorize
        texts = [self._join_item(item) for item in all_items]
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(texts)
        sim_matrix = cosine_similarity(vec)

        # Mark indices to remove
        n = len(all_items)
        to_remove: set[int] = set()
        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                if sim_matrix[i, j] >= threshold:
                    # Keep the one with longer text (more informative)
                    if len(texts[i]) >= len(texts[j]):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # i is removed, move to next i

        # Build global index → (engine_index, intra_engine_index) mapping
        index_map: list[tuple[int, int]] = []  # [(engine_i, item_j), ...]
        for ei, engine_data in enumerate(self.cleaned_data):
            for ej, item in enumerate(engine_data.get("results", [])):
                index_map.append((ei, ej))

        # Collect retained results per engine
        new_results: dict[int, list[dict]] = {ei: [] for ei in range(len(self.cleaned_data))}
        for gi, (ei, ej) in enumerate(index_map):
            if gi not in to_remove:
                new_results[ei].append(self.cleaned_data[ei]["results"][ej])

        for ei, engine_data in enumerate(self.cleaned_data):
            engine_data["results"] = new_results[ei]

    # --------------------- Relevance ranking ---------------------

    def rank_by_relevance(self, query: str) -> None:
        """
        Reorder results by relevance to the query
        Uses TF-IDF + cosine_similarity to compute similarity between each result and the query
        :param query: Original search keyword
        """
        if not self.cleaned_data:
            return

        for engine_data in self.cleaned_data:
            results = engine_data.get("results", [])
            if len(results) <= 1:
                continue

            texts = [self._join_item(item) for item in results]
            # Place query first for vectorization
            all_texts = [query] + texts
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(all_texts)
            # Compute cosine similarity between query and each result
            sim_scores = cosine_similarity(vec[0:1], vec[1:]).flatten()

            # Sort by similarity descending
            scored = list(zip(results, sim_scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            engine_data["results"] = [item for item, _ in scored]

    # --------------------- Main cleaning workflow ---------------------

    def wash(self, query: str) -> list[dict]:
        """
        Complete cleaning workflow: ad filtering → dedup → ranking
        :param query: Original search keyword
        :return: Cleaned results
        """
        self.console.print(Panel.fit(
            f"[bold cyan]Cleaning data[/bold cyan]\nQuery: [yellow]{query}[/yellow]",
            border_style="cyan",
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            # 1. Filter ads
            task1 = progress.add_task("[cyan]Ad detection & filtering...", total=100)
            self.filter_ads()
            total_before = sum(len(d.get("results", [])) for d in self.cleaned_data)
            progress.update(task1, completed=100)

            # 2. Deduplicate
            task2 = progress.add_task("[cyan]Similarity deduplication...", total=100)
            self.deduplicate()
            total_after = sum(len(d.get("results", [])) for d in self.cleaned_data)
            progress.update(task2, completed=100)

            # 3. Relevance ranking
            task3 = progress.add_task("[cyan]Relevance ranking...", total=100)
            self.rank_by_relevance(query)
            progress.update(task3, completed=100)

        # Stats panel
        removed = total_before - total_after
        stats = Text()
        stats.append(f"Engines: ", style="dim")
        stats.append(f"{len(self.extracted_data)}", style="bold cyan")
        stats.append(f"  |  After cleaning: ", style="dim")
        stats.append(f"{total_after} results", style="bold green")
        if removed > 0:
            stats.append(f"  |  Removed: ", style="dim")
            stats.append(f"{removed} items", style="bold yellow")
        self.console.print(Panel(stats, border_style="green"))

        return self.cleaned_data

    # --------------------- Display ---------------------

    # Engine color mapping — international engines
    ENGINE_COLORS: dict[str, str] = {
        "Google": "bright_green",
        "Bing": "bright_cyan",
        "DuckDuckGo": "bright_yellow",
        "Brave Search": "orange1",
        "Yahoo": "purple",
        "Ecosia": "green",
        "Startpage": "bright_blue",
        "Mojeek": "magenta",
        "Perplexity.ai": "bright_magenta",
        "ChatGPT Search": "bright_white",
        "You.com": "cyan",
        "Kagi": "yellow",
        "WolframAlpha": "red",
        "Wikipedia": "white",
        "Stack Overflow": "orange1",
        "Github": "bright_black",
        "Reddit": "bright_red",
        "Yandex": "red",
        "Baidu": "bright_blue",
    }
    _DEFAULT_ENGINE_COLOR = "bright_white"

    @classmethod
    def _engine_color(cls, name: str) -> str:
        """Return display color for an engine name"""
        return cls.ENGINE_COLORS.get(name, cls._DEFAULT_ENGINE_COLOR)

    def display(self) -> None:
        """Display cleaned results in the terminal using Rich Tables"""
        if not self.cleaned_data:
            self.console.print(Panel("[yellow]No results to display[/yellow]", border_style="yellow"))
            return

        total = 0
        for engine_data in self.cleaned_data:
            engine_name = engine_data.get("engine", "Unknown")
            results = engine_data.get("results", [])
            if not results:
                continue

            engine_color = self._engine_color(engine_name)

            # Engine header
            self.console.print(Rule(
                f"[bold {engine_color}]{engine_name}[/bold {engine_color}] "
                f"[dim]({len(results)} items)[/dim]",
                style=engine_color,
            ))

            # Results table
            table = Table(
                show_header=True,
                header_style=f"bold {engine_color}",
                box=box.ROUNDED,
                border_style=engine_color,
                expand=True,
            )
            table.add_column("#", width=3, justify="right", style="dim")
            table.add_column("Title", style="bold white", no_wrap=False, ratio=3)
            table.add_column("URL", style="blue", no_wrap=False, ratio=3)
            table.add_column("Snippet", style="dim italic", no_wrap=False, ratio=4)

            for i, item in enumerate(results, 1):
                title = item.get("title", "Untitled")
                url = item.get("url", "")
                snippet = item.get("snippet", "")
                # Terminal table is preview only; light truncation to avoid row overflow.
                # Full text is preserved in JSON/CSV exports.
                if len(snippet) > 800:
                    snippet = snippet[:800] + "..."

                table.add_row(
                    str(i),
                    title,
                    url,
                    snippet,
                )

            self.console.print(table)
            total += len(results)

        # Footer stats
        self.console.print(Rule(f"[bold]{total} total results[/bold]", style="bright_black"))

    # --------------------- Save ---------------------

    def save_json(self, filename: str | None = None) -> str:
        """
        Save results to a local JSON file
        :param filename: Filename; defaults to timestamp-based
        :return: Saved file path
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        filepath = self.output_dir / filename
        output_data = {
            "export_time": datetime.now().isoformat(),
            "total_engines": len(self.cleaned_data),
            "total_results": sum(len(d.get("results", [])) for d in self.cleaned_data),
            "data": self.cleaned_data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.console.print(f"  [green][OK][/green] JSON saved to: [bold]{filepath}[/bold]")
        return str(filepath)

    def save_csv(self, filename: str | None = None) -> str:
        """
        Save results to a local CSV file
        :param filename: Filename; defaults to timestamp-based
        :return: Saved file path
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.csv"

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Engine", "#", "Title", "URL", "Snippet"])
            for engine_data in self.cleaned_data:
                engine_name = engine_data.get("engine", "Unknown")
                for i, item in enumerate(engine_data.get("results", []), 1):
                    writer.writerow([
                        engine_name,
                        i,
                        item.get("title", ""),
                        item.get("url", ""),
                        item.get("snippet", ""),
                    ])

        self.console.print(f"  [green][OK][/green] CSV saved to: [bold]{filepath}[/bold]")
        return str(filepath)

    # --------------------- Main entry ---------------------

    def run(self, extracted_data: list[dict], query: str) -> None:
        """
        Cleaning module main workflow
        1. Receive data
        2. Clean and filter
        3. Display results
        4. Save to local files
        """
        self.receive(extracted_data)
        self.wash(query)
        self.display()
        self.save_json()
        self.save_csv()


# --------------------- Standalone test ---------------------

if __name__ == "__main__":
    # Simulate data from extract_data.py
    mock_data = [
        {
            "engine": "Google",
            "query": "Python tutorial",
            "results": [
                {
                    "title": "Python Official Tutorial",
                    "url": "https://docs.python.org/3/tutorial/",
                    "snippet": "Python is an easy to learn, powerful programming language. Suitable for beginners.",
                },
                {
                    "title": "Learn Python Programming - W3Schools",
                    "url": "https://www.w3schools.com/python/",
                    "snippet": "Well organized and easy to understand Web building tutorials with lots of examples.",
                },
                {
                    "title": "SPONSORED: Best Python Course 2026 - Buy Now & Save 50%",
                    "url": "https://example-ad.com/python?promotion=1&utm_source=search",
                    "snippet": "Limited time offer! Learn Python from pro developers. Click here to enroll today. Money back guarantee!",
                },
                {
                    "title": "Python Tutorial - Real Python",
                    "url": "https://realpython.com/",
                    "snippet": "Learn Python online: Python tutorials for developers of all skill levels, from beginner to expert.",
                },
                {
                    "title": "The Python Tutorial — Python 3 documentation",
                    "url": "https://docs.python.org/3/tutorial/index.html",
                    "snippet": "This tutorial introduces the reader informally to the basic concepts and features of Python.",
                },
            ],
        },
        {
            "engine": "Bing",
            "query": "Python tutorial",
            "results": [
                {
                    "title": "Python Basics | W3Schools",
                    "url": "https://www.w3schools.com/python/",
                    "snippet": "Python basics covering syntax, data types, conditionals and more.",
                },
                {
                    "title": "AD: Top Python Books - Free Shipping Today Only",
                    "url": "https://shop-ad.com/python-book?sponsored=1&gclid=xyz",
                    "snippet": "Sponsored: Complete set of Python learning books with discount code SAVE20. Limited stock available!",
                },
                {
                    "title": "Python Tutorial - LearnPython.org",
                    "url": "https://www.learnpython.org/",
                    "snippet": "Learn Python programming online with interactive tutorials, exercises and examples.",
                },
            ],
        },
    ]

    washer = DataWasher()
    washer.run(mock_data, "Python tutorial")
