"""
extract_data.py — Data extraction module
Dual-engine extraction: bs4 (CSS selectors) + sklearn (text pattern recognition)
Prevents extraction breakage when target websites update CSS
Receives crawl results from request_webs.py → extracts → hands off to wash_data.py
Optimized for English / international search engines
"""

import re
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup, Tag
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn


class DataExtractor:
    """HTML data extractor — extracts structured data from search engine result pages"""

    # ============ CSS selector strategies per search engine ============
    # Each strategy: { container, title, url, snippet }
    ENGINE_SELECTORS: dict[str, dict] = {
        "Google": {
            "container": "div.g, div[data-sokoban-container], div.MjjYud, "
                         "div[data-hveid], div.Gx5Zad, div.tF2Cxc",
            "title":     "h3, h3.LC20lb, div.vvjwJb",
            "url":       "a[jsname], div.yuRUbf a, a[data-ved], a[ping]",
            "snippet":   "div.VwiC3b, span.aCOpRe, div[data-sncf], span.st, "
                         "div.IsZvec, div.lEBKkf",
        },
        "Bing": {
            "container": "li.b_algo, li.b_ans, ol#b_results > li, "
                         "div.b_algo, div.b_title",
            "title":     "h2 a, div.b_tpcn a, a[target='_blank']",
            "url":       "h2 a, div.b_tpcn a",
            "snippet":   "div.b_caption p, p.b_lineclamp2, p.b_lineclamp4, "
                         "div.b_snippet p, div.b_caption span",
        },
        "Yahoo": {
            "container": "div.dd.algo, div.algo-sr, div.compArticleList li, "
                         "div[data-testid='web-result'], ol.searchCenterMiddle li",
            "title":     "h3 a, h3.title a, a.ac-algo",
            "url":       "h3 a, h3.title a, a.ac-algo",
            "snippet":   "div.compText, p.s-desc, div.dd, span.fc-falcon",
        },
        "DuckDuckGo": {
            "container": "li[data-layout='organic'], article[data-testid='result'], "
                         "div.result.results_links, div.nrn-react-div",
            "title":     "h2 a, a[data-testid='result-title-a'], a.result__a",
            "url":       "h2 a, a[data-testid='result-title-a'], a.result__a",
            "snippet":   "span[data-testid='result-snippet'], "
                         "div[data-result='snippet'], "
                         "span.line-clamp-3, div.result__snippet",
        },
        "Brave Search": {
            "container": "div.snippet, div.result, div[data-type='web'], "
                         "div.fdb.fragment",
            "title":     "a.snippet-title, a.heading-serpresult, h3 a, a[class*='title']",
            "url":       "a.snippet-title, a.heading-serpresult, h3 a",
            "snippet":   "div.snippet-description, div.snippet-content, "
                         "p.snippet-description, span.snippet-description",
        },
        "Ecosia": {
            "container": "div.result, div.result__body, div.card-web, "
                         "div[data-test-id='mainline-result-web']",
            "title":     "h2.result__title a, a.result__link, a[data-testid='result-title-a']",
            "url":       "h2.result__title a, a.result__link",
            "snippet":   "p.result__snippet, div.result__snippet, "
                         "span[data-testid='result-snippet']",
        },
        "Startpage": {
            "container": "div.result, div.w-gl__result, div.result-item, "
                         "a[class*='w-gl']",
            "title":     "h3 a, a.result-title, a[class*='title']",
            "url":       "h3 a, a.result-title",
            "snippet":   "p.result-description, div.result-description, "
                         "p.desc, span[class*='desc']",
        },
        "Mojeek": {
            "container": "li.results-standard, div.result, ul.results li",
            "title":     "h2 a, a.title, h3 a",
            "url":       "h2 a, a.title, h3 a",
            "snippet":   "p.snippet, div.snippet, p.desc, span.s",
        },
        "Swisscows": {
            "container": "div.result, div.item, div.web-result",
            "title":     "h2 a, a.title, h3 a",
            "url":       "h2 a, a.title, h3 a",
            "snippet":   "p.description, div.description, p.desc, span.snippet",
        },
        "Yandex": {
            "container": "li.serp-item, div.serp-item",
            "title":     "h2 a, a.link_theme_outer, a.OrganicTitle-Link",
            "url":       "h2 a, a.link_theme_outer",
            "snippet":   "div.text-container, span.extended-text__short, "
                         "div.OrganicTextContentWrapper",
        },
        "Qwant": {
            "container": "div[data-testid='result'], div.result",
            "title":     "a[data-testid='result-title'], h3 a",
            "url":       "a[data-testid='result-title'], h3 a",
            "snippet":   "span[data-testid='result-description'], p.desc",
        },
        "Perplexity.ai": {
            "container": "div[class*='result'], div[class*='search-result'], "
                         "a[class*='result']",
            "title":     "h3, span[class*='title'], a[class*='title']",
            "url":       "a[href*='http'], a[class*='source']",
            "snippet":   "p, span[class*='snippet'], div[class*='description']",
        },
        "You.com": {
            "container": "div[data-testid='result'], div.result, "
                         "li[data-testid='search-result']",
            "title":     "h3 a, a[class*='title'], a[data-testid='result-title']",
            "url":       "h3 a, a[class*='title']",
            "snippet":   "p, div[class*='snippet'], div[class*='description']",
        },
        "Yep.com": {
            "container": "div.result, div[class*='Result'], article",
            "title":     "h3 a, a[class*='title'], h2 a",
            "url":       "h3 a, a[class*='title'], h2 a",
            "snippet":   "p, div[class*='desc'], div[class*='snippet']",
        },
        "Kagi": {
            "container": "div.result-item, div.search-result, div[class*='_result']",
            "title":     "h3 a, a.result-title, a[class*='title']",
            "url":       "h3 a, a.result-title",
            "snippet":   "p.result-snippet, div.result-snippet, "
                         "div[class*='snippet'], div[class*='desc']",
        },
        "ChatGPT Search": {
            "container": "div[class*='result'], a[class*='citation'], "
                         "div[class*='source']",
            "title":     "span[class*='title'], div[class*='title']",
            "url":       "a[href*='http'], a[class*='citation']",
            "snippet":   "p, span[class*='snippet'], div[class*='text']",
        },
        "Sogou": {
            "container": "div.results, div.vrwrap, div.rb, div[class*='result']",
            "title":     "h3 a, h3.vr-title a, a[class*='title']",
            "url":       "h3 a, h3.vr-title a",
            "snippet":   "div.str-text, div.space-txt, p.str_info, "
                         "div[class*='abstract'], div[class*='summary']",
        },
        "Naver": {
            "container": "li.bx, div.total_wrap, ul.lst_total li, "
                         "div[class*='total']",
            "title":     "a.api_txt_lines, a.title_link, a[class*='title']",
            "url":       "a.api_txt_lines, a.title_link",
            "snippet":   "div.api_txt_lines, div.dsc, div.total_dsc, "
                         "p[class*='dsc'], span[class*='desc']",
        },
        "Baidu": {
            "container": "div.result, div.c-container, div.result-op",
            "title":     "h3 a, h3[class] a",
            "url":       "h3 a",
            "snippet":   "span.content-right_8Zs40, div.c-abstract, span.c-abstract, "
                         "div.c-span-last, div.c-row",
        },
        "Wikipedia": {
            "container": "li.mw-search-result, div.searchresults li, "
                         "div.mw-search-result-heading",
            "title":     "a[data-serp-pos], div.mw-search-result-heading a",
            "url":       "a[data-serp-pos], div.mw-search-result-heading a",
            "snippet":   "div.searchresult, div.mw-search-result-data, "
                         "span.searchmatch",
        },
    }

    def __init__(self):
        """Initialize the extractor"""
        self.raw_data: list[dict] = []
        self.extracted_data: list[dict] = []
        self.console = Console()

    # ==================== Receive data ====================

    def receive(self, raw_data: list[dict]) -> None:
        """
        Receive raw crawl data from request_webs.py
        :param raw_data: [{"engine": "Google", "url": "...", "query": "...", "html": "..."}, ...]
        """
        self.raw_data = raw_data
        self.extracted_data = []

    # ==================== Basic extraction utilities ====================

    @staticmethod
    def _make_soup(html: str) -> BeautifulSoup:
        """Create a BeautifulSoup object from HTML"""
        return BeautifulSoup(html, "html.parser")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text: remove excess whitespace and control characters"""
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\u3000", " ")  # full-width space
        return text.strip()

    @staticmethod
    def _resolve_url(href: str, base_url: str) -> str:
        """Convert relative links to absolute URLs"""
        if not href:
            return ""
        return urljoin(base_url, href)

    @staticmethod
    def _clean_title(title: str) -> str:
        """
        Clean common noise from titles
        - Remove leading domain/URL fragments
        - Remove trailing breadcrumb paths (e.g.  › article › details)
        - Remove HTML entities
        - Remove common prefix noise (e.g. "AD:", "Sponsored:", "Promoted:")
        """
        if not title:
            return title

        # Remove HTML entities
        import html as html_mod
        title = html_mod.unescape(title)

        # Remove ad/sponsored prefixes
        title = re.sub(
            r'^(ad\s*[:\-—–]\s*|sponsored\s*[:\-—–]\s*|promoted\s*[:\-—–]\s*'
            r'|advertisement\s*[:\-—–]\s*|paid\s+result\s*[:\-—–]\s*)',
            '', title, flags=re.IGNORECASE
        )

        # Remove leading URL fragments (e.g. "csdn.nethttps://blog.csdn.net...")
        cleaned = re.sub(
            r'^[\w.-]+\.[a-z]{2,}(?=https?://)',
            '', title
        )

        # Remove trailing breadcrumb paths
        breadcrumb = r'(?:\s*[›»>]\s*[\w\s\-.#@&?=%!+]+)+$'
        cleaned = re.sub(breadcrumb, '', cleaned)

        cleaned = cleaned.strip()

        # If result is empty after cleaning or starts with http (citation link),
        # try extracting domain as fallback label
        if not cleaned:
            return title.strip()
        if cleaned.startswith("http"):
            parsed = urlparse(cleaned)
            if parsed.netloc:
                return parsed.netloc
            return cleaned

        return cleaned

    def extract_text(self, html: str) -> str:
        """Extract plain text from HTML"""
        soup = self._make_soup(html)
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return self._clean_text(soup.get_text())

    def extract_links(self, html: str, base_url: str = "") -> list[str]:
        """Extract all links from the page"""
        soup = self._make_soup(html)
        links = []
        for a in soup.find_all("a", href=True):
            url = self._resolve_url(a.get("href", ""), base_url)
            if url and url.startswith(("http://", "https://")):
                links.append(url)
        return links

    def extract_titles(self, html: str) -> list[str]:
        """Extract all titles (h1-h6) from the page"""
        soup = self._make_soup(html)
        return [self._clean_text(tag.get_text()) for tag in soup.find_all(["h1", "h2", "h3", "h4"])
                if self._clean_text(tag.get_text())]

    def extract_snippets(self, html: str) -> list[str]:
        """Extract all paragraph/summary texts from the page"""
        soup = self._make_soup(html)
        snippets = []
        for tag in soup.find_all(["p", "span", "div"]):
            text = self._clean_text(tag.get_text())
            if 10 < len(text) < 500:
                parent_classes = " ".join(tag.get("class", [])) if tag.get("class") else ""
                if any(kw in parent_classes.lower() for kw in ("nav", "footer", "header", "menu")):
                    continue
                snippets.append(text)
        return snippets

    # ==================== Engine-specific extraction ====================

    def _extract_engine_results(
        self, html: str, engine_name: str, base_url: str
    ) -> list[dict]:
        """
        Extract search results using engine-specific selectors
        :return: [{"title": "...", "url": "...", "snippet": "..."}, ...]
        """
        soup = self._make_soup(html)

        # Remove noise elements (cite breadcrumbs, time timestamps, etc.)
        for tag in soup(["cite", "time", "small"]):
            tag.decompose()

        selectors = self.ENGINE_SELECTORS.get(engine_name)

        if not selectors:
            # No dedicated selectors → fall back to generic extraction
            return self._extract_generic_results(soup, base_url)

        containers = soup.select(selectors["container"])
        if not containers:
            # Selectors didn't match → fall back to generic extraction
            return self._extract_generic_results(soup, base_url)

        results = []
        seen_urls = set()

        for container in containers:
            # Extract title — try multiple selectors
            title = ""
            title_tag = None
            for ts in selectors["title"].split(", "):
                title_tag = container.select_one(ts.strip())
                if title_tag and title_tag.get_text(strip=True):
                    break
            if title_tag:
                title = self._clean_text(title_tag.get_text())
                title = self._clean_title(title)

            # Extract link — try multiple selectors
            url = ""
            for us in selectors["url"].split(", "):
                url_tag = container.select_one(us.strip())
                if url_tag and url_tag.get("href"):
                    url = self._resolve_url(url_tag["href"], base_url)
                    break

            # If no URL found via dedicated selectors, try any link in container
            if not url:
                for a in container.find_all("a", href=True):
                    href = a.get("href", "")
                    url = self._resolve_url(href, base_url)
                    if url.startswith(("http://", "https://")):
                        break

            # Extract snippet — try multiple selectors
            snippet = ""
            for ss in selectors["snippet"].split(", "):
                snippet_tag = container.select_one(ss.strip())
                if snippet_tag:
                    snippet = self._clean_text(snippet_tag.get_text())
                    break

            # Require at least title or URL for a valid result
            if not title and not url:
                continue

            # Deduplicate
            if url and url in seen_urls:
                continue
            seen_urls.add(url)

            results.append({"title": title, "url": url, "snippet": snippet})

        return results

    @staticmethod
    def _extract_generic_results(soup: BeautifulSoup, base_url: str) -> list[dict]:
        """
        Generic extraction method (used when no dedicated selectors exist)
        Extracts all linked text blocks from the page
        """
        # Remove noise elements
        for tag in soup(["cite", "time", "small", "script", "style", "noscript"]):
            tag.decompose()

        results = []
        seen_urls = set()

        # Strategy 1: find containers with h3/a + paragraph
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
            title = DataExtractor._clean_title(title)

            # Look for snippet text in sibling or parent
            p_tag = tag.find("p") or tag.find("span")
            snippet = DataExtractor._clean_text(p_tag.get_text()) if p_tag else ""

            # Skip too-short titles (likely footer/nav links)
            if len(title) < 3:
                continue

            results.append({"title": title, "url": url, "snippet": snippet})

        # Strategy 2: if nothing found, simply extract all links
        if not results:
            for a in soup.find_all("a", href=True):
                url = urljoin(base_url, a["href"])
                if not url.startswith(("http://", "https://")):
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                title = DataExtractor._clean_text(a.get_text())
                title = DataExtractor._clean_title(title)
                if len(title) < 3:
                    continue
                results.append({"title": title, "url": url, "snippet": ""})

        return results[:50]  # cap at 50 results

    # ==================== sklearn text pattern extraction ====================

    @staticmethod
    def _find_nearby_text(a_tag: Tag, max_distance: int = 4) -> str:
        """
        Find descriptive text near a link (no CSS class dependency)
        Walk up parent nodes, find sufficiently long non-link text
        :param a_tag: Target <a> tag
        :param max_distance: Maximum levels to walk up
        :return: Found nearby text
        """
        link_text = DataExtractor._clean_text(a_tag.get_text())

        # Strategy 1: search parent nodes, exclude the link's own text
        parent = a_tag.parent
        for _ in range(max_distance):
            if parent is None:
                break
            full_text = DataExtractor._clean_text(parent.get_text())
            remaining = full_text
            if link_text and link_text in remaining:
                remaining = remaining.replace(link_text, "", 1).strip()
            if len(remaining) > 15:
                return remaining[:300]
            parent = parent.parent

        # Strategy 2: check next sibling
        sibling = a_tag.find_next_sibling()
        if sibling:
            text = DataExtractor._clean_text(sibling.get_text())
            if len(text) > 10:
                return text[:300]

        # Strategy 3: check parent's next sibling
        if a_tag.parent:
            parent_sibling = a_tag.parent.find_next_sibling()
            if parent_sibling:
                text = DataExtractor._clean_text(parent_sibling.get_text())
                if len(text) > 10:
                    return text[:300]

        return ""

    # Noise URL feature keywords (internal redirects, tracking, utility pages)
    NOISE_URL_PATTERNS: list[str] = [
        # Search engine internal redirect / routing links
        "baidu.php?url=",            # Baidu ad links
        "/search?", "/search/",      # in-site search links
        "/s?wd=", "/s?keyword=",     # search suggestion links
        # Navigation / utility / footer
        "/duty/", "/legal", "beian.miit.gov.cn",
        "/sitemap", "/careers", "/jobs",
        # Social media profiles
        "/profile/", "/user/", "/users/",
        "profile.zjurl.cn",         # Toutiao user page
        # Tracking / ad redirects
        "googleadservices.com",
        "doubleclick.net",
        "googlesyndication.com",
        "/aclk", "/pagead/",
    ]

    @staticmethod
    def _is_result_like(item: dict, query: str, score: float) -> bool:
        """
        Judge whether a candidate looks like a valid search result
        Combines text features and query relevance for assessment
        English-optimized noise word list
        """
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        combined = (title + " " + snippet).lower()

        # Filter conditions
        if len(title) < 3:
            return False

        # URL feature filtering: internal redirects, navigation links
        if url:
            for pat in DataExtractor.NOISE_URL_PATTERNS:
                if pat in url.lower():
                    if "baidu.php?url=" in url.lower():
                        return False
                    if "profile.zjurl.cn" in url.lower() or "/profile/" in url.lower():
                        return False
                    if any(kw in url.lower() for kw in [
                        "googleadservices", "doubleclick", "googlesyndication",
                        "/aclk", "/pagead/"
                    ]):
                        return False

        # Exclude obvious non-result content (English-optimized)
        noise_words = [
            # Pagination / navigation
            "next page", "previous page", "more results", "home page",
            "login", "sign up", "sign in", "register", "settings",
            "page 1", "page 2", "page 3",
            # Legal / policy
            "about us", "contact us", "privacy policy", "terms of service",
            "terms and conditions", "help center", "cookie policy",
            "legal notice", "site map", "accessibility statement",
            "community guidelines", "acceptable use",
            # Corporate
            "press releases", "investor relations", "careers at",
            "job openings", "our offices", "leadership team",
            "board of directors", "annual report", "corporate social",
            # Site navigation
            "advertising on", "advertise with us", "media kit",
            "content policy", "copyright notice", "all rights reserved",
            "© 202", "©202",
            # Social / sharing
            "share on facebook", "share on twitter", "share on linkedin",
            "follow us on", "subscribe to our", "newsletter signup",
            # Cookie / GDPR
            "cookie settings", "manage cookies", "cookie preferences",
            "gdpr", "ccpa", "do not sell my",
            # Generic noise
            "loading", "please wait", "redirecting",
            "javascript is disabled", "enable javascript",
            "browser not supported", "update your browser",
            # Ad / sponsored markers
            "advertisement:", "sponsored by:", "promoted by:",
            "paid content:", "partner content:",
            # Feedback / reporting
            "report abuse", "give feedback", "send feedback",
            "rate this page", "was this helpful",
            # Technical
            "http 404", "page not found", "error occurred",
            "try again later", "temporarily unavailable",
        ]
        for nw in noise_words:
            if nw in combined:
                return False

        # Exact short-word exclusion (only when title exactly matches)
        exact_noise = {
            "stackoverflow",            # keep as meta reference only if title alone
            "github - search results",
            "search results - google",
            "home", "about", "contact",
            "sign in", "log in", "register",
            "cookies", "privacy", "terms",
        }
        title_lower = title.strip().lower()
        if title_lower in exact_noise:
            return False

        # Pagination detection: "Page 1", "Page 2", etc.
        if re.search(r"^page\s*\d+$", title_lower):
            return False

        # Very short snippets need minimum relevance
        if len(snippet) < 8 and score < 0.1:
            return False

        return True

    def _extract_sklearn_results(
        self, html: str, base_url: str, query: str
    ) -> list[dict]:
        """
        sklearn text pattern extraction (no CSS selector dependency)
        1. Extract all linked candidate text blocks from HTML
        2. Vectorize candidates using TF-IDF, compute cosine similarity with query
        3. Filter low-relevance candidates, sort by similarity
        :param html: Raw HTML
        :param base_url: Page base URL
        :param query: Search keyword
        :return: [{"title": "...", "url": "...", "snippet": "..."}, ...]
        """
        soup = self._make_soup(html)

        # Remove noise elements
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header",
                         "cite", "time", "small"]):
            tag.decompose()

        # Step 1: Extract all candidate blocks
        candidates: list[dict] = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "")
            url = self._resolve_url(href, base_url)

            # Keep only valid external links
            if not url.startswith(("http://", "https://")):
                continue
            # Exclude in-page anchors and self-referencing links
            parsed_base = urlparse(base_url)
            parsed_url = urlparse(url)
            if parsed_url.netloc == parsed_base.netloc:
                continue

            title = self._clean_text(a_tag.get_text())
            title = self._clean_title(title)
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

        # Step 2: TF-IDF vectorization + similarity scoring
        candidate_texts = [f"{c['title']} {c['snippet']}" for c in candidates]

        try:
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                max_features=1000,
                sublinear_tf=True,
                stop_words="english",
            )
            all_vectors = vectorizer.fit_transform([query] + candidate_texts)
            query_vec = all_vectors[0:1]
            candidate_vecs = all_vectors[1:]
            scores = cosine_similarity(query_vec, candidate_vecs).flatten()
        except ValueError:
            # Too few candidates — fallback
            scores = np.ones(len(candidates)) * 0.5

        # Step 3: Filter + deduplicate + sort
        results: list[dict] = []
        seen_urls: set[str] = set()

        # Sort by similarity descending
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

    # ==================== Dual-path merge ====================

    @staticmethod
    def _merge_dual_results(
        bs4_results: list[dict],
        sklearn_results: list[dict],
    ) -> list[dict]:
        """
        Merge bs4 and sklearn extraction results
        - Same URL → keep the one with the longer snippet
        - Only one path hit → include directly
        :return: Deduplicated merged result list
        """
        merged: dict[str, dict] = {}  # key=url or title, value=best result

        for r in bs4_results:
            key = r.get("url") or r.get("title", "")
            if key:
                merged[key] = r

        for r in sklearn_results:
            key = r.get("url") or r.get("title", "")
            if not key:
                continue
            if key in merged:
                # Keep the one with the richer snippet
                existing_len = len(merged[key].get("snippet", ""))
                new_len = len(r.get("snippet", ""))
                if new_len > existing_len:
                    merged[key] = r
            else:
                merged[key] = r

        return list(merged.values())

    # ==================== Post-extraction filtering ====================

    def _post_filter_results(
        self, results: list[dict], engine: str, query: str
    ) -> list[dict]:
        """
        Global post-extraction filter: clean titles, resolve links, relevance filter
        :param results: Merged result list
        :param engine: Engine name
        :param query: Search keyword
        :return: Filtered results
        """
        if not results:
            return results

        filtered: list[dict] = []
        seen_urls: set[str] = set()

        for item in results:
            # 1. Clean title (removes ad prefixes, breadcrumbs, etc.)
            item["title"] = self._clean_title(item.get("title", ""))

            # 2. URL deduplication
            url = item.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)

            # 3. Title quality check
            title = item.get("title", "")
            if len(title) < 3:
                continue

            # 4. URL quality check
            if url:
                # Exclude pure anchor / javascript links
                if url.startswith("#") or url.startswith("javascript:"):
                    continue
                # Exclude ad/tracker redirect URLs
                if any(kw in url.lower() for kw in [
                    "baidu.php?url=", "googleadservices",
                    "doubleclick", "/aclk", "/pagead/",
                ]):
                    continue
                # Exclude known low-quality domains
                parsed = urlparse(url)
                low_quality_domains = {
                    "beian.miit.gov.cn", "profile.zjurl.cn",
                }
                if parsed.netloc in low_quality_domains:
                    continue

            filtered.append(item)

        # 5. Secondary relevance filtering (TF-IDF, stricter for non-search-engine sites)
        if len(filtered) > 5 and query:
            filtered = self._filter_by_relevance(filtered, query, engine)

        return filtered

    def _filter_by_relevance(
        self, results: list[dict], query: str, engine: str
    ) -> list[dict]:
        """
        Secondary relevance filtering using TF-IDF
        Non-major engines (e.g. social media, niche sites) use a higher threshold
        """
        # Major search engines — lower threshold (they return relevant results)
        major_engines = {
            "Google", "Bing", "DuckDuckGo", "Brave Search",
            "Ecosia", "Startpage", "Mojeek", "Yahoo",
            "Yandex", "Qwant", "Swisscows", "Yep.com",
            "Sogou", "Baidu", "Naver",
        }
        # AI-powered / meta search — medium threshold
        ai_engines = {
            "Perplexity.ai", "You.com", "Kagi", "ChatGPT Search",
            "Google AI Mode",
        }
        is_major = engine in major_engines
        is_ai = engine in ai_engines

        if is_major:
            min_score = 0.02
        elif is_ai:
            min_score = 0.04
        else:
            # Non-search-engine sites (social media, encyclopedias, etc.) — stricter
            min_score = 0.08

        texts = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in results]

        try:
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=800,
                sublinear_tf=True,
                stop_words="english",
            )
            all_vecs = vectorizer.fit_transform([query] + texts)
            query_vec = all_vecs[0:1]
            scores = cosine_similarity(query_vec, all_vecs[1:]).flatten()
        except ValueError:
            return results

        # Non-major, non-AI engine: if all results have extremely low relevance, discard entirely
        if not is_major and not is_ai and np.max(scores) < 0.05:
            return []

        # If more than 10 results, filter out the lowest-relevance ones
        if len(results) > 10:
            scored = list(zip(results, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [r for r, s in scored if s >= min_score][:50]

        return [r for r, s in zip(results, scores) if s >= min_score]

    # ==================== Batch processing ====================

    def process_all(self) -> list[dict]:
        """
        Structured extraction for all raw data (dual-engine: bs4 + sklearn)
        :return: [
            {"engine": "Google", "query": "...", "results": [...]},
            ...
        ]
        """
        if not self.raw_data:
            self.console.print(Panel("[yellow]No raw data to extract[/yellow]", border_style="yellow"))
            return []

        self.console.print(Panel.fit(
            f"[bold cyan]Starting dual-engine extraction[/bold cyan] "
            f"(bs4 CSS selectors + sklearn text patterns)\n"
            f"Processing [yellow]{len(self.raw_data)}[/yellow] engine results",
            border_style="cyan",
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("[cyan]Extracting...", total=len(self.raw_data))

            for item in self.raw_data:
                engine = item.get("engine", "Unknown")
                html = item.get("html", "")
                base_url = item.get("url", "")
                query = item.get("query", "")

                if not html:
                    progress.advance(task)
                    continue

                # Path 1: bs4 CSS selector extraction
                bs4_results = self._extract_engine_results(html, engine, base_url)

                # Path 2: sklearn text pattern extraction
                sklearn_results = self._extract_sklearn_results(html, base_url, query)

                # Merge dual-path results
                merged_results = self._merge_dual_results(bs4_results, sklearn_results)

                # Post-extraction global filter (clean titles, resolve links, relevance)
                merged_results = self._post_filter_results(
                    merged_results, engine, query
                )

                self.extracted_data.append({
                    "engine": engine,
                    "query": query,
                    "results": merged_results,
                })
                progress.advance(task)

        # Statistics
        total_results = sum(len(d["results"]) for d in self.extracted_data)
        self.console.print(Panel(
            f"[green]Extraction complete[/green] — "
            f"[cyan]{len(self.extracted_data)}[/cyan] engines, "
            f"[cyan]{total_results}[/cyan] total results",
            border_style="green",
        ))

        return self.extracted_data

    def export(self) -> list[dict]:
        """Export extraction results for wash_data.py"""
        return self.extracted_data

    def run(self, raw_data: list[dict]) -> list[dict]:
        """
        Extraction module main workflow
        1. Receive raw data
        2. Batch extract
        3. Export
        """
        self.receive(raw_data)
        return self.process_all()


# ==================== Standalone test ====================

if __name__ == "__main__":
    # Simulate HTML returned by different search engines
    mock_google_html = """
    <html><body>
    <div class="g">
        <div><h3>Python Tutorial - W3Schools</h3></div>
        <div class="yuRUbf"><a href="https://www.w3schools.com/python/">https://www.w3schools.com/python/</a></div>
        <div class="VwiC3b">Well organized and easy to understand Web building tutorials with lots of examples.</div>
    </div>
    <div class="g">
        <h3>Welcome to Python.org</h3>
        <div class="yuRUbf"><a href="https://www.python.org/">https://www.python.org/</a></div>
        <div class="VwiC3b">The official home of the Python Programming Language.</div>
    </div>
    <div class="g">
        <h3>SPONSORED: Learn Python Fast - Best Course 2026</h3>
        <div class="yuRUbf"><a href="https://ad-example.com/python?gclid=xyz">https://ad-example.com/python</a></div>
        <div class="VwiC3b">Paid advertisement: Limited offer, sign up today!</div>
    </div>
    </body></html>
    """

    mock_bing_html = """
    <html><body>
    <ol id="b_results">
    <li class="b_algo">
        <h2><a href="https://www.w3schools.com/python/">Python Basics | W3Schools</a></h2>
        <div class="b_caption"><p>Python basics covering syntax, data types and more.</p></div>
    </li>
    <li class="b_algo">
        <h2><a href="https://realpython.com/">Python Tutorial - Real Python</a></h2>
        <div class="b_caption"><p>Learn Python online: Python tutorials for developers of all skill levels.</p></div>
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
        {"engine": "Google", "url": "https://www.google.com/search?q=Python",
         "query": "Python tutorial", "html": mock_google_html},
        {"engine": "Bing", "url": "https://www.bing.com/search?q=Python",
         "query": "Python tutorial", "html": mock_bing_html},
        {"engine": "Example Site", "url": "https://example.com",
         "query": "test", "html": mock_generic_html},
    ]

    extractor = DataExtractor()
    results = extractor.run(raw_data)

    print("\nExtraction results preview:")
    for engine_data in results:
        print(f"\n[{engine_data['engine']}] ({len(engine_data['results'])} items)")
        for i, item in enumerate(engine_data["results"], 1):
            print(f"  {i}. {item['title']}")
            print(f"     {item['url']}")
            if item["snippet"]:
                print(f"     {item['snippet'][:80]}")
