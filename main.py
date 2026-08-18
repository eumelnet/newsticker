import os
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional

import feedparser
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("newsticker")

# --- Config ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "1800"))  # 30 Minuten
ARTICLES_PER_SOURCE = int(os.getenv("ARTICLES_PER_SOURCE", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "15"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "5"))

RSS_FEEDS = [
    # Öffentlich-Rechtlich
    "https://www.tagesschau.de/xml/rss2/",
    "https://www.zdf.de/rss/zdf/nachrichten",
    "https://www.deutschlandfunk.de/nachrichten-100.rss",
    # Überregional
    "https://www.zeit.de/news/index",
    "https://rss.dw.com/xml/rss-de-all",
    "https://www.welt.de/feeds/latest.rss",
    # Alternative / Meinungsvielfalt
    "https://reitschuster.de/feed/",
    "https://www.tichyseinblick.de/feed/",
    "https://jungefreiheit.de/feed/",
    "https://www.cicero.de/rss.xml",
    "https://www.epochtimes.de/feed",
    "https://www.berliner-zeitung.de/feed.xml",
    # Wirtschaft
    "https://www.handelsblatt.com/contentexport/feed/top-themen/",
    # Tech & Wissenschaft
    "https://www.heise.de/rss/heise-atom.xml",
    "https://www.golem.de/rss.php?feed=RSS2.0",
    # International (deutschsprachig)
    "https://www.nzz.ch/recent.rss",
    "https://derstandard.at/rss",
]


# --- Cache ---
class NewsCache:
    def __init__(self):
        self.articles: list = []
        self.last_update: float = 0
        self.updating: bool = False
        self.content_hash: str = ""
        self._lock = asyncio.Lock()

    def is_stale(self) -> bool:
        return (time.time() - self.last_update) >= CACHE_TTL_SECONDS

    def get_page(self, page: int) -> list:
        start = page * PAGE_SIZE
        return self.articles[start:start + PAGE_SIZE]

    @property
    def total_pages(self) -> int:
        if not self.articles:
            return 0
        return -(-len(self.articles) // PAGE_SIZE)

    async def update(self):
        """Run update in background. Never blocks requests."""
        if self.updating:
            return
        async with self._lock:
            if self.updating or not self.is_stale():
                return
            self.updating = True
        try:
            logger.info("Fetching RSS feeds...")
            articles = await fetch_rss_feeds()

            # Hash-Check: skip Claude if content hasn't changed
            import hashlib
            new_hash = hashlib.md5(
                json.dumps([a["title"] for a in articles], sort_keys=True).encode()
            ).hexdigest()

            if new_hash == self.content_hash and self.articles:
                logger.info("RSS content unchanged, skipping Ollama")
                self.last_update = time.time()
            else:
                logger.info(f"RSS content changed, processing {len(articles)} articles...")
                new_articles = await process_articles_incrementally(articles, self)
                if new_articles:
                    self.articles = new_articles
                    self.content_hash = new_hash
                    self.last_update = time.time()
                    logger.info(f"News updated: {len(new_articles)} articles")
                else:
                    logger.warning("Ollama returned no articles, keeping old data")
                    if self.articles:
                        self.last_update = time.time()
        except Exception as e:
            logger.error(f"News update failed: {e}")
            if self.articles:
                self.last_update = time.time()
        finally:
            self.updating = False


cache = NewsCache()


# --- Background updater ---
async def periodic_update():
    """Runs forever, updates cache every CACHE_TTL_SECONDS."""
    # Initial load on startup
    await cache.update()
    while True:
        await asyncio.sleep(CACHE_TTL_SECONDS)
        await cache.update()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background updater on startup."""
    task = asyncio.create_task(periodic_update())
    yield
    task.cancel()


# --- App ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Newsticker", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Zu viele Anfragen. Bitte warte einen Moment."},
    )


# --- News Fetching ---
async def fetch_single_feed(client: httpx.AsyncClient, feed_url: str) -> list[dict]:
    """Fetch articles from a single RSS feed."""
    try:
        resp = await client.get(feed_url, follow_redirects=True)
        feed = feedparser.parse(resp.text)
        source_name = feed.feed.get("title", feed_url)
        articles = []
        for entry in feed.entries[:ARTICLES_PER_SOURCE]:
            articles.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", entry.get("description", ""))[:500],
                "link": entry.get("link", ""),
                "source": source_name,
                "published": entry.get("published", ""),
            })
        return articles
    except Exception as e:
        logger.warning(f"Feed fetch failed for {feed_url}: {e}")
        return []


async def fetch_rss_feeds() -> list[dict]:
    """Fetch articles from all RSS feeds concurrently."""
    async with httpx.AsyncClient(timeout=15) as client:
        tasks = [fetch_single_feed(client, url) for url in RSS_FEEDS]
        results = await asyncio.gather(*tasks)

    all_articles = []
    for feed_articles in results:
        all_articles.extend(feed_articles)
    return all_articles


# --- AI Processing (Ollama) ---
SYSTEM_PROMPT = """Du bist ein Nachrichtenredakteur. Du bekommst Artikel mit Titel, Zusammenfassung, Quelle und Link.

REGELN:
1. ENTFERNE Artikel mit Spekulation (Konjunktiv: "könnte", "dürfte", "würde", "möglicherweise", "es wird vermutet")
2. ENTFERNE Meinungsartikel und Einordnungen
3. ENTFERNE Duplikate (gleiche Nachricht nur einmal)
4. Schreibe verbleibende Nachrichten MIT EIGENEN WORTEN neu: neutral, sachlich, positiv gestimmt, max 2 Sätze
5. BEHALTE die originale Quelle und den originalen Link bei!

WICHTIG: Antworte NUR mit einem JSON-Array. Kein anderer Text. Kein Markdown. Format:
[{"headline": "Kurze Überschrift", "text": "Ein bis zwei Sätze.", "source": "Originalquelle", "link": "https://original-link"}]"""


async def process_batch(http_client: httpx.AsyncClient, articles: list[dict]) -> list[dict]:
    """Process a batch of articles with Ollama."""
    if not articles:
        return []

    articles_text = "\n\n".join(
        f"Artikel {i+1}:\nTitel: {a['title']}\nZusammenfassung: {a['summary']}\nQuelle: {a['source']}\nLink: {a['link']}"
        for i, a in enumerate(articles)
    )

    try:
        response = await http_client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": articles_text},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 4000},
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"].strip()

        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        # Parse JSON
        parsed = json.loads(content)

        # Handle wrapped objects like {"result": [...]} or {"articles": [...]}
        if isinstance(parsed, dict):
            for key in parsed:
                if isinstance(parsed[key], list):
                    return parsed[key]
            return []
        return parsed if isinstance(parsed, list) else []
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Ollama batch processing failed: {e}")
        return []


async def process_articles_incrementally(articles: list[dict], cache_ref) -> list:
    """Process articles batch by batch, updating cache after each batch."""
    all_processed = []
    async with httpx.AsyncClient() as http_client:
        for i in range(0, len(articles), BATCH_SIZE):
            batch = articles[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = -(-len(articles) // BATCH_SIZE)
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} articles)...")

            processed = await process_batch(http_client, batch)
            all_processed.extend(processed)
            logger.info(f"Batch {batch_num} done: {len(processed)} articles kept")

            # Update cache incrementally so frontend can show results immediately
            if all_processed:
                cache_ref.articles = all_processed
                cache_ref.last_update = time.time()

    return all_processed


# --- Routes ---
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def index(request: Request):
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/news")
@limiter.limit("20/minute")
async def get_news(request: Request, page: int = Query(default=0, ge=0)):
    news = cache.get_page(page)
    return {
        "news": news,
        "page": page,
        "total_pages": cache.total_pages,
        "has_more": page < cache.total_pages - 1 if cache.total_pages > 0 else False,
        "updating": cache.updating,
        "last_update": datetime.fromtimestamp(cache.last_update).isoformat() if cache.last_update else None,
        "next_update_in": max(0, int(CACHE_TTL_SECONDS - (time.time() - cache.last_update))) if cache.last_update else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
