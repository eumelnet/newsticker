import os
import time
import json
import asyncio
from datetime import datetime
from typing import Optional

import feedparser
import httpx
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# --- Config ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
CACHE_TTL_SECONDS = 300  # 5 Minuten Cache
ARTICLES_PER_SOURCE = 3  # Max Artikel pro Quelle (verhindert Einseitigkeit)
BATCH_SIZE = 8  # Artikel pro KI-Batch
PAGE_SIZE = 5  # Artikel pro Seite an Frontend

RSS_FEEDS = [
    # Öffentlich-Rechtlich
    "https://www.tagesschau.de/xml/rss2/",
    "https://www.zdf.de/rss/zdf/nachrichten",
    "https://www.deutschlandfunk.de/nachrichten-100.rss",
    # Überregional
    "https://www.zeit.de/news/index",
    "https://rss.dw.com/xml/rss-de-all",
    # Alternative Medien
    "https://www.nius.de/feed",
    "https://www.weltwoche.ch/feed",
    "https://reitschuster.de/feed/",
    "https://www.tichyseinblick.de/feed/",
    "https://www.bild.de/rssfeeds/vw-alles/vw-alles-26970192,sort=1,view=rss2.bild.xml",
    # Wirtschaft
    "https://www.handelsblatt.com/contentexport/feed/top-themen/",
    # Tech & Wissenschaft
    "https://www.heise.de/rss/heise-atom.xml",
    "https://www.golem.de/rss.php?feed=RSS2.0",
    # International (deutschsprachig)
    "https://www.nzz.ch/recent.rss",
    "https://derstandard.at/rss",
]

# --- Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Newsticker")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Zu viele Anfragen. Bitte warte einen Moment."},
    )


# --- Cache ---
class NewsCache:
    def __init__(self):
        self.articles: list = []  # Processed articles
        self.last_update: float = 0
        self.lock = asyncio.Lock()
        self.updating: bool = False

    def is_valid(self) -> bool:
        return len(self.articles) > 0 and (time.time() - self.last_update) < CACHE_TTL_SECONDS

    def get_page(self, page: int) -> list:
        start = page * PAGE_SIZE
        end = start + PAGE_SIZE
        return self.articles[start:end]

    @property
    def total_pages(self) -> int:
        return max(1, -(-len(self.articles) // PAGE_SIZE))  # ceil division

    async def ensure_fresh(self):
        """Trigger background update if cache is stale."""
        if not self.is_valid() and not self.updating:
            asyncio.create_task(self._update())

    async def _update(self):
        async with self.lock:
            if self.is_valid():
                return
            self.updating = True
            try:
                self.articles = await fetch_and_process_news()
                self.last_update = time.time()
            finally:
                self.updating = False


cache = NewsCache()


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
    except Exception:
        return []


async def fetch_rss_feeds() -> list[dict]:
    """Fetch articles from all RSS feeds concurrently."""
    async with httpx.AsyncClient(timeout=15) as client:
        tasks = [fetch_single_feed(client, url) for url in RSS_FEEDS]
        results = await asyncio.gather(*tasks)

    # Flatten and interleave sources for diversity
    all_articles = []
    for feed_articles in results:
        all_articles.extend(feed_articles)
    return all_articles


# --- AI Processing ---
async def process_batch(client: Anthropic, articles: list[dict]) -> list[dict]:
    """Process a batch of articles with Claude."""
    if not articles:
        return []

    articles_text = "\n\n---\n\n".join(
        f"TITEL: {a['title']}\nZUSAMMENFASSUNG: {a['summary']}\nQUELLE: {a['source']}\nLINK: {a['link']}"
        for a in articles
    )

    prompt = f"""Du bist ein Nachrichtenredakteur. Hier sind aktuelle Nachrichtenartikel:

{articles_text}

AUFGABE:
1. ENTFERNE alle Artikel die Spekulation enthalten (Konjunktiv wie "könnte", "dürfte", "würde", Phrasen wie "es wird vermutet", "möglicherweise")
2. ENTFERNE alle Meinungsartikel und Einordnungen
3. ENTFERNE Duplikate (gleiche Nachricht aus verschiedenen Quellen -> nur einmal, bevorzuge die detailliertere Version)
4. Schreibe die verbleibenden Nachrichten MIT EIGENEN WORTEN neu:
   - Neutral und sachlich
   - Positiv gestimmt (ohne zu verfälschen)
   - Kurz und prägnant (max 2 Sätze pro Nachricht)
   - Nur verifizierte Fakten

Antworte als JSON-Array mit Objekten: {{"headline": "...", "text": "...", "source": "...", "link": "..."}}
Gib NUR das JSON-Array zurück, nichts anderes."""

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        return []


async def fetch_and_process_news() -> list:
    """Full pipeline: fetch RSS -> filter with Claude in batches."""
    articles = await fetch_rss_feeds()
    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # Process in batches for faster first results
    all_processed = []
    for i in range(0, len(articles), BATCH_SIZE):
        batch = articles[i:i + BATCH_SIZE]
        processed = await process_batch(client, batch)
        all_processed.extend(processed)

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
    # First request triggers update if needed
    if not cache.is_valid():
        await cache._update()

    news = cache.get_page(page)
    return {
        "news": news,
        "page": page,
        "total_pages": cache.total_pages,
        "has_more": page < cache.total_pages - 1,
        "last_update": datetime.fromtimestamp(cache.last_update).isoformat() if cache.last_update else None,
        "next_update_in": max(0, int(CACHE_TTL_SECONDS - (time.time() - cache.last_update))),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
