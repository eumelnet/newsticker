import os
import time
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import httpx
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# --- Config ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
CACHE_TTL_SECONDS = 300  # 5 Minuten Cache
MAX_ARTICLES_PER_FETCH = 20

RSS_FEEDS = [
    "https://www.tagesschau.de/xml/rss2/",
    "https://www.spiegel.de/schlagzeilen/index.rss",
    "https://www.zeit.de/news/index",
    "https://www.heise.de/rss/heise-atom.xml",
    "https://rss.dw.com/xml/rss-de-all",
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
        self.data: Optional[list] = None
        self.last_update: float = 0
        self.lock = asyncio.Lock()

    def is_valid(self) -> bool:
        return self.data is not None and (time.time() - self.last_update) < CACHE_TTL_SECONDS

    async def get_or_update(self) -> list:
        if self.is_valid():
            return self.data
        async with self.lock:
            # Double-check after acquiring lock
            if self.is_valid():
                return self.data
            self.data = await fetch_and_process_news()
            self.last_update = time.time()
            return self.data


cache = NewsCache()


# --- News Fetching ---
async def fetch_rss_feeds() -> list[dict]:
    """Fetch articles from all RSS feeds."""
    articles = []
    async with httpx.AsyncClient(timeout=10) as client:
        for feed_url in RSS_FEEDS:
            try:
                resp = await client.get(feed_url)
                feed = feedparser.parse(resp.text)
                for entry in feed.entries[:5]:
                    articles.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", entry.get("description", "")),
                        "link": entry.get("link", ""),
                        "source": feed.feed.get("title", feed_url),
                        "published": entry.get("published", ""),
                    })
            except Exception:
                continue
    return articles[:MAX_ARTICLES_PER_FETCH]


# --- AI Processing ---
async def process_with_claude(articles: list[dict]) -> list[dict]:
    """Filter and rewrite articles using Claude."""
    if not articles:
        return []

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    articles_text = "\n\n---\n\n".join(
        f"TITEL: {a['title']}\nZUSAMMENFASSUNG: {a['summary']}\nQUELLE: {a['source']}\nLINK: {a['link']}"
        for a in articles
    )

    prompt = f"""Du bist ein Nachrichtenredakteur. Hier sind aktuelle Nachrichtenartikel:

{articles_text}

AUFGABE:
1. ENTFERNE alle Artikel die Spekulation enthalten (Konjunktiv wie "könnte", "dürfte", "würde", Phrasen wie "es wird vermutet", "möglicherweise")
2. ENTFERNE alle Meinungsartikel und Einordnungen
3. Schreibe die verbleibenden Nachrichten MIT EIGENEN WORTEN neu:
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

    # Parse response
    import json
    try:
        content = response.content[0].text.strip()
        # Handle markdown code blocks
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        return []


async def fetch_and_process_news() -> list:
    """Full pipeline: fetch RSS -> filter with Claude."""
    articles = await fetch_rss_feeds()
    processed = await process_with_claude(articles)
    return processed


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
@limiter.limit("6/minute")
async def get_news(request: Request):
    news = await cache.get_or_update()
    return {
        "news": news,
        "last_update": datetime.fromtimestamp(cache.last_update).isoformat() if cache.last_update else None,
        "next_update_in": max(0, int(CACHE_TTL_SECONDS - (time.time() - cache.last_update))),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
