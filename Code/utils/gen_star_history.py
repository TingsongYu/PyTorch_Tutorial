#!/usr/bin/env python3
"""Render this repo's star history as a PNG image.

Fetches stargazer timestamps from the GitHub REST API, drops everything
before the repository creation date (or an optional --start-date), and
draws a cumulative "stars over time" chart with a gradient fill.
Output: Data/star-history.png

Usage:
    python Code/utils/gen_star_history.py [--repo owner/name] [--refresh]
                                          [--start-date YYYY-MM-DD] [--out-dir DIR]

By default the chart starts at the repository's creation date; use
--start-date to override it.

Auth: set GITHUB_TOKEN (or GH_TOKEN, or have an authenticated `gh` CLI).
Unauthenticated requests work too but are rate-limited to 60/hour
(~1 request per 100 stars). Timestamps are cached next to this script so
style tweaks don't re-hit the API; pass --refresh to re-fetch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.ticker import FuncFormatter

REPO = "TingsongYu/PyTorch_Tutorial"
CACHE = Path(__file__).with_name(".star-history-cache.json")

ACCENT = "#f5a623"  # warm amber, reads well on both light and dark

THEME = dict(bg="#ffffff", text="#1f2328", subtext="#6a737d", grid="#dfe3e8")

# Upper bound on x-axis labels. The real guarantee comes from measuring the
# rendered labels (see thin_xticklabels); this just keeps the tick step sane.
MAX_XTICKS = 12
DAY_STEPS = (1, 2, 3, 7, 14)  # days between ticks
MONTH_STEPS = (1, 2, 3, 6)
YEAR_STEPS = (1, 2, 5, 10)


def get_token() -> str | None:
    """Get GitHub token from environment or gh CLI."""
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if token := os.environ.get(var, "").strip():
            return token
    try:
        out = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def get_json(url: str, headers: dict, retries: int = 4) -> Any:
    """Fetch JSON from url with retry on failure."""
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"request failed ({exc}); retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return []  # unreachable


def fetch_repo_created_at(repo: str) -> datetime:
    """Return the repository creation timestamp (UTC)."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "gen-star-history",
    }
    if token := get_token():
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}"
    data = get_json(url, headers)
    created_at = data.get("created_at")
    if not created_at:
        raise ValueError(f"could not find created_at for repo {repo!r}")
    return datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


def fetch_starred_at(repo: str, refresh: bool) -> list[str]:
    """Return sorted ISO-8601 UTC timestamps of every star event."""
    if CACHE.exists() and not refresh:
        print(f"using cached stargazers from {CACHE}", file=sys.stderr)
        return json.loads(CACHE.read_text())

    headers = {
        "Accept": "application/vnd.github.star+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "gen-star-history",
    }
    if token := get_token():
        headers["Authorization"] = f"Bearer {token}"

    starred: list[str] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/stargazers?per_page=100&page={page}"
        data = get_json(url, headers)
        if not data:
            break
        starred.extend(item["starred_at"] for item in data)
        print(f"\rfetched {len(starred)} stargazers...", end="", file=sys.stderr)
        page += 1
    print(file=sys.stderr)

    starred.sort()
    CACHE.write_text(json.dumps(starred))
    return starred


def build_series(starred: list[str], start: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative star count per star event, cropped to `start` (UTC)."""
    times = [
        datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        for s in starred
    ]
    base = sum(1 for t in times if t < start)
    times = [t for t in times if t >= start]
    # Anchor the line at the start date so the curve begins at the axis edge.
    x = [mdates.date2num(start)] + [mdates.date2num(t) for t in times]
    y = [base] + [base + i for i in range(1, len(times) + 1)]
    return np.array(x), np.array(y)


def pick_xticks(x0: float, x1: float) -> tuple[list[float], str]:
    """Evenly spaced x tick positions plus a date format for the given span.

    Ticks are anchored at the newest date and step backwards, so the latest
    day is always labeled. The granularity coarsens from days to months to
    years as the history grows, keeping the label count at or below
    MAX_XTICKS instead of drawing one tick per day forever.
    """
    start = mdates.num2date(x0)
    end = mdates.num2date(x1)
    span_days = x1 - x0

    for step in DAY_STEPS:
        if span_days / step <= MAX_XTICKS:
            anchor = end.replace(hour=0, minute=0, second=0, microsecond=0)
            ticks = []
            while (num := mdates.date2num(anchor)) >= x0:
                ticks.append(num)
                anchor -= timedelta(days=step)
            fmt = "%b %-d" if start.year == end.year else "%b %-d, %Y"
            return sorted(ticks), fmt

    span_months = (end.year - start.year) * 12 + end.month - start.month
    for step in MONTH_STEPS:
        if span_months / step <= MAX_XTICKS:
            # Month starts read better than an offset from "today" here.
            year, month = end.year, end.month
            ticks = []
            while (num := mdates.date2num(end.replace(
                year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0
            ))) >= x0:
                ticks.append(num)
                month -= step
                while month < 1:
                    month += 12
                    year -= 1
            fmt = "%b %Y" if start.year != end.year else "%b"
            return sorted(ticks), fmt

    # Year granularity is the coarsest fallback, so widen the step as far as
    # needed rather than giving up and returning a crowded axis.
    span_years = end.year - start.year
    step = next(
        (s for s in YEAR_STEPS if span_years / s <= MAX_XTICKS),
        max(1, -(-span_years // MAX_XTICKS)),
    )
    year = end.year
    ticks = []
    while (num := mdates.date2num(end.replace(
        year=year, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
    ))) >= x0:
        ticks.append(num)
        year -= step
    return sorted(ticks), "%Y"


def thin_xticklabels(fig, ax, min_gap: float = 14.0) -> None:
    """Drop every n-th label until neighbours no longer crowd each other.

    pick_xticks bounds the tick *count*, but whether the labels actually fit
    depends on the rendered text width and figure size, so measure the drawn
    labels and thin from the right (keeping the newest date) until every pair
    is at least `min_gap` pixels apart.
    """
    ticks = list(ax.get_xticks())
    for keep in range(1, max(len(ticks), 1) + 1):
        kept = ticks[::-1][::keep][::-1]
        ax.set_xticks(kept)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [
            lbl.get_window_extent(renderer=renderer)
            for lbl in ax.get_xticklabels()
            if lbl.get_text()
        ]
        if all(
            nxt.x0 - cur.x1 >= min_gap for cur, nxt in zip(boxes, boxes[1:])
        ):
            return


def draw(x: np.ndarray, y: np.ndarray, repo: str, out: Path) -> None:
    """Draw and save the star history chart."""
    bg, text, subtext, grid = THEME["bg"], THEME["text"], THEME["subtext"], THEME["grid"]

    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=200)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.80, bottom=0.10)

    ax.set_ylim(0, y.max() * 1.10)
    ax.set_xlim(x[0], x[-1] + (x[-1] - x[0]) * 0.03)

    # Gradient fill under the curve: accent fading from top to transparent.
    r, g, b, _ = to_rgba(ACCENT)
    fade = LinearSegmentedColormap.from_list("fade", [(r, g, b, 0.0), (r, g, b, 0.35)])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(
        grad,
        aspect="auto",
        cmap=fade,
        origin="lower",
        extent=[ax.get_xlim()[0], ax.get_xlim()[1], 0, ax.get_ylim()[1]],
        zorder=1,
    )
    xs = np.concatenate([[x[0]], x, [x[-1]]])
    ys = np.concatenate([[0.0], y, [0.0]])
    (clip,) = ax.fill(xs, ys, alpha=0, zorder=1)
    im.set_clip_path(clip)

    # Glow underlay + main line.
    ax.plot(x, y, color=ACCENT, linewidth=7, alpha=0.10, solid_capstyle="round", zorder=2)
    ax.plot(x, y, color=ACCENT, linewidth=2.6, solid_capstyle="round", zorder=3)

    # Latest value: end dot + bold annotation.
    ax.scatter([x[-1]], [y[-1]], s=70, color=ACCENT, edgecolor=bg, linewidth=2.2, zorder=4)
    ax.annotate(
        f"{int(y[-1]):,} stars",
        xy=(x[-1], y[-1]),
        xytext=(-6, 14),
        textcoords="offset points",
        ha="right",
        fontsize=16,
        fontweight="bold",
        color=text,
    )

    # Titles.
    fig.text(0.075, 0.93, "Star History", fontsize=22, fontweight="bold", color=text)
    fig.text(0.075, 0.862, repo, fontsize=12.5, color=subtext)

    # Grid, spines, ticks.
    ax.yaxis.grid(True, color=grid, linewidth=0.9, linestyle=(0, (5, 4)))
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(axis="both", length=0, labelsize=11.5, colors=subtext, pad=8)
    ticks, date_fmt = pick_xticks(*ax.get_xlim())
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v):,}"))
    thin_xticklabels(fig, ax)

    fig.savefig(out, facecolor=bg, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    """Parse arguments and generate the star history chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument(
        "--start-date",
        default=None,
        help="chart start date (YYYY-MM-DD); defaults to repository creation date",
    )
    parser.add_argument("--out-dir", default="Data")
    parser.add_argument("--refresh", action="store_true", help="ignore the timestamp cache")
    args = parser.parse_args()

    if args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start = fetch_repo_created_at(args.repo)
    starred = fetch_starred_at(args.repo, refresh=args.refresh)
    x, y = build_series(starred, start)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    draw(x, y, args.repo, out_dir / "star-history.png")


if __name__ == "__main__":
    main()