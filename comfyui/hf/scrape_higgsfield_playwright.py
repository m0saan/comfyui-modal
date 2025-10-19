#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys
import time
from typing import Iterable, Optional, Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

ROOT = "https://higgsfield.ai"

# ----------------------------- helpers -----------------------------

def _norm_key(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", s.upper())

def _folderize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.upper()).strip("_") or "UNKNOWN"

def _name_map(names: Iterable[str]) -> Dict[str, str]:
    return {_norm_key(n): n for n in names}

def log(v: bool, *msg) -> None:
    if v:
        print(*msg)

# ----------------------------- allowlists -----------------------------

CAMERAS_NAMES = [
    "General","Bullet Time","Aerial Pullback","Arc Left","BTS","Buckle Up",
    "Car Chasing","Car Grip","Crane Down","Crane Over The Head","Crane Up",
    "Crash Zoom In","Crash Zoom Out","Dolly In","Dolly Left","Dolly Out",
    "Dolly Right","Dolly Zoom In","Dolly Zoom Out","Double Dolly","Dutch Angle",
    "Eating Zoom","Fisheye","Flying Cam Transition","Focus Change","FPV Drone",
    "Glam","Handheld","Head Tracking","Hero Cam","Hyperlapse","Incline",
    "Jib down","Jib up","Lazy Susan","Low Shutter","Mouth In","Object POV",
    "Overhead","Rapid Zoom In","Rapid Zoom Out","Road Rush","Robo Arm",
    "Snorricam","Static","Super Dolly In","Super Dolly Out","Through Object In",
    "Through Object Out","Tilt Down","Tilt up","Timelapse Glam","Timelapse Human",
    "Timelapse Landscape","Whip Pan","Wiggle","YoYo Zoom","360 Orbit","Zoom In",
    "Zoom Out","Arc Right","3D Rotation",
]
CAMERAS_MAP = _name_map(CAMERAS_NAMES)

ACTION_NAMES = [
    "Action Run","Baseball Kick","Basketball Dunks","Boxing","Catwalk",
    "Downhill POV","Drunk Master","Fast Sprint","Flip","Handheld Run",
    "Helicopter Escape","Moonwalk Left","Moonwalk Right","Rap Flex","Skate Cruise",
    "Skateboard Glide","Skateboard Ollie","Ski Carving","Ski Powder",
    "Snowboard Carving","Snowboard Powder",
]
ACTION_MAP = _name_map(ACTION_NAMES)

SEEDANCE_NAMES = [
    "General","Beach Ride","Buddy","Car Drive","Crying","Fix and pose",
    "Flashback","Hair Style","Happy","Hero Flight","Look, BOOM!",
    "Morning routine","Motor Ride","Oni Mask","Outfit Check","Outfit Switch",
    "Peak Moment","Plate Check","Red Carpet","Sand Cut","Selfie","Shocked",
    "Spirit Animal","Turning Monkey","Yacht",
]
SEEDANCE_MAP = _name_map(SEEDANCE_NAMES)

MOTIONS_NAMES = [
    "Levitation","Low Shutter","Mouth In","Object POV","Overhead",
    "Rap Flex","Robo Arm","Snorricam Low Shutter","Static",
    "Super Dolly In","Super Dolly Out","Tentacles","Through Object In",
    "Through Object Out","Tilt Down","Tilt Up","Timelapse Human",
    "Timelapse Landscape","Whip Pan","Wiggle","360 Orbit",
]
MOTIONS_MAP = _name_map(MOTIONS_NAMES)

EFFECTS_MAP: Dict[str, str] = {}
UGC_MAP: Dict[str, str] = {}

# ----------------------------- network -----------------------------

def head_content_length(session: requests.Session, url: str) -> int:
    try:
        r = session.head(url, timeout=30, allow_redirects=True)
        if r.status_code < 400:
            return int(r.headers.get("content-length", 0))
    except Exception:
        pass
    return 0

def download(session: requests.Session, url: str, dst: pathlib.Path, verbose: bool=False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    expected = head_content_length(session, url)
    if dst.exists() and (expected == 0 or dst.stat().st_size == expected):
        log(verbose, f"   ↪︎ skip complete: {dst.name}")
        return

    with session.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", expected)) or None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name, leave=False) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                f.write(chunk)
                if total:
                    bar.update(len(chunk))
    tmp.replace(dst)
    log(verbose, f"   ✓ saved: {dst}")

# ----------------------------- parsing helpers -----------------------------

def extract_prompt_from_lightbox_html(html: str) -> Optional[str]:
    """Pick the right-panel text that follows 'Prompt'."""
    soup = BeautifulSoup(html, "lxml")
    # find any element containing 'Prompt' then read its container text
    candidate = None
    for el in soup.find_all(True):
        if el.get_text(strip=True).lower() == "prompt":
            candidate = el.parent if el.parent else el
            break
    if not candidate:
        # fallback: any block with the word 'Prompt'
        candidate = soup.find(lambda tag: tag.name in ("div","section","aside") and "prompt" in tag.get_text(" ", strip=True).lower())
    if not candidate:
        return None
    txt = candidate.get_text("\n", strip=True)
    # remove the heading word
    txt = re.sub(r"^\s*Prompt\s*[:\n]*", "", txt, flags=re.I)
    return txt.strip() or None

def infer_label_from_html(html: str, label_map: Dict[str, str]) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    norm_text = _norm_key(text)
    for key, pretty in label_map.items():
        if key and key in norm_text:
            return pretty
    # fallback: chip-like ALL CAPS
    stop = {"VISUAL EFFECTS","UGC","TOP CHOICE","NEW"}
    for m in re.finditer(r"\b([A-Z][A-Z0-9 ]{2,28})\b", text):
        cand = m.group(1).strip()
        if len(cand) >= 3 and cand.upper() not in stop and any(ch.isalpha() for ch in cand):
            return cand
    return None

# ----------------------------- playwright helpers -----------------------------

def safe_goto(page, url: str, wait: str, timeout_ms: int, verbose: bool=False) -> bool:
    for w in [wait, "load", "domcontentloaded", "commit"]:
        try:
            page.goto(url, wait_until=w, timeout=timeout_ms)
            return True
        except PWTimeout:
            log(verbose, f"[timeout] goto {url} waiting for {w}")
        except Exception as e:
            log(verbose, f"[error] goto {url}: {e}")
            time.sleep(0.25)
    return False

def discover_motion_links(play, mode: str, wait: str, timeout_ms: int, verbose: bool=False) -> List[str]:
    browser = play.chromium.launch(headless=True)
    ctx = browser.new_context(ignore_https_errors=True)
    page = ctx.new_page()
    ctx.set_default_navigation_timeout(timeout_ms)

    start_urls: List[str] = []
    if mode == "cameras":
        start_urls = [f"{ROOT}/cameras", f"{ROOT}/camera-controls", f"{ROOT}/"]
    elif mode == "action":
        start_urls = [f"{ROOT}/action-movement"]
    elif mode == "seedance":
        start_urls = [f"{ROOT}/seedance"]
    elif mode == "motions":
        start_urls = [f"{ROOT}/motions"]
    elif mode == "effects":
        start_urls = [f"{ROOT}/visual-effects"]
    elif mode == "ugc":
        start_urls = [f"{ROOT}/ugc"]
    else:
        browser.close()
        raise SystemExit("Unknown mode")

    links: List[str] = []
    for u in start_urls:
        if not safe_goto(page, u, wait, timeout_ms, verbose):
            continue
        time.sleep(0.5)

        # collect /motion links found anywhere
        anchors = page.locator("a[href*='/motion/']")
        for i in range(anchors.count()):
            href = anchors.nth(i).get_attribute("href") or ""
            if "/motion/" in href:
                links.append(urljoin(ROOT, href))

        log(verbose, f"[info] collected {anchors.count()} motion links from {u}")

    browser.close()
    # de-dup
    out, seen = [], set()
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    log(verbose, f"[info] discovered {len(out)} unique motion pages")
    return out

# ----------------------------- main scraping -----------------------------

def scrape_mode(
    play,
    mode: str,
    out_root: pathlib.Path,
    allow_map: Dict[str, str],
    selected_keys: set[str],     # empty => allow all
    save_prompts: bool,
    headful: bool,
    verbose: bool,
    dry_run: bool,
    wait: str,
    timeout_ms: int,
) -> Tuple[int, int]:

    motion_links = discover_motion_links(play, mode, wait, timeout_ms, verbose)

    browser = play.chromium.launch(headless=not headful)
    ctx = browser.new_context(ignore_https_errors=True)
    page = ctx.new_page()
    ctx.set_default_navigation_timeout(timeout_ms)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    })

    total_dl, total_skip = 0, 0
    seen_cloudfront_names: set[str] = set()

    for murl in motion_links:
        if not safe_goto(page, murl, wait, timeout_ms, verbose):
            log(verbose, f"   (skip) cannot open {murl}")
            continue
        time.sleep(0.25)
        base_html = page.content()

        # label + folder
        if mode == "seedance":
            label = infer_label_from_html(base_html, SEEDANCE_MAP) or "UNKNOWN"
            if selected_keys and _norm_key(label) not in selected_keys:
                continue
            out_dir = out_root / "SEEDANCE" / _folderize(label)
        elif mode == "action":
            label = infer_label_from_html(base_html, ACTION_MAP) or "UNKNOWN"
            if selected_keys and _norm_key(label) not in selected_keys:
                continue
            out_dir = out_root / _folderize(label)
        elif mode == "cameras":
            label = infer_label_from_html(base_html, CAMERAS_MAP) or "UNKNOWN"
            if selected_keys and _norm_key(label) not in selected_keys:
                continue
            out_dir = out_root / _folderize(label)
        elif mode == "motions":
            label = infer_label_from_html(base_html, MOTIONS_MAP) or "UNKNOWN"
            if selected_keys and _norm_key(label) not in selected_keys:
                continue
            out_dir = out_root / _folderize(label)
        elif mode == "effects":
            label = infer_label_from_html(base_html, EFFECTS_MAP) or "UNKNOWN"
            out_dir = out_root / "VISUAL_EFFECTS" / _folderize(label)
        elif mode == "ugc":
            label = infer_label_from_html(base_html, UGC_MAP) or "UNKNOWN"
            out_dir = out_root / "UGC" / _folderize(label)
        else:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        log(verbose, f"[{label}] {murl}")

        # --- CLICK EVERY TILE to open the lightbox and capture HD url ---
        # try a few likely tile selectors
        tile_sel = (
            "section img, section video, div img, div video"
        )
        tiles = page.locator(tile_sel)
        n = tiles.count()
        if verbose:
            print(f"   tiles detected: {n}")

        # keep a per-page counter for filenames (order)
        clip_index = 1

        for i in range(n):
            t = tiles.nth(i)
            try:
                t.scroll_into_view_if_needed(timeout=3000)
                time.sleep(0.05)
                # click the nearest clickable parent to open the lightbox
                handle = t.element_handle()
                if handle:
                    handle.click(force=True, timeout=5000)
                else:
                    t.click(force=True, timeout=5000)
            except Exception:
                continue

            # wait for a cloudfront video to appear in the lightbox
            v = None
            try:
                # wait until at least one <video> exists
                page.wait_for_selector("video", timeout=8000)
                v = page.locator("video").first
            except Exception:
                # close and continue
                page.keyboard.press("Escape")
                time.sleep(0.1)
                continue

            # read url + resolution from within the page
            video_url = None
            vw = 0
            try:
                video_url = page.evaluate(
                    """() => {
                        const vid = document.querySelector('video');
                        if (!vid) return null;
                        const src = vid.currentSrc || vid.src || (vid.querySelector('source')?.src) || null;
                        return src;
                    }"""
                )
                vw = page.evaluate(
                    """() => { const vid = document.querySelector('video'); return vid ? (vid.videoWidth||0) : 0; }"""
                )
            except Exception:
                pass

            # extract prompt from the lightbox (right panel)
            prompt_txt = None
            if save_prompts:
                try:
                    light_html = page.content()
                    prompt_txt = extract_prompt_from_lightbox_html(light_html)
                except Exception:
                    prompt_txt = None

            # close lightbox
            try:
                page.keyboard.press("Escape")
                time.sleep(0.1)
            except Exception:
                pass

            # only accept real HD cloudfront files
            if not video_url:
                continue
            host = urlparse(video_url).netloc.lower()
            if "cloudfront" not in host:
                # thumbnail or inline asset – skip
                continue
            if vw and vw < 400:
                # low-res preview – skip
                continue

            cf_name = pathlib.Path(urlparse(video_url).path).name  # e.g., 14894c78-...mp4
            if cf_name in seen_cloudfront_names:
                if verbose:
                    print(f"   ↪︎ dup cloudfront: {cf_name}")
                continue
            seen_cloudfront_names.add(cf_name)

            # build destination path
            # keep cloudfront name to avoid false duplicates
            dst = out_dir / cf_name
            if dst.exists():
                # fallback unique
                dst = out_dir / f"{cf_name.rsplit('.',1)[0]}_{clip_index:02d}.mp4"

            if dry_run:
                print(f"   • would download {video_url} -> {dst}")
            else:
                try:
                    download(session, video_url, dst, verbose)
                    total_dl += 1
                except Exception as exc:
                    print(f"   ❌  download failed: {exc}")
                    continue

                # write prompt (if found)
                if save_prompts and prompt_txt:
                    (dst.with_suffix(".prompt.txt")).write_text(prompt_txt, encoding="utf-8")

            clip_index += 1

    browser.close()
    return total_dl, total_skip

# ----------------------------- CLI ------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Higgsfield scraper (opens lightbox, grabs CloudFront HD)")
    p.add_argument("--mode", choices=["cameras", "action", "seedance", "motions", "effects", "ugc"], required=True)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("datasets"))
    p.add_argument("--all", action="store_true", help="Download all categories (ignore allowlist)")
    p.add_argument("--include", type=str, default="", help="Comma-separated labels to include when not using --all")
    p.add_argument("--save-prompts", action="store_true", default=True)
    p.add_argument("--headful", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--timeout-ms", type=int, default=60000)
    p.add_argument("--wait", choices=["networkidle", "load", "domcontentloaded", "commit"], default="load")
    return p.parse_args(argv)

def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "cameras":
        allow_map = CAMERAS_MAP
    elif args.mode == "action":
        allow_map = ACTION_MAP
    elif args.mode == "seedance":
        allow_map = SEEDANCE_MAP
    elif args.mode == "effects":
        allow_map = EFFECTS_MAP
    elif args.mode == "ugc":
        allow_map = UGC_MAP
    else:
        allow_map = MOTIONS_MAP

    print(f"ARGS: {args}")
    print(f"ALLOW MAP: {allow_map}")
    print(f"SAVE PROMPTS: {args.save_prompts}")
    print(f"HEADFUL: {args.headful}")
    print(f"VERBOSE: {args.verbose}")
    print(f"DRY RUN: {args.dry_run}")
    print(f"TIMEOUT MS: {args.timeout_ms}")
    print(f"WAIT: {args.wait}")

    if args.all or args.mode in {"effects", "ugc"}:
        selected: set[str] = set()
    else:
        selected = set(allow_map.keys())
        if args.include:
            for item in args.include.split(","):
                k = _norm_key(item.strip())
                if k:
                    selected.add(k)

    with sync_playwright() as play:
        dl, sk = scrape_mode(
            play,
            args.mode,
            args.out,
            allow_map,
            selected,
            save_prompts=args.save_prompts,
            headful=args.headful,
            verbose=args.verbose,
            dry_run=args.dry_run,
            wait=args.wait,
            timeout_ms=args.timeout_ms,
        )

    print(f"\nDone. downloaded={dl}, skipped={sk}")
    if args.dry_run:
        print("(dry run: no files written)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
