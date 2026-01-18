#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests

ENV_URL = "https://poly.cam/env.js"
ALGOLIA_APP_ID = "R4Z39A8FL1"
ALGOLIA_INDEX = "capture"
DEFAULT_FILTERS = "tags:sneaker OR tags:shoe"
DEFAULT_HITS_PER_PAGE = 50


def parse_env_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate env JSON in env.js")
    return json.loads(text[start : end + 1])


def fetch_algolia_search_key(session):
    resp = session.get(ENV_URL, timeout=30)
    resp.raise_for_status()
    env = parse_env_json(resp.text)
    key = env.get("ALGOLIA_SEARCH_KEY")
    if not key:
        raise ValueError("ALGOLIA_SEARCH_KEY not found in env.js")
    return key


def slugify(value, max_len=60):
    value = value.strip()
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = value.strip("._-")
    if not value:
        value = "capture"
    return value[:max_len]


def relative_path_for_url(url, capture_id):
    path = urlparse(url).path
    marker = f"/captures/{capture_id}/"
    idx = path.find(marker)
    if idx != -1:
        rel = path[idx + len(marker) :]
        return rel.lstrip("/")
    return Path(path).name


def is_placeholder_glb(url):
    return not url or "update-placeholder.glb" in url


def algolia_search(session, app_id, api_key, params):
    url = f"https://{app_id}-dsn.algolia.net/1/indexes/{ALGOLIA_INDEX}/query"
    headers = {
        "X-Algolia-API-Key": api_key,
        "X-Algolia-Application-Id": app_id,
    }
    resp = session.post(url, json={"params": params}, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_file(session, url, dest, overwrite):
    if dest.exists() and not overwrite:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 512):
                if chunk:
                    handle.write(chunk)
    return True


def collect_assets(hit, prefer_glb):
    assets = []
    if prefer_glb:
        glb_url = hit.get("glb")
        if glb_url and not is_placeholder_glb(glb_url):
            assets.append(glb_url)
            return assets
    gltf = hit.get("gltf") or {}
    for key in ("main", "geometry"):
        url = gltf.get(key)
        if url:
            assets.append(url)
    textures = gltf.get("textures") or {}
    assets.extend(textures.values())
    return assets


def load_expected_sizes(gltf_path):
    try:
        with open(gltf_path, "r", encoding="utf-8") as handle:
            gltf = json.load(handle)
    except Exception:
        return {}
    sizes = {}
    for buf in gltf.get("buffers", []):
        uri = buf.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        sizes[uri] = buf.get("byteLength")
    return sizes


def file_ok(path, expected_size=None):
    if not path.exists():
        return False
    size = path.stat().st_size
    if expected_size is not None:
        return size >= expected_size
    return size > 0


def log_status(message):
    print("\n" + message)


def redownload_with_retries(session, url, dest, expected_size, retries):
    for attempt in range(1, retries + 1):
        try:
            download_file(session, url, dest, overwrite=True)
        except Exception as exc:
            log_status(f"verify: retry {attempt} failed for {dest.name} ({exc})")
        if file_ok(dest, expected_size):
            log_status(f"verify: redownloaded {dest.name} (attempt {attempt}) ok")
            return True
    log_status(f"verify: redownload failed for {dest.name} after {retries} attempts")
    return False


def format_progress(done, total, width=30):
    if not total:
        return f"{done}"
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total}"


def print_progress(done, total, ok, failed, skipped):
    bar = format_progress(done, total)
    msg = f"{bar} files ok={ok} failed={failed} skipped={skipped}"
    print("\r" + msg, end="", flush=True)


def build_params(filters, hits_per_page, page):
    params = {
        "filters": filters,
        "hitsPerPage": hits_per_page,
        "page": page,
        "ignorePlurals": "true",
        "advancedSyntax": "true",
    }
    return urlencode(params, safe=": ")


def main():
    parser = argparse.ArgumentParser(
        description="Download Polycam sneaker models as glTF assets."
    )
    parser.add_argument(
        "--output-dir",
        default="polycam-downloads",
        help="Output directory (relative to this script).",
    )
    parser.add_argument(
        "--hits-per-page",
        type=int,
        default=DEFAULT_HITS_PER_PAGE,
        help="Algolia hits per page.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages to fetch (0 means all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of models to download (0 means no limit).",
    )
    parser.add_argument(
        "--include-unsavable",
        action="store_true",
        help="Include captures marked as unsavable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--prefer-glb",
        action="store_true",
        help="Prefer real GLB assets when available.",
    )
    parser.add_argument(
        "--filters",
        default=DEFAULT_FILTERS,
        help="Algolia filter string.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded files and retry any mismatches.",
    )
    parser.add_argument(
        "--verify-retries",
        type=int,
        default=2,
        help="Retry count when verification fails.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Mozilla/5.0 (compatible; PolycamSneakerFetcher/1.0)"}
    )

    try:
        search_key = fetch_algolia_search_key(session)
    except Exception as exc:
        print(f"Failed to fetch Algolia search key: {exc}", file=sys.stderr)
        return 1

    seen_ids = set()
    downloaded = []
    processed = 0
    total_hits = None
    files_ok = 0
    files_failed = 0
    files_skipped = 0
    files_retry_ok = 0
    files_retry_failed = 0
    page = 0
    total_pages = None

    while True:
        params = build_params(args.filters, args.hits_per_page, page)
        try:
            payload = algolia_search(session, ALGOLIA_APP_ID, search_key, params)
        except Exception as exc:
            print(f"Search failed on page {page}: {exc}", file=sys.stderr)
            return 1

        hits = payload.get("hits") or []
        if total_pages is None:
            total_pages = payload.get("nbPages")
        if total_hits is None:
            total_hits = payload.get("nbHits")
        if not hits:
            break

        for hit in hits:
            capture_id = hit.get("id")
            if not capture_id or capture_id in seen_ids:
                continue
            seen_ids.add(capture_id)
            processed = len(seen_ids)
            print_progress(processed, total_hits, files_ok, files_failed, files_skipped)

            if not args.include_unsavable and not hit.get("savable", True):
                continue

            name = hit.get("name") or ""
            slug = slugify(name)
            folder = output_dir / f"{slug}_{capture_id}"
            folder.mkdir(parents=True, exist_ok=True)

            assets = collect_assets(hit, args.prefer_glb)
            if not assets:
                continue

            expected_sizes = {}
            url_by_relpath = {}
            files_downloaded = 0
            for url in assets:
                rel_path = relative_path_for_url(url, capture_id)
                dest = folder / rel_path
                url_by_relpath[rel_path] = url
                expected_size = expected_sizes.get(rel_path) if args.verify else None
                try:
                    downloaded_now = download_file(session, url, dest, args.overwrite)
                    if downloaded_now:
                        files_downloaded += 1
                        files_ok += 1
                    else:
                        if args.verify and not file_ok(dest, expected_size):
                            if redownload_with_retries(
                                session,
                                url,
                                dest,
                                expected_size,
                                args.verify_retries,
                            ):
                                files_retry_ok += 1
                            else:
                                files_retry_failed += 1
                        else:
                            files_skipped += 1
                except Exception as exc:
                    files_failed += 1
                    print(f"Download failed: {url} ({exc})", file=sys.stderr)
                    print_progress(
                        processed, total_hits, files_ok, files_failed, files_skipped
                    )
                    continue

                if args.verify and rel_path.lower().endswith(".gltf"):
                    expected_sizes = load_expected_sizes(dest)

                print_progress(processed, total_hits, files_ok, files_failed, files_skipped)

            if args.verify and expected_sizes:
                for rel_path, expected_size in expected_sizes.items():
                    dest = folder / rel_path
                    if file_ok(dest, expected_size):
                        continue
                    url = url_by_relpath.get(rel_path)
                    if not url:
                        log_status(
                            f"verify: missing url for {rel_path} in {capture_id}"
                        )
                        files_retry_failed += 1
                        continue
                    if redownload_with_retries(
                        session,
                        url,
                        dest,
                        expected_size,
                        args.verify_retries,
                    ):
                        files_retry_ok += 1
                    else:
                        files_retry_failed += 1

            meta_path = folder / "meta.json"
            if not meta_path.exists() or args.overwrite:
                with open(meta_path, "w", encoding="utf-8") as handle:
                    json.dump(hit, handle, ensure_ascii=True, indent=2)

            downloaded.append(
                {
                    "id": capture_id,
                    "name": name,
                    "folder": str(folder.relative_to(script_dir)),
                    "assets": assets,
                    "files": files_downloaded,
                }
            )

            if args.limit and len(downloaded) >= args.limit:
                break

            time.sleep(0.2)

        if args.limit and len(downloaded) >= args.limit:
            break

        page += 1
        if args.max_pages and page >= args.max_pages:
            break
        if total_pages is not None and page >= total_pages:
            break

    index_path = output_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(downloaded, handle, ensure_ascii=True, indent=2)

    print()
    print(
        f"Downloaded {len(downloaded)} captures into {output_dir} "
        f"(files ok={files_ok}, failed={files_failed}, skipped={files_skipped}, "
        f"retry_ok={files_retry_ok}, retry_failed={files_retry_failed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
