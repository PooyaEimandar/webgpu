#!/usr/bin/env python3
"""Render a sitemap for the gallery and demo pages present in a web build."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"
ARTICLES_ROOT = REPOSITORY_ROOT / "web" / "articles"
SITE_URL = "https://pooya.ai/webgpu/"
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
SLUG_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")


def last_changed(*paths: Path) -> str | None:
    """The date of the last commit that changed any of `paths`.

    None without the complete history, as in a shallow clone, or outside git:
    a sitemap may leave lastmod out, but must not guess it.
    """
    root = REPOSITORY_ROOT

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        if git("rev-parse", "--is-shallow-repository") != "false":
            return None
        return git("log", "-1", "--format=%cs", "--", *(str(path) for path in paths)) or None
    except (OSError, subprocess.CalledProcessError):
        return None


def render_sitemap(web_root: Path, examples_root: Path = EXAMPLES_ROOT) -> str:
    """Return XML for built pages, excluding unknown directories and unbuilt demos.

    The source examples are the allowlist; an output folder alone does not make
    an asset or temporary preview a demo. Only direct, regular index.html files
    are included, so partial builds never advertise pages they did not produce.
    """
    if not web_root.is_dir():
        raise ValueError(f"Build output directory not found: {web_root}")
    gallery = web_root / "index.html"
    if not gallery.is_file() or gallery.is_symlink():
        raise ValueError(f"Gallery page not found or not a regular file: {gallery}")
    if not examples_root.is_dir():
        raise ValueError(f"Examples directory not found: {examples_root}")

    urls = [(SITE_URL, last_changed(REPOSITORY_ROOT / "web" / "index.html"))]
    for source in sorted(examples_root.glob("*.rs")):
        slug = source.stem
        if not source.is_file() or not SLUG_PATTERN.fullmatch(slug):
            continue
        demo = web_root / slug
        page = demo / "index.html"
        if demo.is_symlink() or page.is_symlink() or not page.is_file():
            continue
        # A page with an article changes when the article does.
        article = ARTICLES_ROOT / f"{slug}.html"
        sources = [article, article.with_suffix(".json")] if article.is_file() else [source]
        urls.append((f"{SITE_URL}{quote(slug, safe='')}/", last_changed(*sources)))

    ET.register_namespace("", SITEMAP_NAMESPACE)
    urlset = ET.Element(f"{{{SITEMAP_NAMESPACE}}}urlset")
    for url, lastmod in urls:
        entry = ET.SubElement(urlset, f"{{{SITEMAP_NAMESPACE}}}url")
        ET.SubElement(entry, f"{{{SITEMAP_NAMESPACE}}}loc").text = url
        if lastmod:
            ET.SubElement(entry, f"{{{SITEMAP_NAMESPACE}}}lastmod").text = lastmod
    ET.indent(urlset, space="  ")
    return ET.tostring(urlset, encoding="utf-8", xml_declaration=True).decode("utf-8") + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("web_root", type=Path, help="Build output directory, such as target/web")
    arguments = parser.parse_args()
    try:
        sitemap = render_sitemap(arguments.web_root)
    except (OSError, ValueError) as error:
        print(f"render-sitemap: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(sitemap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
