#!/usr/bin/env python3
"""Render a sitemap for the gallery and demo pages present in a web build."""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote


EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"
SITE_URL = "https://pooya.ai/webgpu/"
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
SLUG_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")


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

    urls = [SITE_URL]
    for source in sorted(examples_root.glob("*.rs")):
        slug = source.stem
        if not source.is_file() or not SLUG_PATTERN.fullmatch(slug):
            continue
        demo = web_root / slug
        page = demo / "index.html"
        if demo.is_symlink() or page.is_symlink() or not page.is_file():
            continue
        urls.append(f"{SITE_URL}{quote(slug, safe='')}/")

    ET.register_namespace("", SITEMAP_NAMESPACE)
    urlset = ET.Element(f"{{{SITEMAP_NAMESPACE}}}urlset")
    for url in urls:
        entry = ET.SubElement(urlset, f"{{{SITEMAP_NAMESPACE}}}url")
        ET.SubElement(entry, f"{{{SITEMAP_NAMESPACE}}}loc").text = url
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
