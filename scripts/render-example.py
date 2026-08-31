#!/usr/bin/env python3
"""Render a demo page and its optional article using only the Python standard library."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote


WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
SITE_URL = "https://pooya.ai"
AUTHOR = "Pooya Eimandar"
TOKEN_PATTERN = re.compile(r"__[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*__")
ARTICLE_NAV = """<a class="more-info" href="#about" aria-label="More info about this demo">
  <span>More info</span>
  <svg width="34" height="14" viewBox="0 0 34 14" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
    <path fill-rule="evenodd" clip-rule="evenodd" d="M33.5609 1.54346C34.0381 2.5875 33.6881 3.87821 32.7791 4.42633L17.0387 13.9181L1.48663 4.42115C0.580153 3.86761 0.235986 2.57483 0.717909 1.53365C1.19983 0.492464 2.32535 0.097152 3.23182 0.650692L17.0497 9.08858L31.051 0.64551C31.96 0.0973872 33.0837 0.499411 33.5609 1.54346Z" fill="currentColor"/>
  </svg>
</a>"""


def replace_tokens(source: str, values: dict[str, str], label: str) -> str:
    """Expand authored tokens once, without interpreting inserted content as a template."""
    def replace(match: re.Match[str]) -> str:
        token = match.group()
        if token not in values:
            raise ValueError(f"Unresolved template token {token} in {label}")
        return values[token]

    return TOKEN_PATTERN.sub(replace, source)


def json_ld_script(data: dict[str, Any]) -> str:
    """Serialize structured data without allowing it to close its script element."""
    json_ld = (
        json.dumps(data, ensure_ascii=False, indent=2)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return f'<script type="application/ld+json">\n{json_ld}\n</script>'


def article_head(
    example: str, metadata: dict[str, Any], *, has_breadcrumbs: bool = False
) -> str:
    canonical = f"{SITE_URL}/webgpu/{example}/"
    image_url = f"{SITE_URL}/webgpu/screenshots/{quote(metadata['image'], safe='')}"
    title = metadata["title"]
    description = metadata["description"]
    tags = [
        f"<title>{html.escape(title)} | {AUTHOR}</title>",
        f'<link rel="canonical" href="{canonical}" />',
    ]
    meta = [
        ("name", "description", description),
        ("name", "author", AUTHOR),
        ("name", "robots", "index, follow, max-image-preview:large"),
        ("name", "theme-color", "#000000"),
        ("property", "og:type", "article"),
        ("property", "og:title", title),
        ("property", "og:description", description),
        ("property", "og:url", canonical),
        ("property", "og:site_name", AUTHOR),
        ("property", "og:image", image_url),
        ("property", "og:image:alt", metadata["imageAlt"]),
    ]
    if "imageWidth" in metadata:
        meta.extend([
            ("property", "og:image:width", str(metadata["imageWidth"])),
            ("property", "og:image:height", str(metadata["imageHeight"])),
        ])
    meta.extend([
        ("property", "article:author", SITE_URL),
        ("name", "twitter:card", "summary_large_image"),
        ("name", "twitter:title", title),
        ("name", "twitter:description", description),
        ("name", "twitter:image", image_url),
        ("name", "twitter:image:alt", metadata["imageAlt"]),
    ])
    tags.extend(
        f'<meta {attribute}="{key}" content="{html.escape(value, quote=True)}" />'
        for attribute, key, value in meta
    )
    structured_data = {
        "@context": "https://schema.org",
        "@type": "BlogPosting",
        "@id": f"{canonical}#about",
        "headline": title,
        "description": description,
        "author": {"@type": "Person", "name": AUTHOR, "url": SITE_URL},
        "image": image_url,
        "mainEntityOfPage": {"@type": "WebPage", "@id": canonical},
        "url": canonical,
        "inLanguage": "en",
    }
    if has_breadcrumbs:
        structured_data["mainEntityOfPage"]["breadcrumb"] = {
            "@id": f"{canonical}#breadcrumbs"
        }
    tags.append(json_ld_script(structured_data))
    if has_breadcrumbs:
        tags.append(json_ld_script({
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "@id": f"{canonical}#breadcrumbs",
            "itemListElement": [
                {
                    "@type": "ListItem", "position": 1,
                    "name": "WebGPU examples", "item": f"{SITE_URL}/webgpu/",
                },
                {
                    "@type": "ListItem", "position": 2,
                    "name": metadata.get("breadcrumbName", title), "item": canonical,
                },
            ],
        }))
    return "\n    ".join(tags)


def render_example(example: str, build_id: str, web_root: Path = WEB_ROOT) -> str:
    """Return the complete HTML page, raising ValueError for invalid article inputs."""
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", example):
        raise ValueError(
            "Invalid example slug: use lowercase letters, numbers, hyphens, or "
            "underscores, starting with a letter"
        )
    if not build_id:
        raise ValueError("Build ID must not be empty")

    template_path = web_root / "example.html"
    metadata_path = web_root / "articles" / f"{example}.json"
    article_path = metadata_path.with_suffix(".html")
    has_metadata = metadata_path.exists()
    has_article = article_path.exists()
    if has_metadata != has_article:
        missing = article_path if has_metadata else metadata_path
        raise ValueError(f"Incomplete article for {example}: missing {missing}")

    values = {
        "__EXAMPLE__": example,
        "__EXAMPLE_LABEL__": html.escape(example, quote=True),
        "__BUILD_ID__": quote(build_id, safe=""),
        "__PAGE_HEAD__": f"<title>webgpu {example}</title>",
        "__PAGE_CLASS__": "",
        "__ARTICLE__": "",
        "__ARTICLE_NAV__": "",
    }
    if has_article:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid article metadata in {metadata_path}: {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"Article metadata must be a JSON object: {metadata_path}")
        for field in ("title", "description", "image", "imageAlt"):
            if not isinstance(metadata.get(field), str) or not metadata[field].strip():
                raise ValueError(f"Article metadata {metadata_path}: {field} must be a nonempty string")
        if "breadcrumbName" in metadata and (
            not isinstance(metadata["breadcrumbName"], str) or not metadata["breadcrumbName"].strip()
        ):
            raise ValueError(f"Article metadata {metadata_path}: breadcrumbName must be a nonempty string")
        if ("imageWidth" in metadata) != ("imageHeight" in metadata):
            raise ValueError(f"Article metadata {metadata_path}: imageWidth and imageHeight must be supplied together")
        for field in ("imageWidth", "imageHeight"):
            if field in metadata and (type(metadata[field]) is not int or metadata[field] <= 0):
                raise ValueError(f"Article metadata {metadata_path}: {field} must be a positive integer")
        image_name = metadata["image"]
        if image_name in (".", "..") or "/" in image_name or "\\" in image_name:
            raise ValueError(f"Article metadata {metadata_path}: image must be a screenshot basename")

        article_source = article_path.read_text(encoding="utf-8")
        has_breadcrumbs = "__ARTICLE_BREADCRUMBS__" in article_source
        breadcrumb_name = html.escape(metadata.get("breadcrumbName", metadata["title"]), quote=True)
        breadcrumbs = (
            '<nav id="breadcrumbs" class="breadcrumbs" aria-label="Breadcrumb">'
            '<ol role="list"><li><a href="../">WebGPU examples</a></li>'
            f'<li aria-current="page">{breadcrumb_name}</li></ol></nav>'
        )
        article = replace_tokens(
            article_source,
            {
                "__EXAMPLE__": example,
                "__ARTICLE_TITLE__": html.escape(metadata["title"], quote=True),
                "__ARTICLE_DESCRIPTION__": html.escape(metadata["description"], quote=True),
                "__ARTICLE_BREADCRUMBS__": breadcrumbs,
                "__ARTICLE_IMAGE__": f"../screenshots/{quote(image_name, safe='')}",
                "__ARTICLE_IMAGE_ALT__": html.escape(metadata["imageAlt"], quote=True),
                "__CURRENT_YEAR__": str(date.today().year),
            },
            str(article_path),
        )
        values.update({
            "__EXAMPLE_LABEL__": html.escape(
                re.sub(
                    r"^WebGPU\s+",
                    "",
                    metadata.get("breadcrumbName", example),
                    flags=re.IGNORECASE,
                ),
                quote=True,
            ),
            "__PAGE_HEAD__": article_head(example, metadata, has_breadcrumbs=has_breadcrumbs),
            "__PAGE_CLASS__": "has-article",
            "__ARTICLE__": article,
            "__ARTICLE_NAV__": ARTICLE_NAV,
        })

    return replace_tokens(template_path.read_text(encoding="utf-8"), values, str(template_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", help="Example name, such as triangle")
    parser.add_argument("build_id", help="Cache version for the JavaScript and WebAssembly assets")
    arguments = parser.parse_args()
    try:
        rendered = render_example(arguments.example, arguments.build_id)
    except (OSError, ValueError) as error:
        print(f"render-example: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
