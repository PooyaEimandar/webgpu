#!/usr/bin/env python3
"""Tests for optional article rendering; no WebGPU build or dependencies required."""

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin


sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location(
    "render_example", Path(__file__).with_name("render-example.py")
)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)

TEMPLATE = """<!doctype html>
<html lang="en" class="__PAGE_CLASS__">
<head>__PAGE_HEAD__</head>
<body><main><section id="demo">__ARTICLE_NAV__</section>__ARTICLE__</main>
<script type="module">
import init from "./__EXAMPLE__.js?build=__BUILD_ID__";
init("./__EXAMPLE___bg.wasm?build=__BUILD_ID__");
</script></body></html>"""
ARTICLE = """<article id="about" tabindex="-1">
__ARTICLE_BREADCRUMBS__
<h1>__ARTICLE_TITLE__</h1><p>__ARTICLE_DESCRIPTION__</p>
<p>This is the __EXAMPLE__ example.</p>
<img src="__ARTICLE_IMAGE__" alt="__ARTICLE_IMAGE_ALT__" width="1280" height="732">
<footer>&copy; <span data-current-year>__CURRENT_YEAR__</span> Pooya Eimandar. All rights reserved.</footer>
</article>"""
METADATA = {
    "title": "Drawing a Triangle with WebGPU",
    "description": "A look at Rust, vertex buffers, and WGSL shaders.",
    "image": "triangle.jpg",
    "imageAlt": "A triangle with a smooth RGB gradient on black",
}


class HeadParser(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.metadata = {}
        self.canonical = None
        self.scripts = []
        self.ids = []
        self.elements_by_id = {}
        self.images = []
        self.links = []
        self.h2_ids = []
        self.fragment_links = []
        self.h1_count = 0
        self.feed(document)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.append(attrs["id"])
            self.elements_by_id[attrs["id"]] = attrs
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img":
            self.images.append(attrs)
        if tag == "h2":
            self.h2_ids.append(attrs.get("id"))
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.fragment_links.append(attrs["href"][1:])
        if tag == "h1":
            self.h1_count += 1
        if tag == "meta":
            self.metadata[attrs.get("name", attrs.get("property"))] = attrs.get("content")
        elif tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href")
        elif tag == "script":
            self.scripts.append(attrs)


class RenderExampleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="webgpu-article-test-")
        self.addCleanup(self.temporary.cleanup)
        self.web_root = Path(self.temporary.name)
        (self.web_root / "example.html").write_text(TEMPLATE, encoding="utf-8")
        (self.web_root / "articles").mkdir()

    def add_article(self, metadata=None, fragment=ARTICLE):
        article_path = self.web_root / "articles" / "triangle"
        article_path.with_suffix(".json").write_text(
            json.dumps(METADATA if metadata is None else metadata), encoding="utf-8"
        )
        article_path.with_suffix(".html").write_text(fragment, encoding="utf-8")

    def render(self, example="triangle", build_id="123456"):
        return renderer.render_example(example, build_id, self.web_root)

    def json_ld(self, document, schema_type="BlogPosting"):
        nodes = [
            json.loads(source) for source in re.findall(
                r'<script type="application/ld\+json">(.*?)</script>', document, re.S
            )
        ]
        matches = [node for node in nodes if node.get("@type") == schema_type]
        self.assertEqual(len(matches), 1, f"Expected one {schema_type} node")
        return matches[0]

    def test_article_is_present_in_initial_html_with_search_and_social_metadata(self):
        self.add_article()
        document = self.render()
        head = HeadParser(document)
        canonical = "https://pooya.ai/webgpu/triangle/"
        image_url = "https://pooya.ai/webgpu/screenshots/triangle.jpg"
        self.assertIn('<html lang="en" class="has-article">', document)
        self.assertIn('<article id="about" tabindex="-1">', document)
        self.assertIn(f'<h1>{METADATA["title"]}</h1>', document)
        self.assertIn('<a class="more-info" href="#about"', document)
        self.assertIn('aria-hidden="true" focusable="false"', document)
        self.assertIn(f'<title>{METADATA["title"]} | Pooya Eimandar</title>', document)
        self.assertEqual(head.canonical, canonical)
        self.assertEqual(head.metadata["description"], METADATA["description"])
        self.assertEqual(head.metadata["author"], "Pooya Eimandar")
        self.assertEqual(head.metadata["robots"], "index, follow, max-image-preview:large")
        self.assertEqual(head.metadata["og:type"], "article")
        self.assertEqual(head.metadata["og:url"], canonical)
        self.assertEqual(head.metadata["og:image"], image_url)
        self.assertEqual(head.metadata["og:image:alt"], METADATA["imageAlt"])
        self.assertEqual(head.metadata["twitter:card"], "summary_large_image")
        self.assertEqual(head.metadata["twitter:image"], image_url)
        data = self.json_ld(document)
        self.assertEqual(data["@type"], "BlogPosting")
        self.assertEqual(data["headline"], METADATA["title"])
        self.assertEqual(data["author"], {
            "@type": "Person", "name": "Pooya Eimandar", "url": "https://pooya.ai"
        })
        self.assertEqual(data["mainEntityOfPage"]["@id"], canonical)
        self.assertEqual(data["url"], canonical)
        self.assertEqual(data["image"], image_url)
        self.assertEqual(data["inLanguage"], "en")
        self.assertNotIn("datePublished", data)
        self.assertNotIn("dateModified", data)
        self.assertIsNone(renderer.TOKEN_PATTERN.search(document))
        self.assertIn('./triangle.js?build=123456', document)
        self.assertIn('./triangle_bg.wasm?build=123456', document)

    def test_breadcrumb_schema_matches_generated_navigation(self):
        self.add_article(dict(METADATA, breadcrumbName="Triangle"))
        document = self.render()
        breadcrumbs = self.json_ld(document, "BreadcrumbList")
        self.assertEqual(breadcrumbs["@id"], "https://pooya.ai/webgpu/triangle/#breadcrumbs")
        self.assertEqual(breadcrumbs["itemListElement"], [
            {
                "@type": "ListItem", "position": 1,
                "name": "WebGPU examples", "item": "https://pooya.ai/webgpu/",
            },
            {
                "@type": "ListItem", "position": 2,
                "name": "Triangle", "item": "https://pooya.ai/webgpu/triangle/",
            },
        ])
        self.assertIn('<a href="../">WebGPU examples</a>', document)
        self.assertIn('<li aria-current="page">Triangle</li>', document)
        self.assertEqual(
            self.json_ld(document)["mainEntityOfPage"]["breadcrumb"]["@id"],
            breadcrumbs["@id"],
        )

    def test_breadcrumb_name_defaults_to_the_article_title(self):
        self.add_article()
        document = self.render()
        self.assertEqual(
            self.json_ld(document, "BreadcrumbList")["itemListElement"][-1]["name"],
            METADATA["title"],
        )
        self.assertIn(f'<li aria-current="page">{METADATA["title"]}</li>', document)

    def test_articles_without_visible_breadcrumbs_do_not_emit_breadcrumb_schema(self):
        self.add_article(fragment=ARTICLE.replace("__ARTICLE_BREADCRUMBS__", ""))
        document = self.render()
        self.assertNotIn("BreadcrumbList", document)
        self.assertNotIn("breadcrumb", self.json_ld(document)["mainEntityOfPage"])

    def test_image_dimensions_are_optional_and_exported_for_social_previews(self):
        self.add_article()
        self.assertNotIn("og:image:width", HeadParser(self.render()).metadata)
        self.add_article(dict(METADATA, imageWidth=1280, imageHeight=732))
        head = HeadParser(self.render())
        self.assertEqual(head.metadata["og:image:width"], "1280")
        self.assertEqual(head.metadata["og:image:height"], "732")

    def test_image_dimensions_and_optional_breadcrumb_name_are_validated(self):
        for fields in (
            {"imageWidth": 1280}, {"imageHeight": 732},
            {"imageWidth": 0, "imageHeight": 732},
            {"imageWidth": 1280, "imageHeight": -1},
            {"imageWidth": "1280", "imageHeight": 732},
            {"imageWidth": True, "imageHeight": 732},
            {"breadcrumbName": ""}, {"breadcrumbName": None},
        ):
            with self.subTest(fields=fields):
                self.add_article(dict(METADATA, **fields))
                with self.assertRaisesRegex(ValueError, "imageWidth|imageHeight|breadcrumbName"):
                    self.render()

    def test_image_url_and_alt_text_are_safe_in_article_html(self):
        self.add_article(dict(METADATA, image='triangle <&".jpg', imageAlt='RGB <triangle> & "colors"'))
        head = HeadParser(self.render())
        self.assertEqual(head.images[0]["src"], "../screenshots/triangle%20%3C%26%22.jpg")
        self.assertEqual(head.images[0]["alt"], 'RGB <triangle> & "colors"')
        self.assertEqual(urljoin(head.canonical, head.images[0]["src"]), head.metadata["og:image"])

    def test_no_article_preserves_the_fullscreen_demo_shell(self):
        document = self.render("texture")
        self.assertIn('<html lang="en" class="">', document)
        self.assertIn('<title>webgpu texture</title>', document)
        self.assertNotIn('class="more-info"', document)
        self.assertNotIn('<article', document)
        self.assertNotIn('application/ld+json', document)
        self.assertIn('./texture_bg.wasm?build=123456', document)
        self.assertIsNone(renderer.TOKEN_PATTERN.search(document))

    def test_copyright_fallback_uses_the_current_build_year(self):
        self.add_article()
        with patch.object(renderer, "date") as current_date:
            current_date.today.return_value.year = 2037
            document = self.render()
        self.assertIn(
            '&copy; <span data-current-year>2037</span> Pooya Eimandar. All rights reserved.',
            document,
        )

    def test_partial_article_fails_instead_of_silently_disappearing(self):
        for missing_suffix in (".json", ".html"):
            with self.subTest(missing=missing_suffix):
                self.add_article()
                (self.web_root / "articles" / f"triangle{missing_suffix}").unlink()
                with self.assertRaisesRegex(ValueError, rf"Incomplete article.*triangle\{missing_suffix}"):
                    self.render()

    def test_metadata_cannot_inject_html_or_close_the_json_ld_script(self):
        dangerous = 'Rust <3 & "WGSL" </script><script>alert(1)</script>\u2028'
        metadata = dict(
            METADATA, title=dangerous, description=dangerous,
            imageAlt=dangerous, breadcrumbName=dangerous,
        )
        self.add_article(metadata)
        document = self.render()
        head = HeadParser(document)
        self.assertEqual(head.metadata["description"], dangerous)
        self.assertEqual(head.metadata["og:image:alt"], dangerous)
        self.assertEqual(self.json_ld(document)["headline"], dangerous)
        self.assertEqual(self.json_ld(document, "BreadcrumbList")["itemListElement"][-1]["name"], dangerous)
        self.assertEqual(head.images[0]["alt"], dangerous)
        self.assertNotIn('<script>alert(1)</script>', document)
        self.assertIn('&lt;/script&gt;&lt;script&gt;', document)
        self.assertEqual(len(head.scripts), 3)

    def test_inserted_metadata_is_not_reinterpreted_as_template_tokens(self):
        title = "The literal __EXAMPLE__ token"
        self.add_article(dict(METADATA, title=title))
        document = self.render()
        self.assertIn(f"<h1>{title}</h1>", document)
        self.assertEqual(self.json_ld(document)["headline"], title)

    def test_invalid_slugs_cannot_escape_the_articles_directory(self):
        for example in ("../triangle", "/triangle", "triangle/other", "triangle.js", "<script>", "", "Triangle"):
            with self.subTest(example=example), self.assertRaisesRegex(ValueError, "Invalid example slug"):
                self.render(example)

    def test_build_id_is_encoded_for_the_asset_urls(self):
        document = self.render(build_id='x&y="</script>')
        self.assertIn('build=x%26y%3D%22%3C%2Fscript%3E', document)
        self.assertNotIn('build=x&y=', document)

    def test_metadata_requires_nonempty_strings_and_a_screenshot_basename(self):
        for field, value in (("title", ""), ("description", None), ("imageAlt", 3), ("image", "../triangle.jpg")):
            with self.subTest(field=field):
                self.add_article(dict(METADATA, **{field: value}))
                with self.assertRaisesRegex(ValueError, field):
                    self.render()

    def test_unknown_article_or_page_tokens_fail_with_context(self):
        self.add_article(fragment=ARTICLE + "__MISSING_TOKEN__")
        with self.assertRaisesRegex(ValueError, "Unresolved template token.*triangle.html"):
            self.render()
        self.add_article()
        (self.web_root / "example.html").write_text(TEMPLATE + "__MISSING_TOKEN__", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unresolved template token.*example.html"):
            self.render()

    def test_repository_pages_render_with_resolvable_article_navigation(self):
        examples = renderer.WEB_ROOT.parent / "examples"
        for source in sorted(examples.glob("*.rs")):
            with self.subTest(example=source.stem):
                document = renderer.render_example(source.stem, "integration-test")
                page = HeadParser(document)
                self.assertIsNone(renderer.TOKEN_PATTERN.search(document))
                self.assertEqual(len(page.ids), len(set(page.ids)), "Duplicate element IDs")
                for target in page.fragment_links:
                    self.assertIn(target, page.ids, f"Missing anchor target #{target}")
                has_article = (renderer.WEB_ROOT / "articles" / f"{source.stem}.html").exists()
                self.assertEqual(page.h1_count, 1 if has_article else 0)
                self.assertEqual("about" in page.ids, has_article)

    def test_repository_articles_have_crawlable_content_and_consistent_metadata(self):
        for metadata_path in sorted((renderer.WEB_ROOT / "articles").glob("*.json")):
            with self.subTest(example=metadata_path.stem):
                example = metadata_path.stem
                document = renderer.render_example(example, "seo-test")
                page = HeadParser(document)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(page.canonical, f"https://pooya.ai/webgpu/{example}/")
                self.assertEqual(page.metadata["og:url"], page.canonical)
                self.assertEqual(page.metadata["description"], metadata["description"])
                self.assertEqual(page.metadata["og:title"], metadata["title"])
                self.assertEqual(page.metadata["twitter:title"], metadata["title"])
                self.assertEqual(self.json_ld(document)["headline"], metadata["title"])
                self.assertEqual(page.h1_count, 1)
                self.assertEqual(len(page.images), 1)
                screenshot = page.images[0]
                self.assertEqual(screenshot["alt"], metadata["imageAlt"])
                self.assertEqual(int(screenshot["width"]), metadata["imageWidth"])
                self.assertEqual(int(screenshot["height"]), metadata["imageHeight"])
                self.assertEqual(urljoin(page.canonical, screenshot["src"]), page.metadata["og:image"])
                self.assertTrue((renderer.WEB_ROOT.parent / "screenshots" / metadata["image"]).is_file())
                self.assertIn("data-nosnippet", page.elements_by_id["loading-screen"])
                self.assertNotIn("data-nosnippet", page.elements_by_id["about"])
                self.assertGreater(len(page.h2_ids), 0)
                for target in page.h2_ids:
                    self.assertIn(target, page.fragment_links)
                self.assertIn("../", page.links)
                self.assertIn("https://github.com/PooyaEimandar/sib", page.links)
                self.assertIn("https://pooya.ai", page.links)
                self.assertEqual(
                    self.json_ld(document, "BreadcrumbList")["itemListElement"][-1]["name"],
                    metadata["breadcrumbName"],
                )

    def test_repository_triangle_keeps_its_source_and_video_links(self):
        document = renderer.render_example("triangle", "seo-test")
        page = HeadParser(document)
        self.assertEqual(len(page.h2_ids), 5)
        self.assertIn("../vertexattributes/", page.links)
        self.assertIn("https://youtu.be/VswCpKw4fc8?si=DQ5V61z_IxTkm_Q4", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/examples/triangle.rs", page.links)

    def test_repository_vertex_attributes_has_its_own_article_and_navigation(self):
        document = renderer.render_example("vertexattributes", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "two-buffer-layouts", "interleaved-vertex-buffer", "separate-attribute-buffers",
            "wgsl-vertex-inputs", "choosing-a-layout", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "vertexattributes")
        self.assertIn("attribute-layout-caption", page.ids)
        self.assertIn('role="region" aria-labelledby="attribute-layout-caption" tabindex="0"', document)
        self.assertIn("../triangle/#about", page.links)
        self.assertIn("../triangle/", page.links)
        self.assertIn("../texture/", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/examples/vertexattributes.rs", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/shaders/vertexattributes.wgsl", page.links)
        self.assertIn("cargo run --example vertexattributes", document)
        self.assertIn("scripts/build-wasm.sh --release vertexattributes", document)
        self.assertIn("./vertexattributes.js?build=seo-test", document)
        self.assertIn("./vertexattributes_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/triangle", document)

    def test_repository_texture_has_its_own_article_and_navigation(self):
        document = renderer.render_example("texture", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "textured-quad", "uv-coordinates", "texture-upload",
            "texture-bind-group", "wgsl-texture-sampling", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "texture")
        self.assertIn("texture-bindings-caption", page.ids)
        self.assertIn('role="region" aria-labelledby="texture-bindings-caption" tabindex="0"', document)
        self.assertIn("../vertexattributes/#about", page.links)
        self.assertIn("../vertexattributes/", page.links)
        self.assertIn("../texturecubemap/", page.links)
        self.assertIn("../texturemipmapgen/", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/examples/texture.rs", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texture.wgsl", page.links)
        self.assertIn("cargo run --example texture", document)
        self.assertIn("scripts/build-wasm.sh --release texture", document)
        self.assertIn("./texture.js?build=seo-test", document)
        self.assertIn("./texture_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/vertexattributes", document)

    def test_repository_cubemap_has_its_own_article_and_navigation(self):
        document = renderer.render_example("texturecubemap", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "skybox-and-reflections", "cubemap-faces", "cube-texture-view",
            "skybox-depth", "environment-reflections", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "texturecubemap")
        self.assertIn("cubemap-faces-caption", page.ids)
        self.assertIn('role="region" aria-labelledby="cubemap-faces-caption" tabindex="0"', document)
        self.assertIn("../texture/#about", page.links)
        self.assertIn("../texture/", page.links)
        self.assertIn("../texturearray/", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/examples/texturecubemap.rs", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texturecubemap.wgsl", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/src/skybox.rs", page.links)
        self.assertIn("cargo run --example texturecubemap", document)
        self.assertIn("scripts/build-wasm.sh --release texturecubemap", document)
        self.assertIn("./texturecubemap.js?build=seo-test", document)
        self.assertIn("./texturecubemap_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/texture.jpg", document)

    def test_repository_texture_array_has_its_own_article_and_navigation(self):
        document = renderer.render_example("texturearray", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "seven-layers-one-draw", "preparing-array-layers", "texture-array-view",
            "instanced-quads", "wgsl-layer-selection", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "texturearray")
        self.assertIn("texture-array-layers-caption", page.ids)
        self.assertIn('role="region" aria-labelledby="texture-array-layers-caption" tabindex="0"', document)
        self.assertIn("../texturecubemap/#about", page.links)
        self.assertIn("../texturecubemap/", page.links)
        self.assertIn("../texturemipmapgen/", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/examples/texturearray.rs", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texturearray.wgsl", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs", page.links)
        self.assertIn("cargo run --example texturearray", document)
        self.assertIn("scripts/build-wasm.sh --release texturearray", document)
        self.assertIn("./texturearray.js?build=seo-test", document)
        self.assertIn("./texturearray_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/texturecubemap", document)

    def test_repository_mipmap_generation_has_its_own_article_and_navigation(self):
        document = renderer.render_example("texturemipmapgen", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "why-mipmaps", "allocate-mip-chain", "generate-mipmaps",
            "mipmap-filtering", "sample-the-tunnel", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "texturemipmapgen")
        self.assertIn("mipmap-samplers-caption", page.ids)
        self.assertIn('role="region" aria-labelledby="mipmap-samplers-caption" tabindex="0"', document)
        self.assertIn("../texturearray/#about", page.links)
        self.assertIn("../texturearray/", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/examples/texturemipmapgen.rs", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texturemipmapgen.wgsl", page.links)
        self.assertIn("https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texturemipmapgen_mipmap.wgsl", page.links)
        self.assertIn("cargo run --example texturemipmapgen", document)
        self.assertIn("scripts/build-wasm.sh --release texturemipmapgen", document)
        self.assertIn("./texturemipmapgen.js?build=seo-test", document)
        self.assertIn("./texturemipmapgen_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/texturearray", document)


if __name__ == "__main__":
    unittest.main()
