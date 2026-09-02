#!/usr/bin/env python3
"""Tests for optional article rendering; no WebGPU build or dependencies required."""

import importlib.util
import html
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
            self.metadata[attrs.get("name", attrs.get(
                "property"))] = attrs.get("content")
        elif tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href")
        elif tag == "script":
            self.scripts.append(attrs)


class RenderExampleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="webgpu-article-test-")
        self.addCleanup(self.temporary.cleanup)
        self.web_root = Path(self.temporary.name)
        (self.web_root / "example.html").write_text(TEMPLATE, encoding="utf-8")
        (self.web_root / "articles").mkdir()

    def add_article(self, metadata=None, fragment=ARTICLE):
        article_path = self.web_root / "articles" / "triangle"
        article_path.with_suffix(".json").write_text(
            json.dumps(METADATA if metadata is None else metadata), encoding="utf-8"
        )
        article_path.with_suffix(".html").write_text(
            fragment, encoding="utf-8")

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
        self.assertIn(
            f'<title>{METADATA["title"]} | Pooya Eimandar</title>', document)
        self.assertEqual(head.canonical, canonical)
        self.assertEqual(head.metadata["description"], METADATA["description"])
        self.assertEqual(head.metadata["author"], "Pooya Eimandar")
        self.assertEqual(head.metadata["robots"],
                         "index, follow, max-image-preview:large")
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
        self.assertEqual(
            breadcrumbs["@id"], "https://pooya.ai/webgpu/triangle/#breadcrumbs")
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
            self.json_ld(document, "BreadcrumbList")[
                "itemListElement"][-1]["name"],
            METADATA["title"],
        )
        self.assertIn(
            f'<li aria-current="page">{METADATA["title"]}</li>', document)

    def test_articles_without_visible_breadcrumbs_do_not_emit_breadcrumb_schema(self):
        self.add_article(fragment=ARTICLE.replace(
            "__ARTICLE_BREADCRUMBS__", ""))
        document = self.render()
        self.assertNotIn("BreadcrumbList", document)
        self.assertNotIn("breadcrumb", self.json_ld(
            document)["mainEntityOfPage"])

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
        self.add_article(dict(METADATA, image='triangle <&".jpg',
                         imageAlt='RGB <triangle> & "colors"'))
        head = HeadParser(self.render())
        self.assertEqual(head.images[0]["src"],
                         "../screenshots/triangle%20%3C%26%22.jpg")
        self.assertEqual(head.images[0]["alt"], 'RGB <triangle> & "colors"')
        self.assertEqual(
            urljoin(head.canonical, head.images[0]["src"]), head.metadata["og:image"])

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
                (self.web_root / "articles" /
                 f"triangle{missing_suffix}").unlink()
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
        self.assertEqual(self.json_ld(document, "BreadcrumbList")[
                         "itemListElement"][-1]["name"], dangerous)
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
        (self.web_root / "example.html").write_text(TEMPLATE +
                                                    "__MISSING_TOKEN__", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unresolved template token.*example.html"):
            self.render()

    def test_repository_pages_render_with_resolvable_article_navigation(self):
        examples = renderer.WEB_ROOT.parent / "examples"
        for source in sorted(examples.glob("*.rs")):
            with self.subTest(example=source.stem):
                document = renderer.render_example(
                    source.stem, "integration-test")
                page = HeadParser(document)
                self.assertIsNone(renderer.TOKEN_PATTERN.search(document))
                self.assertEqual(len(page.ids), len(
                    set(page.ids)), "Duplicate element IDs")
                for target in page.fragment_links:
                    self.assertIn(target, page.ids,
                                  f"Missing anchor target #{target}")
                has_article = (renderer.WEB_ROOT / "articles" /
                               f"{source.stem}.html").exists()
                self.assertEqual(page.h1_count, 1 if has_article else 0)
                self.assertEqual("about" in page.ids, has_article)

    def test_repository_article_footers_follow_the_complete_reading_order(self):
        articles = [
            ("triangle", "WebGPU triangle"),
            ("vertexattributes", "WebGPU vertex attributes"),
            ("texture", "WebGPU texture loading"),
            ("texturecubemap", "WebGPU texture cubemap"),
            ("texturearray", "WebGPU texture array"),
            ("texturemipmapgen", "WebGPU mipmap generation"),
            ("textoverlay", "WebGPU text overlay"),
            ("textmesh", "WebGPU 3D text mesh"),
            ("htmlmesh", "WebGPU HTML mesh"),
            ("gltf", "WebGPU glTF 2.0"),
            ("gears", "Procedural WebGPU gears"),
            ("stencilbuffer", "WebGPU stencil buffer outlines"),
            ("gltfskinning", "WebGPU glTF vertex skinning"),
            ("instancing", "WebGPU asteroid instancing"),
            ("indirectdraw", "WebGPU indirect draw"),
            ("pipelines", "WebGPU multiple render pipelines"),
            ("particlesystem", "WebGPU CPU particle system"),
            ("occlusionquery", "WebGPU occlusion queries"),
            ("radialblur", "WebGPU radial blur"),
            ("bloom", "WebGPU bloom"),
            ("shadowmapping", "WebGPU shadow mapping"),
            ("shadowmappingcascade", "WebGPU cascaded shadow mapping"),
            ("shadowmappingomni", "WebGPU omnidirectional shadow mapping"),
            ("pbr", "WebGPU PBR Basic"),
            ("pbrtexture", "WebGPU PBR texture"),
            ("pbribl", "WebGPU PBR image-based lighting"),
            ("parallaxmapping", "WebGPU parallax occlusion mapping"),
            ("multisampling", "WebGPU 4x MSAA multisampling"),
            ("multisamplingalphatocoverage", "WebGPU alpha-to-coverage"),
            ("deferred", "WebGPU deferred shading"),
            ("deferredmultisampling", "WebGPU deferred multisampling"),
            ("deferredshadows", "WebGPU deferred shadows"),
            ("ssao", "WebGPU screen-space ambient occlusion"),
            ("computeparticles", "WebGPU compute particles"),
            ("computecloth", "WebGPU compute cloth simulation"),
            ("computecullandlod", "WebGPU compute culling and LOD"),
            ("computenbody", "WebGPU N-body simulation"),
            ("computeraytracing", "WebGPU compute shader ray tracing"),
            ("raytracingshadows", "WebGPU ray-traced shadows"),
            ("raytracingreflections", "WebGPU ray-traced reflections"),
            ("raytracinggltf", "WebGPU glTF ray tracing"),
            ("nanite", "WebGPU Nanite-style mesh rendering"),
            ("metropolis", "WebGPU Metropolis renderer"),
            ("restirdi", "WebGPU ReSTIR direct illumination"),
            ("restirgi", "WebGPU ReSTIR global illumination"),
            ("residentevil2", "WebGPU Resident Evil fixed-camera scene"),
            ("geometrydash", "WebGPU Geometry Dash game"),
        ]
        source_slugs = {
            source.stem for source in (renderer.WEB_ROOT.parent / "examples").glob("*.rs")
        }
        self.assertEqual({slug for slug, _ in articles}, source_slugs)
        self.assertEqual(len(articles), 47)

        for index, (example, _) in enumerate(articles):
            with self.subTest(example=example):
                article_path = renderer.WEB_ROOT / \
                    "articles" / f"{example}.html"
                metadata_path = article_path.with_suffix(".json")
                self.assertTrue(article_path.is_file(), "Missing article HTML")
                self.assertTrue(metadata_path.is_file(),
                                "Missing article metadata")
                source = article_path.read_text(encoding="utf-8")
                self.assertIn(
                    f'<p class="eyebrow">WebGPU notes &nbsp; / &nbsp; {index + 1:02d}</p>',
                    source,
                )
                footer_match = re.search(
                    r'<footer class="article-footer">(.*?)</footer>', source, re.S
                )
                self.assertIsNotNone(footer_match, "Missing article footer")
                footer = footer_match.group(1)
                previous = re.findall(
                    r'<a href="([^"]+)">&larr; Previous: ([^<]+)</a>', footer
                )
                following = re.findall(
                    r'<a href="([^"]+)">Next: ([^<]+) &rarr;</a>', footer
                )

                if index == 0:
                    self.assertEqual(
                        previous, [], "The first article has no predecessor")
                else:
                    previous_slug, previous_label = articles[index - 1]
                    self.assertEqual(
                        previous,
                        [(f"../{previous_slug}/", previous_label)],
                    )

                if index + 1 < len(articles):
                    next_slug, next_label = articles[index + 1]
                    self.assertEqual(
                        following, [(f"../{next_slug}/", next_label)])
                    self.assertNotEqual(next_slug, example)
                    self.assertTrue(
                        (renderer.WEB_ROOT.parent / "examples" /
                         f"{next_slug}.rs").is_file(),
                        f"Next destination {next_slug} has no example source",
                    )
                else:
                    self.assertEqual(
                        following, [("../", "Browse all WebGPU examples")])

    def test_repository_articles_have_crawlable_content_and_consistent_metadata(self):
        for metadata_path in sorted((renderer.WEB_ROOT / "articles").glob("*.json")):
            with self.subTest(example=metadata_path.stem):
                example = metadata_path.stem
                document = renderer.render_example(example, "seo-test")
                page = HeadParser(document)
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    page.canonical, f"https://pooya.ai/webgpu/{example}/")
                self.assertEqual(page.metadata["og:url"], page.canonical)
                self.assertEqual(
                    page.metadata["description"], metadata["description"])
                self.assertEqual(page.metadata["og:title"], metadata["title"])
                self.assertEqual(
                    page.metadata["twitter:title"], metadata["title"])
                self.assertEqual(self.json_ld(document)[
                                 "headline"], metadata["title"])
                self.assertEqual(page.h1_count, 1)
                self.assertEqual(len(page.images), 1)
                screenshot = page.images[0]
                self.assertEqual(screenshot["alt"], metadata["imageAlt"])
                self.assertEqual(
                    int(screenshot["width"]), metadata["imageWidth"])
                self.assertEqual(
                    int(screenshot["height"]), metadata["imageHeight"])
                self.assertEqual(
                    urljoin(page.canonical, screenshot["src"]), page.metadata["og:image"])
                self.assertTrue((renderer.WEB_ROOT.parent /
                                "screenshots" / metadata["image"]).is_file())
                self.assertIn("data-nosnippet",
                              page.elements_by_id["loading-screen"])
                self.assertNotIn("data-nosnippet",
                                 page.elements_by_id["about"])
                self.assertGreater(len(page.h2_ids), 0)
                for target in page.h2_ids:
                    self.assertIn(target, page.fragment_links)
                self.assertIn("../", page.links)
                self.assertIn(
                    "https://github.com/PooyaEimandar/sib", page.links)
                self.assertIn("https://pooya.ai", page.links)
                self.assertEqual(
                    self.json_ld(document, "BreadcrumbList")[
                        "itemListElement"][-1]["name"],
                    metadata["breadcrumbName"],
                )

    def test_repository_gallery_matches_every_article_metadata_record(self):
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            "<title>Rust WebGPU Examples & Tutorials | wgpu &amp; WGSL</title>",
            gallery,
        )
        self.assertIn("<h1>Rust WebGPU examples & tutorials</h1>", gallery)
        self.assertIn(
            'content="index, follow, max-image-preview:large"', gallery)
        nodes = [
            json.loads(source) for source in re.findall(
                r'<script type="application/ld\+json">(.*?)</script>', gallery, re.S
            )
        ]
        collections = [node for node in nodes if node.get(
            "@type") == "CollectionPage"]
        self.assertEqual(len(collections), 1)
        collection = collections[0]
        self.assertEqual(collection["author"]["url"], "https://pooya.ai")
        items = collection["mainEntity"]["itemListElement"]
        self.assertEqual([item["position"]
                         for item in items], list(range(1, 48)))
        items_by_url = {item["url"]: item for item in items}
        self.assertEqual(len(items_by_url), 47)
        new_article_names = {
            "computecloth": "WebGPU compute cloth simulation",
            "computecullandlod": "WebGPU compute culling and LOD",
            "computeraytracing": "WebGPU compute shader ray tracing",
            "geometrydash": "WebGPU Geometry Dash-style game",
            "htmlmesh": "WebGPU HTML mesh",
            "metropolis": "WebGPU Metropolis renderer",
            "nanite": "WebGPU Nanite-style mesh rendering",
            "pipelines": "WebGPU multiple render pipelines",
            "raytracinggltf": "WebGPU glTF ray tracing",
            "raytracingreflections": "WebGPU ray-traced reflections",
            "raytracingshadows": "WebGPU ray-traced shadows",
            "residentevil2": "WebGPU Resident Evil fixed-camera scene",
            "restirdi": "WebGPU ReSTIR direct illumination",
            "restirgi": "WebGPU ReSTIR global illumination",
            "shadowmappingcascade": "WebGPU cascaded shadow mapping",
            "shadowmappingomni": "WebGPU omnidirectional shadow mapping",
        }

        metadata_paths = sorted(
            (renderer.WEB_ROOT / "articles").glob("*.json"))
        self.assertEqual(len(metadata_paths), 47)
        for metadata_path in metadata_paths:
            with self.subTest(example=metadata_path.stem):
                example = metadata_path.stem
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8"))
                url = f"https://pooya.ai/webgpu/{example}/"
                self.assertIn(url, items_by_url)
                self.assertTrue(items_by_url[url]["name"].strip())
                card_match = re.search(
                    rf'<a href="\./{re.escape(example)}/">(.*?)^      </a>',
                    gallery,
                    re.S | re.M,
                )
                self.assertIsNotNone(card_match, "Missing gallery card")
                card = card_match.group(1)
                self.assertRegex(card, r'<img [^>]*alt="[^"]+"')
                self.assertRegex(card, r'<strong>[^<]+</strong>')
                if example in new_article_names:
                    expected_name = new_article_names[example]
                    self.assertEqual(items_by_url[url]["name"], expected_name)
                    self.assertIn(
                        f'alt="{html.escape(metadata["imageAlt"], quote=True)}"',
                        card,
                    )
                    self.assertIn(
                        f'<strong>{html.escape(expected_name)}</strong>',
                        card,
                    )

    def test_new_article_batch_has_complete_blog_structure_and_source_context(self):
        examples = (
            "computecloth", "computecullandlod", "computeraytracing",
            "geometrydash", "htmlmesh", "metropolis", "nanite", "pipelines",
            "raytracinggltf", "raytracingreflections", "raytracingshadows",
            "residentevil2", "restirdi", "restirgi", "shadowmappingcascade",
            "shadowmappingomni",
        )
        for example in examples:
            with self.subTest(example=example):
                fragment_path = renderer.WEB_ROOT / \
                    "articles" / f"{example}.html"
                metadata_path = fragment_path.with_suffix(".json")
                self.assertTrue(fragment_path.is_file())
                self.assertTrue(metadata_path.is_file())
                fragment = fragment_path.read_text(encoding="utf-8")
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8"))
                document = renderer.render_example(example, "batch-qa")
                page = HeadParser(document)

                self.assertGreaterEqual(len(page.h2_ids), 6)
                self.assertEqual(len(page.h2_ids), len(set(page.h2_ids)))
                self.assertIn('class="article-toc"', fragment)
                self.assertIn('class="article-figure"', fragment)
                self.assertGreaterEqual(fragment.count(
                    '<div class="article-table"'), 1)
                self.assertIn(
                    f"https://github.com/PooyaEimandar/webgpu/blob/main/examples/{example}.rs",
                    page.links,
                )
                self.assertTrue(
                    any(link.startswith("https://github.com/PooyaEimandar/webgpu/commit/")
                        for link in page.links),
                    "Missing source-history link",
                )
                self.assertIn(f"cargo run --example {example}", document)
                self.assertIn(
                    f"scripts/build-wasm.sh --release {example}", document)
                self.assertIn("cargo run --bin serve", document)
                self.assertIn('<footer class="article-footer">', fragment)
                self.assertIn("data-current-year", fragment)
                self.assertGreaterEqual(len(metadata["title"]), 30)
                self.assertLessEqual(len(metadata["title"]), 70)
                self.assertGreaterEqual(len(metadata["description"]), 120)
                self.assertLessEqual(len(metadata["description"]), 165)

    def test_repository_triangle_keeps_its_source_and_video_links(self):
        document = renderer.render_example("triangle", "seo-test")
        page = HeadParser(document)
        self.assertEqual(len(page.h2_ids), 5)
        self.assertIn("../vertexattributes/", page.links)
        self.assertIn(
            "https://youtu.be/VswCpKw4fc8?si=DQ5V61z_IxTkm_Q4", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/triangle.rs", page.links)

    def test_repository_vertex_attributes_has_its_own_article_and_navigation(self):
        document = renderer.render_example("vertexattributes", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "two-buffer-layouts", "interleaved-vertex-buffer", "separate-attribute-buffers",
            "wgsl-vertex-inputs", "choosing-a-layout", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "vertexattributes")
        self.assertIn("attribute-layout-caption", page.ids)
        self.assertIn(
            'role="region" aria-labelledby="attribute-layout-caption" tabindex="0"', document)
        self.assertIn("../triangle/#about", page.links)
        self.assertIn("../triangle/", page.links)
        self.assertIn("../texture/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/vertexattributes.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/vertexattributes.wgsl", page.links)
        self.assertIn("cargo run --example vertexattributes", document)
        self.assertIn(
            "scripts/build-wasm.sh --release vertexattributes", document)
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
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "texture")
        self.assertIn("texture-bindings-caption", page.ids)
        self.assertIn(
            'role="region" aria-labelledby="texture-bindings-caption" tabindex="0"', document)
        self.assertIn("../vertexattributes/#about", page.links)
        self.assertIn("../vertexattributes/", page.links)
        self.assertIn("../texturecubemap/", page.links)
        self.assertIn("../texturemipmapgen/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/texture.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texture.wgsl", page.links)
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
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "texturecubemap")
        self.assertIn("cubemap-faces-caption", page.ids)
        self.assertIn(
            'role="region" aria-labelledby="cubemap-faces-caption" tabindex="0"', document)
        self.assertIn("../texture/#about", page.links)
        self.assertIn("../texture/", page.links)
        self.assertIn("../texturearray/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/texturecubemap.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texturecubemap.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/skybox.rs", page.links)
        self.assertIn("cargo run --example texturecubemap", document)
        self.assertIn(
            "scripts/build-wasm.sh --release texturecubemap", document)
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
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "texturearray")
        self.assertIn("texture-array-layers-caption", page.ids)
        self.assertIn(
            'role="region" aria-labelledby="texture-array-layers-caption" tabindex="0"', document)
        self.assertIn("../texturecubemap/#about", page.links)
        self.assertIn("../texturecubemap/", page.links)
        self.assertIn("../texturemipmapgen/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/texturearray.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texturearray.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs", page.links)
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
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "texturemipmapgen")
        self.assertIn("mipmap-samplers-caption", page.ids)
        self.assertIn(
            'role="region" aria-labelledby="mipmap-samplers-caption" tabindex="0"', document)
        self.assertIn("../texturearray/#about", page.links)
        self.assertIn("../texturearray/", page.links)
        self.assertIn("../textoverlay/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/texturemipmapgen.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texturemipmapgen.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/texturemipmapgen_mipmap.wgsl", page.links)
        self.assertIn("cargo run --example texturemipmapgen", document)
        self.assertIn(
            "scripts/build-wasm.sh --release texturemipmapgen", document)
        self.assertIn("./texturemipmapgen.js?build=seo-test", document)
        self.assertIn("./texturemipmapgen_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/texturearray", document)

    def test_repository_text_overlay_has_its_own_article_and_navigation(self):
        document = renderer.render_example("textoverlay", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "scene-and-overlay", "fonts-and-glyph-atlas", "text-style-and-placement",
            "unicode-and-rtl", "prepare-and-render", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "textoverlay")
        self.assertIn(
            'role="region" aria-labelledby="text-overlay-styles-caption" tabindex="0"', document)
        self.assertEqual(
            page.elements_by_id["persian-text-sample"]["lang"], "fa")
        self.assertEqual(
            page.elements_by_id["persian-text-sample"]["dir"], "rtl")
        self.assertIn("سلام ایران", document)
        self.assertIn("متن راست به چپ", document)
        self.assertIn("../texturemipmapgen/#about", page.links)
        self.assertIn("../texturemipmapgen/", page.links)
        self.assertIn("../textmesh/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/textoverlay.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/textoverlay.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/sib/blob/960c39bcde152f50c87fa3926c8bc8ff53e2b5eb/src/render/text.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/fonts/Vazirmatn-Regular.ttf", page.links)
        self.assertIn("cargo run --example textoverlay", document)
        self.assertIn("scripts/build-wasm.sh --release textoverlay", document)
        self.assertIn("./textoverlay.js?build=seo-test", document)
        self.assertIn("./textoverlay_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/texturemipmapgen", document)

    def test_repository_text_mesh_has_its_own_article_and_navigation(self):
        document = renderer.render_example("textmesh", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "text-as-geometry", "shape-the-text", "extrude-font-outlines",
            "combine-text-meshes", "render-and-light", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "textmesh")
        self.assertIn(
            'role="region" aria-labelledby="text-mesh-options-caption" tabindex="0"', document)
        self.assertEqual(
            page.elements_by_id["persian-mesh-sample"]["lang"], "fa")
        self.assertEqual(
            page.elements_by_id["persian-mesh-sample"]["dir"], "rtl")
        self.assertIn("هی وب جی پی یو!", document)
        self.assertIn("../textoverlay/#about", page.links)
        self.assertIn("../textoverlay/", page.links)
        self.assertIn("../htmlmesh/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/textmesh.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/textmesh.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/sib/blob/960c39bcde152f50c87fa3926c8bc8ff53e2b5eb/src/render/text_mesh.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/fonts/Vazirmatn-Regular.ttf", page.links)
        self.assertIn("cargo run --example textmesh", document)
        self.assertIn("scripts/build-wasm.sh --release textmesh", document)
        self.assertIn("./textmesh.js?build=seo-test", document)
        self.assertIn("./textmesh_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/textoverlay", document)

    def test_repository_gltf_has_its_own_article_and_navigation(self):
        document = renderer.render_example("gltf", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "gltf-to-webgpu", "fetch-gltf-resources", "traverse-and-merge",
            "material-and-texture", "render-with-wgsl", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "gltf")
        self.assertIn(
            'role="region" aria-labelledby="gltf-asset-caption" tabindex="0"', document)
        self.assertIn("../htmlmesh/#about", page.links)
        self.assertIn("../htmlmesh/", page.links)
        self.assertIn("../gears/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/gltf.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_scene.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/gltf.wgsl", page.links)
        self.assertIn(
            "https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/BoxTextured/glTF", page.links)
        self.assertIn("cargo run --example gltf", document)
        self.assertIn("scripts/build-wasm.sh --release gltf", document)
        self.assertIn("./gltf.js?build=seo-test", document)
        self.assertIn("./gltf_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/textmesh", document)

    def test_repository_gears_has_its_own_article_and_navigation(self):
        document = renderer.render_example("gears", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "procedural-gears", "gear-specifications", "generate-gear-mesh",
            "shared-buffers", "animate-and-light", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "gears")
        self.assertIn(
            'role="region" aria-labelledby="gear-specs-caption" tabindex="0"', document)
        self.assertIn("../gltf/#about", page.links)
        self.assertIn("../gltf/", page.links)
        self.assertIn("../stencilbuffer/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/gears.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/gears.wgsl", page.links)
        self.assertIn("cargo run --example gears", document)
        self.assertIn("scripts/build-wasm.sh --release gears", document)
        self.assertIn("1,600 vertices", document)
        self.assertIn("2,640", document)
        self.assertIn("880 submitted triangles", document)
        self.assertIn("800 nondegenerate triangles", document)
        self.assertIn("./gears.js?build=seo-test", document)
        self.assertIn("./gears_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/gltf", document)

    def test_repository_stencil_buffer_has_its_own_article_and_navigation(self):
        document = renderer.render_example("stencilbuffer", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "stencil-outline", "load-venus-mesh", "write-stencil-mask",
            "draw-expanded-outline", "depth-stencil-pass", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "stencilbuffer")
        self.assertIn(
            'role="region" aria-labelledby="stencil-pipelines-caption" tabindex="0"', document)
        self.assertIn("../gears/#about", page.links)
        self.assertIn("../gears/", page.links)
        self.assertIn("../gltfskinning/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/stencilbuffer.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/stencilbuffer.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_scene.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/venus.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/975b07df28e2c91485548ed9ee351553af3483b0", page.links)
        self.assertIn("cargo run --example stencilbuffer", document)
        self.assertIn(
            "scripts/build-wasm.sh --release stencilbuffer", document)
        self.assertIn("31,398 triangles", document)
        self.assertIn("94,194", document)
        self.assertIn("Depth24PlusStencil8", document)
        self.assertIn("./stencilbuffer.js?build=seo-test", document)
        self.assertIn("./stencilbuffer_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/gears", document)

    def test_repository_gltf_skinning_has_its_own_article_and_navigation(self):
        document = renderer.render_example("gltfskinning", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "gpu-skinned-character", "load-jax-asset", "sample-animation",
            "build-joint-matrices", "skin-in-wgsl", "render-and-light",
            "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "gltfskinning")
        self.assertIn(
            'role="region" aria-labelledby="skinning-bindings-caption" tabindex="0"', document)
        self.assertIn("../gltf/#about", page.links)
        self.assertIn("../stencilbuffer/", page.links)
        self.assertIn("../instancing/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/gltfskinning.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/gltfskinning.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_skin.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.bin", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/jax_base_color.png", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/d58e37c192066f69c042fc9cbc3be0260eb0f58d", page.links)
        self.assertIn("cargo run --example gltfskinning", document)
        self.assertIn("scripts/build-wasm.sh --release gltfskinning", document)
        self.assertIn("35,880", document)
        self.assertIn("11,960", document)
        self.assertIn("46 skin joints", document)
        self.assertIn("Walking_1", document)
        self.assertIn("91 channels", document)
        self.assertIn("8,192-byte", document)
        self.assertIn("./gltfskinning.js?build=seo-test", document)
        self.assertIn("./gltfskinning_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/stencilbuffer", document)

    def test_repository_instancing_has_its_own_article_and_navigation(self):
        document = renderer.render_example("instancing", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "one-draw-thousands", "procedural-asteroid-mesh", "instance-buffer",
            "texture-array-layers", "animate-in-wgsl", "three-scene-draws",
            "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "instancing")
        self.assertIn(
            'role="region" aria-labelledby="instancing-draws-caption" tabindex="0"', document)
        self.assertIn("../gltfskinning/#about", page.links)
        self.assertIn("../gltfskinning/", page.links)
        self.assertIn("../indirectdraw/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/instancing.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/instancing.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/e541272cf963fe2f0c72dc15a3aff775f4ed7038", page.links)
        self.assertIn("cargo run --example instancing", document)
        self.assertIn("scripts/build-wasm.sh --release instancing", document)
        self.assertIn("2,048", document)
        self.assertIn("65,536-byte", document)
        self.assertIn("442,368", document)
        self.assertIn("Rgba8UnormSrgb", document)
        self.assertIn("Depth32Float", document)
        self.assertIn("./instancing.js?build=seo-test", document)
        self.assertIn("./instancing_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/gltfskinning", document)

    def test_repository_indirect_draw_has_its_own_article_and_navigation(self):
        document = renderer.render_example("indirectdraw", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "from-instancing-to-indirect", "load-plant-assets", "build-instance-groups",
            "encode-indirect-commands", "issue-indirect-draws", "texture-and-light-plants",
            "ground-sky-and-depth", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "indirectdraw")
        self.assertIn(
            'role="region" aria-labelledby="indirect-assets-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="indirect-pipelines-caption" tabindex="0"', document)
        self.assertIn("../instancing/#about", page.links)
        self.assertIn("../instancing/", page.links)
        self.assertIn("../pipelines/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/indirectdraw.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/indirectdraw.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/plants.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/texturearray_plants_rgba.ktx", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/49096c775bfc19702ebd7238f2119a767b87ab78", page.links)
        self.assertIn("cargo run --example indirectdraw", document)
        self.assertIn("scripts/build-wasm.sh --release indirectdraw", document)
        self.assertIn("24,576", document)
        self.assertIn("786,432-byte", document)
        self.assertIn("240-byte", document)
        self.assertIn("18,165,760", document)
        self.assertIn("Rgba8Unorm", document)
        self.assertIn("Depth32Float", document)
        self.assertIn("./indirectdraw.js?build=seo-test", document)
        self.assertIn("./indirectdraw_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/instancing", document)

    def test_repository_particle_system_has_its_own_article_and_navigation(self):
        document = renderer.render_example("particlesystem", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "cpu-particles-one-draw", "load-fireplace-assets", "initialize-emitter",
            "update-particles-on-cpu", "stream-instance-buffer", "billboard-sprites",
            "blend-fire-and-smoke", "light-fireplace", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "particlesystem")
        self.assertIn(
            'role="region" aria-labelledby="particle-assets-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="particle-buffers-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="particle-pipelines-caption" tabindex="0"', document)
        self.assertIn("../pipelines/#about", page.links)
        self.assertIn("../pipelines/", page.links)
        self.assertIn("../occlusionquery/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/particlesystem.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/particlesystem.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/fireplace.obj", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/fireplace_colormap_bc3.ktx", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/fireplace_normalmap_bc3.ktx", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/particle_fire.ktx", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/particle_smoke.ktx", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/7644ad9ca461bf78c5c286ad842b6681855180df", page.links)
        self.assertIn("cargo run --example particlesystem", document)
        self.assertIn(
            "scripts/build-wasm.sh --release particlesystem", document)
        self.assertIn("512", document)
        self.assertIn("24,576-byte", document)
        self.assertIn("24,992", document)
        self.assertIn("1,142", document)
        self.assertIn("Rgba8Unorm", document)
        self.assertIn("Depth32Float", document)
        self.assertIn("./particlesystem.js?build=seo-test", document)
        self.assertIn("./particlesystem_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/indirectdraw", document)

    def test_repository_occlusion_query_has_its_own_article_and_navigation(self):
        document = renderer.render_example("occlusionquery", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "test-visibility-with-queries", "load-occlusion-meshes",
            "create-query-resources", "record-query-pass", "resolve-and-map-results",
            "render-visible-scene", "browser-fallback", "move-the-camera",
            "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "occlusionquery")
        self.assertIn(
            'role="region" aria-labelledby="occlusion-assets-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="occlusion-resources-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="occlusion-passes-caption" tabindex="0"', document)
        self.assertIn("../particlesystem/#about", page.links)
        self.assertIn("../particlesystem/", page.links)
        self.assertIn("../radialblur/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/occlusionquery.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/occlusionquery.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_scene.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/joystick.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/plane_z.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/teapot.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/sphere.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/982afb7012500ff4cb26ae6cb6307e96e59dadbd", page.links)
        self.assertIn("cargo run --example occlusionquery", document)
        self.assertIn(
            "scripts/build-wasm.sh --release occlusionquery", document)
        self.assertIn("371,338", document)
        self.assertIn("447,424", document)
        self.assertIn("13,642", document)
        self.assertIn("27,284", document)
        self.assertIn("256 bytes", document)
        self.assertIn("16 bytes", document)
        self.assertIn("18_432", document)
        self.assertIn("USE_GPU_OCCLUSION_QUERY", document)
        self.assertIn("QUERY_RESOLVE", document)
        self.assertIn("MAP_READ", document)
        self.assertIn("Depth32Float", document)
        self.assertIn("./occlusionquery.js?build=seo-test", document)
        self.assertIn("./occlusionquery_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/particlesystem", document)

    def test_repository_radial_blur_has_its_own_article_and_navigation(self):
        document = renderer.render_example("radialblur", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "multipass-radial-glow", "load-glow-sphere-assets",
            "build-offscreen-target", "render-glow-mask", "render-phong-scene",
            "sample-radial-blur", "additive-composite", "animate-orbit-gradient",
            "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "radialblur")
        self.assertIn(
            'role="region" aria-labelledby="radial-assets-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="radial-resources-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="radial-passes-caption" tabindex="0"', document)
        self.assertIn("../occlusionquery/#about", page.links)
        self.assertIn("../occlusionquery/", page.links)
        self.assertIn("../bloom/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/radialblur.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/radialblur.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_scene.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/glowsphere.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/particle_gradient_rgba.ktx", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/7f54cadf3f26d9e07c19d16d9251c7832b91a44e", page.links)
        self.assertIn("cargo run --example radialblur", document)
        self.assertIn("scripts/build-wasm.sh --release radialblur", document)
        self.assertIn("147,558", document)
        self.assertIn("146,434", document)
        self.assertIn("1,124", document)
        self.assertIn("3,104", document)
        self.assertIn("3,120", document)
        self.assertIn("1,040", document)
        self.assertIn("136,640", document)
        self.assertIn("224-byte", document)
        self.assertIn("16-byte", document)
        self.assertIn("512", document)
        self.assertIn("32", document)
        self.assertIn("2,081", document)
        self.assertIn("Rgba8Unorm", document)
        self.assertIn("Rgba8UnormSrgb", document)
        self.assertIn("Depth32Float", document)
        self.assertIn("./radialblur.js?build=seo-test", document)
        self.assertIn("./radialblur_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/occlusionquery", document)

    def test_repository_bloom_has_its_own_article_and_navigation(self):
        document = renderer.render_example("bloom", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "separable-bloom-four-passes", "load-ufo-assets",
            "build-bloom-targets", "render-glow-mask", "blur-vertically",
            "render-lit-ufo", "composite-horizontal-blur", "animate-ufo",
            "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "bloom")
        self.assertIn(
            'role="region" aria-labelledby="bloom-assets-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="bloom-resources-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="bloom-passes-caption" tabindex="0"', document)
        self.assertIn("../radialblur/#about", page.links)
        self.assertIn("../radialblur/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/bloom.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/bloom.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_scene.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/retroufo.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/retroufo_glow.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/55132cbb36b7d1b2bccd842f20544d69f257db0f", page.links)
        self.assertIn("cargo run --example bloom", document)
        self.assertIn("scripts/build-wasm.sh --release bloom", document)
        self.assertIn("2,198,325", document)
        self.assertIn("1,095,375", document)
        self.assertIn("1,102,950", document)
        self.assertIn("19,990", document)
        self.assertIn("49,284", document)
        self.assertIn("16,428", document)
        self.assertIn("20,245", document)
        self.assertIn("50,148", document)
        self.assertIn("16,716", document)
        self.assertIn("40,235", document)
        self.assertIn("99,432", document)
        self.assertIn("33,144", document)
        self.assertIn("2,007,128", document)
        self.assertIn("33,146", document)
        self.assertIn("192-byte", document)
        self.assertIn("16-byte", document)
        self.assertIn("256", document)
        self.assertIn("9-tap", document)
        self.assertIn("Rgba8Unorm", document)
        self.assertIn("Depth32Float", document)
        self.assertIn("1.3864856", document)
        self.assertIn("./bloom.js?build=seo-test", document)
        self.assertIn("./bloom_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/radialblur", document)
        self.assertIn("../shadowmapping/", page.links)

    def test_repository_shadow_mapping_has_its_own_article_and_navigation(self):
        document = renderer.render_example("shadowmapping", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "project-depth-shadows", "load-teapot-build-floor",
            "build-shadow-map-resources", "render-light-depth-pass",
            "project-shadow-coordinates", "filter-shadow-with-pcf",
            "render-lit-scene", "animate-light", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "shadowmapping")
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: shadow mapping",
        )
        self.assertIn('"Live WebGPU rendering: shadow mapping"', document)
        self.assertIn(
            'role="region" aria-labelledby="shadow-assets-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="shadow-resources-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="shadow-passes-caption" tabindex="0"', document)
        self.assertIn("../bloom/#about", page.links)
        self.assertIn("../bloom/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/shadowmapping.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/shadowmapping.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_scene.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/teapot.gltf", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/4463644abffd90766934dba59daa1d71d379476a", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/f3801edef8ed726a011d0aa352fbc2428fccfe94", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/5a44b27f48f8aaa78515e9c787afcec847668ff7", page.links)
        self.assertIn("cargo run --example shadowmapping", document)
        self.assertIn(
            "scripts/build-wasm.sh --release shadowmapping", document)
        self.assertIn("225,816", document)
        self.assertIn("167,328", document)
        self.assertIn("4,690", document)
        self.assertIn("27,384", document)
        self.assertIn("9,128", document)
        self.assertIn("297,136", document)
        self.assertIn("297,320", document)
        self.assertIn("304-byte", document)
        self.assertIn("1024", document)
        self.assertIn("4,194,304", document)
        self.assertIn("8,388,608", document)
        self.assertIn("18,258", document)
        self.assertIn("54,774", document)
        self.assertIn("1.75", document)
        self.assertIn("0.00035", document)
        self.assertIn("textureSampleCompare", document)
        self.assertIn("Rgba8Unorm", document)
        self.assertIn("Depth32Float", document)
        self.assertIn("./shadowmapping.js?build=seo-test", document)
        self.assertIn("./shadowmapping_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/bloom", document)
        self.assertIn("../shadowmappingcascade/", page.links)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/shadowmapping.rs">Read the shadow mapping source &nearr;</a>\n'
            '      <a href="../bloom/">&larr; Previous: WebGPU bloom</a>\n'
            '      <a href="../shadowmappingcascade/">Next: WebGPU cascaded shadow mapping &rarr;</a>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 39, "name": "WebGPU shadow mapping", '
            '"url": "https://pooya.ai/webgpu/shadowmapping/"',
            gallery,
        )
        self.assertIn('<strong>WebGPU shadow mapping</strong>', gallery)
        self.assertIn(
            'alt="A white teapot casting a long dark shadow across a beige floor against a gray background"',
            gallery,
        )
        self.assertIn(
            "project nine comparison samples onto a procedural floor", gallery)

    def test_repository_pbr_basic_has_its_own_article_and_navigation(self):
        document = renderer.render_example("pbr", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "compare-metallic-and-roughness", "generate-sphere-mesh",
            "pack-material-instances", "bind-pbr-resources",
            "evaluate-ggx-specular-brdf", "render-instanced-grid",
            "animate-four-lights", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "pbr")
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: PBR Basic",
        )
        self.assertIn('"Live WebGPU rendering: PBR Basic"', document)
        self.assertIn(
            'role="region" aria-labelledby="pbr-geometry-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="pbr-resources-caption" tabindex="0"', document)
        self.assertIn(
            'role="region" aria-labelledby="pbr-passes-caption" tabindex="0"', document)
        self.assertIn("../shadowmappingomni/#about", page.links)
        self.assertIn("../shadowmappingomni/", page.links)
        self.assertIn("../pbrtexture/", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/pbr.rs", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/pbr.wgsl", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/657a1bca82bacf4017c814077c7ed8391e2d4893", page.links)
        self.assertIn(
            "https://github.com/PooyaEimandar/webgpu/commit/21636cd62fd68f378d72693382c2b9af1cbcf811", page.links)
        self.assertIn("cargo run --example pbr", document)
        self.assertIn("scripts/build-wasm.sh --release pbr", document)
        self.assertIn("2,337", document)
        self.assertIn("13,440", document)
        self.assertIn("4,480", document)
        self.assertIn("49", document)
        self.assertIn("111,624", document)
        self.assertIn("208-byte", document)
        self.assertIn("219,520", document)
        self.assertIn("658,560", document)
        self.assertIn("5,488", document)
        self.assertIn("Depth32Float", document)
        self.assertIn("NdotV", document)
        self.assertIn("HdotV", document)
        self.assertIn("./pbr.js?build=seo-test", document)
        self.assertIn("./pbr_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/shadowmapping", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/pbr.rs">Read the PBR Basic source &nearr;</a>\n'
            '      <a href="../shadowmappingomni/">&larr; Previous: WebGPU omnidirectional shadow mapping</a>\n'
            '      <a href="../pbrtexture/">Next: WebGPU PBR texture &rarr;</a>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 36, "name": "WebGPU PBR Basic", '
            '"url": "https://pooya.ai/webgpu/pbr/"',
            gallery,
        )
        self.assertIn("<strong>WebGPU PBR Basic</strong>", gallery)
        self.assertIn(
            'alt="A 7 by 7 grid of gold spheres showing metallic and roughness changes under four lights"',
            gallery,
        )
        self.assertIn("49 instanced gold spheres", gallery)

    def test_repository_pbr_texture_has_its_own_article_and_navigation(self):
        document = renderer.render_example("pbrtexture", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "move-pbr-properties-into-textures", "load-cerberus-and-eleven-images",
            "normalize-and-upload-cerberus", "upload-five-material-maps",
            "generate-image-based-lighting", "bind-thirteen-pbr-resources",
            "shade-direct-light-with-ggx", "combine-split-sum-ibl",
            "render-skybox-model-and-controls", "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "pbrtexture")
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: PBR textures",
        )
        for caption in (
            "pbrtexture-assets-caption", "pbrtexture-buffers-caption",
            "pbrtexture-textures-caption", "pbrtexture-bindings-caption",
            "pbrtexture-passes-caption",
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
        self.assertIn("../pbr/#about", page.links)
        self.assertIn("../pbr/", page.links)
        self.assertIn("../pbribl/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/pbrtexture.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/pbrtexture.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/cerberus.gltf",
            "https://github.com/PooyaEimandar/webgpu/tree/main/assets/textures/cerberus",
            "https://github.com/PooyaEimandar/webgpu/tree/main/assets/textures/skybox/bridge2",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/skybox.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/ktx.rs",
            "https://github.com/PooyaEimandar/webgpu/commit/657a1bca82bacf4017c814077c7ed8391e2d4893",
            "https://github.com/PooyaEimandar/webgpu/commit/21636cd62fd68f378d72693382c2b9af1cbcf811",
            "https://github.com/PooyaEimandar/webgpu/commit/c73b19eceb187417370e82ebeedb8446554bce38",
            "https://github.com/PooyaEimandar/webgpu/commit/7bb1e1b93fad1cf7683913cfa79394e50b137f98",
        ):
            self.assertIn(link, page.links)
        self.assertIn("cargo run --example pbrtexture", document)
        self.assertIn("scripts/build-wasm.sh --release pbrtexture", document)
        for fact in (
            "50,453,345", "2,372,173", "1,776,320", "32,814",
            "100,623", "33,541", "1,978,020", "83,886,080",
            "109,223,928", "9,437,184", "1,048,576", "Bind entries</dt><dd>13",
            "Depth32Float", "NdotV", "VdotH",
        ):
            self.assertIn(fact, document)
        self.assertIn("./pbrtexture.js?build=seo-test", document)
        self.assertIn("./pbrtexture_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/pbr.jpg", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/pbrtexture.rs">Read the PBR Texture source &nearr;</a>\n'
            '      <a href="../pbr/">&larr; Previous: WebGPU PBR Basic</a>\n'
            '      <a href="../pbribl/">Next: WebGPU PBR image-based lighting &rarr;</a>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 37, "name": "WebGPU PBR textures", '
            '"url": "https://pooya.ai/webgpu/pbrtexture/"',
            gallery,
        )
        self.assertIn("<strong>WebGPU PBR textures</strong>", gallery)
        self.assertIn(
            'alt="An ornate black, brass, and wood Cerberus triple-barrel pistol rendered against a sunlit bridge cubemap beside PBR controls"',
            gallery,
        )
        self.assertIn("five material maps", gallery)

    def test_repository_pbr_ibl_has_its_own_article_and_navigation(self):
        document = renderer.render_example("pbribl", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "add-environment-lighting-to-pbr", "load-bridge2-cubemap",
            "build-gold-sphere-line", "generate-environment-mips",
            "convolve-diffuse-irradiance", "integrate-split-sum-brdf",
            "combine-direct-and-image-lighting", "render-skybox-and-spheres",
            "tone-map-surface-output", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "pbribl")
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: PBR image-based lighting",
        )
        for caption in (
            "pbribl-assets-caption", "pbribl-geometry-caption",
            "pbribl-textures-caption", "pbribl-passes-caption",
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
        self.assertIn("../pbrtexture/#about", page.links)
        self.assertIn("../pbrtexture/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/pbribl.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/pbribl.wgsl",
            "https://github.com/PooyaEimandar/webgpu/tree/main/assets/textures/skybox/bridge2",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/skybox.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/ktx.rs",
            "https://github.com/PooyaEimandar/webgpu/commit/657a1bca82bacf4017c814077c7ed8391e2d4893",
            "https://github.com/PooyaEimandar/webgpu/commit/21636cd62fd68f378d72693382c2b9af1cbcf811",
            "https://github.com/PooyaEimandar/webgpu/commit/7bb1e1b93fad1cf7683913cfa79394e50b137f98",
        ):
            self.assertIn(link, page.links)
        self.assertIn("../parallaxmapping/", page.links)
        self.assertIn("cargo run --example pbribl", document)
        self.assertIn("scripts/build-wasm.sh --release pbribl", document)
        for fact in (
            "25,166,232", "25,165,824", "2,337", "13,440",
            "44,812", "134,436", "25,337,848", "55 initialization writes",
            "9,437,184", "1,048,576", "288-byte", "Depth32Float",
            "N dot V", "V dot H",
        ):
            self.assertIn(fact, document)
        self.assertIn(
            "box-filtered mip pyramid, not a GGX-prefiltered environment", document)
        self.assertIn("./pbribl.js?build=seo-test", document)
        self.assertIn("./pbribl_bg.wasm?build=seo-test", document)
        self.assertNotIn("screenshots/pbrtexture.jpg", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/pbribl.rs">Read the PBR image-based lighting source &nearr;</a>\n'
            '      <a href="../pbrtexture/">&larr; Previous: WebGPU PBR texture</a>\n'
            '      <a href="../parallaxmapping/">Next: WebGPU parallax occlusion mapping &rarr;</a>\n'
            '      <p class="copyright">',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 38, "name": "WebGPU PBR image-based lighting", '
            '"url": "https://pooya.ai/webgpu/pbribl/"',
            gallery,
        )
        self.assertIn(
            "<strong>WebGPU PBR image-based lighting</strong>", gallery)
        self.assertIn(
            'alt="Ten gold spheres with changing roughness and metallic reflections across a suspension bridge cubemap"',
            gallery,
        )
        self.assertIn("seven box-filtered cubemap mips", gallery)
        self.assertLess(gallery.index('href="./pbr/"'),
                        gallery.index('href="./pbrtexture/"'))
        self.assertLess(gallery.index('href="./pbrtexture/"'),
                        gallery.index('href="./pbribl/"'))

    def test_repository_parallax_mapping_has_its_own_article_and_navigation(self):
        document = renderer.render_example("parallaxmapping", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "simulate-depth-on-flat-plane", "load-three-runtime-assets",
            "decode-plane-gltf", "build-tangent-frame", "trace-height-layers",
            "shade-shifted-surface", "render-plane-and-overlay", "run-the-example",
        ])
        self.assertIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "parallaxmapping")
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: parallax occlusion mapping",
        )
        for caption in (
            "parallaxmapping-assets-caption", "parallaxmapping-buffers-caption",
            "parallaxmapping-modes-caption", "parallaxmapping-passes-caption",
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
        self.assertIn("../pbribl/#about", page.links)
        self.assertIn("../pbribl/", page.links)
        self.assertIn("../multisampling/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/parallaxmapping.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/parallaxmapping.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/plane.gltf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/rocks_color_rgba.ktx",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/rocks_normal_height_rgba.ktx",
            "https://github.com/PooyaEimandar/sib",
            "https://github.com/PooyaEimandar/webgpu/commit/d83b56f5dedf7bb510adec31b6be026e8b20cd85",
            "https://github.com/PooyaEimandar/webgpu/commit/c46fad85c97dc6995740d02cf54629dbe7bb5a44",
            "https://github.com/PooyaEimandar/webgpu/commit/b10569567075cab56dcd860da874050f3b0eab34",
            "https://github.com/PooyaEimandar/webgpu/commit/3201d1cd31dd20865d03c29c409a2aec075f1339",
        ):
            self.assertIn(link, page.links)
        self.assertIn("cargo run --example parallaxmapping", document)
        self.assertIn(
            "scripts/build-wasm.sh --release parallaxmapping", document)
        self.assertIn("cargo run --bin serve", document)
        for fact in (
            "Height layers</dt><dd>48", "Plane triangles</dt><dd>2",
            "Asset requests</dt><dd>3", "9,789,600", "9,786,708",
            "5,592,544", "4,194,404", "5,592,404", "4,194,304",
            "2,652", "204-byte", "216-byte GPU mesh", "11 mips",
            "392 bytes", "160-byte", "16-byte", "128 iterations",
            "two-second light orbit", "Depth32Float", "6 indices; 2 triangles",
        ):
            self.assertIn(fact, document)
        self.assertIn("./parallaxmapping.js?build=seo-test", document)
        self.assertIn("./parallaxmapping_bg.wasm?build=seo-test", document)
        self.assertIn("screenshots/parallaxmapping.jpg", document)
        self.assertNotIn("screenshots/pbribl.jpg", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/parallaxmapping.rs">Read the Parallax Mapping source &nearr;</a>\n'
            '      <a href="../pbribl/">&larr; Previous: WebGPU PBR image-based lighting</a>\n'
            '      <a href="../multisampling/">Next: WebGPU 4x MSAA multisampling &rarr;</a>\n'
            f'      <p class="copyright">&copy; <span data-current-year>{renderer.date.today().year}</span> <a href="https://pooya.ai">Pooya Eimandar</a>. All rights reserved.</p>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 33, "name": "WebGPU parallax occlusion mapping", '
            '"url": "https://pooya.ai/webgpu/parallaxmapping/"',
            gallery,
        )
        self.assertIn(
            "<strong>WebGPU parallax occlusion mapping</strong>", gallery)
        self.assertIn(
            'alt="A flat WebGPU plane appearing densely covered with raised rocks through parallax occlusion mapping"',
            gallery,
        )
        self.assertIn(
            "A two-triangle plane ray-marches 48 height layers from alpha, then lights RGB tangent-space normals to create parallax occlusion.",
            gallery,
        )

    def test_repository_multisampling_has_its_own_article_and_navigation(self):
        document = renderer.render_example("multisampling", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "smooth-voyager-edges", "load-one-voyager-asset",
            "flatten-three-material-draws", "upload-material-textures",
            "allocate-msaa-attachments", "shade-resolve-and-overlay",
            "control-the-camera", "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "multisampling")
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: 4x MSAA multisampling",
        )
        self.assertEqual(
            self.json_ld(document)["headline"],
            "WebGPU Multisampling in Rust: 4x MSAA with wgpu",
        )
        self.assertIn(
            '<p class="eyebrow">WebGPU notes &nbsp; / &nbsp; 28</p>',
            document,
        )
        self.assertIn(
            '<h1 id="article-title">WebGPU Multisampling in Rust: 4x MSAA with wgpu</h1>',
            document,
        )
        for caption, caption_text in (
            ("multisampling-assets-caption",
             "Multisampling runtime asset and embedded data"),
            ("multisampling-draws-caption",
             "Voyager geometry and material draw ranges"),
            ("multisampling-resources-caption",
             "Multisampling GPU resources and render attachments"),
            ("multisampling-passes-caption",
             "Multisampling render passes, attachments, and submitted work"),
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
            self.assertIn(
                f'<caption id="{caption}">{caption_text}</caption>', document)
        self.assertIn("../parallaxmapping/#about", page.links)
        self.assertIn("../parallaxmapping/", page.links)
        self.assertIn("../multisamplingalphatocoverage/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/multisampling.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/multisampling.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/voyager.gltf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/joystick.rs",
            "https://github.com/PooyaEimandar/sib",
            "https://github.com/PooyaEimandar/webgpu/commit/a4f0c4e5d8beb0f70420b89d1b111b52aa47fc36",
            "https://github.com/PooyaEimandar/webgpu/commit/550d035fd99724bb9da48c041c06feac5aee3c80",
            "https://github.com/PooyaEimandar/webgpu/commit/c46fad85c97dc6995740d02cf54629dbe7bb5a44",
            "https://github.com/PooyaEimandar/webgpu/commit/de16e06e35263457f2b71db9ab55b4f2376f21f8",
            "https://github.com/PooyaEimandar/webgpu/commit/88837b4e9f531bbe7b8b157803747708d169e88d",
            "https://github.com/PooyaEimandar/webgpu/commit/dabca148f6b919f3438a6159d4c808bcf5904d83",
        ):
            self.assertIn(link, page.links)
        self.assertIn("cargo run --example multisampling", document)
        self.assertIn(
            "scripts/build-wasm.sh --release multisampling", document)
        self.assertIn("cargo run --bin serve", document)
        for fact in (
            "Rasterization samples</dt><dd>4&times;",
            "Voyager triangles</dt><dd>20,378",
            "Runtime scene assets</dt><dd>1",
            "3,203,450", "2,396,228", "23,914", "61,134",
            "44-byte stride", "1,052,216", "244,536", "144-byte uniform",
            "12,582,916", "13,879,812", "14,745,600", "29,491,200",
            "Three indexed material draws", "20,378 triangles",
            "Four-sample color", "Four-sample <code>Depth32Float</code>",
        ):
            self.assertIn(fact, document)
        for limitation in (
            "The sample count is fixed at four with no runtime selector or capability-based fallback.",
            "There is no built-in lower-sample comparison, explicit per-sample shader evaluation, alpha-to-coverage, or depth resolve.",
            "MSAA smooths polygon coverage boundaries; it does not fix shader aliasing or the material textures' missing mipmaps.",
            "The depth samples are also discarded because this example has no depth resolve and no later pass reads them.",
            "Those framework pipelines are single-sample and operate after the resolve, so the overlay itself is not part of the four-sample scene.",
        ):
            self.assertIn(limitation, document)
        self.assertIn("./multisampling.js?build=seo-test", document)
        self.assertIn("./multisampling_bg.wasm?build=seo-test", document)
        self.assertIn("screenshots/multisampling.jpg", document)
        self.assertNotIn("screenshots/parallaxmapping.jpg", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/multisampling.rs">Read the Multisampling source &nearr;</a>\n'
            '      <a href="../parallaxmapping/">&larr; Previous: WebGPU parallax occlusion mapping</a>\n'
            '      <a href="../multisamplingalphatocoverage/">Next: WebGPU alpha-to-coverage &rarr;</a>\n'
            f'      <p class="copyright">&copy; <span data-current-year>{renderer.date.today().year}</span> <a href="https://pooya.ai">Pooya Eimandar</a>. All rights reserved.</p>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 34, "name": "WebGPU 4x MSAA multisampling", '
            '"url": "https://pooya.ai/webgpu/multisampling/"',
            gallery,
        )
        self.assertIn("<strong>WebGPU 4x MSAA multisampling</strong>", gallery)
        self.assertIn(
            'alt="A white and black Voyager space probe with a dish antenna and long lattice boom on a white WebGPU canvas"',
            gallery,
        )
        self.assertIn(
            "Render the textured Voyager in three indexed draws into 4x multisampled color and depth, then resolve once into the surface.",
            gallery,
        )
        self.assertLess(
            gallery.index('href="./parallaxmapping/"'),
            gallery.index('href="./multisampling/"'),
        )
        self.assertLess(
            gallery.index('href="./multisampling/"'),
            gallery.index('href="./multisamplingalphatocoverage/"'),
        )

    def test_repository_multisampling_alpha_to_coverage_has_article_and_navigation(self):
        document = renderer.render_example(
            "multisamplingalphatocoverage", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "turn-alpha-into-coverage", "load-oak-asset",
            "flatten-tree-primitives", "instance-oak-grove",
            "upload-material-textures", "allocate-msaa-targets",
            "convert-alpha-to-samples", "render-resolve-overlay",
            "control-camera", "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"],
            "multisamplingalphatocoverage",
        )
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: alpha-to-coverage foliage",
        )
        self.assertEqual(
            self.json_ld(document)["headline"],
            "WebGPU Alpha-to-Coverage in Rust: 4x MSAA Foliage",
        )
        self.assertEqual(
            page.canonical,
            "https://pooya.ai/webgpu/multisamplingalphatocoverage/",
        )
        self.assertEqual(
            page.metadata["description"],
            "Learn WebGPU alpha-to-coverage with Rust and wgpu by rendering 25 instanced oak trees with 4x MSAA, alpha-textured leaves, depth testing, and hardware resolve.",
        )
        self.assertIn(
            '<p class="eyebrow">WebGPU notes &nbsp; / &nbsp; 29</p>',
            document,
        )
        self.assertIn(
            '<h1 id="article-title">WebGPU Alpha-to-Coverage in Rust: 4x MSAA Foliage</h1>',
            document,
        )
        for caption, caption_text in (
            ("alpha-coverage-assets-caption",
             "Alpha-to-coverage runtime asset and compiled inputs"),
            ("alpha-coverage-geometry-caption",
             "Oak geometry, materials, and instanced draw work"),
            ("alpha-coverage-resources-caption",
             "Alpha-to-coverage explicit GPU resources"),
            ("alpha-coverage-pipeline-caption",
             "How sampled alpha affects the four-sample pipeline"),
            ("alpha-coverage-passes-caption",
             "Alpha-to-coverage passes and submitted scene work"),
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
            self.assertIn(
                f'<caption id="{caption}">{caption_text}</caption>', document)
        self.assertIn("../multisampling/#about", page.links)
        self.assertIn("../multisampling/", page.links)
        self.assertIn("../deferred/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/multisamplingalphatocoverage.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/multisamplingalphatocoverage.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/oaktree.gltf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/joystick.rs",
            "https://github.com/PooyaEimandar/sib",
            "https://github.com/PooyaEimandar/webgpu/commit/a4f0c4e5d8beb0f70420b89d1b111b52aa47fc36",
            "https://github.com/PooyaEimandar/webgpu/commit/550d035fd99724bb9da48c041c06feac5aee3c80",
            "https://github.com/PooyaEimandar/webgpu/commit/c46fad85c97dc6995740d02cf54629dbe7bb5a44",
            "https://github.com/PooyaEimandar/webgpu/commit/de16e06e35263457f2b71db9ab55b4f2376f21f8",
            "https://github.com/PooyaEimandar/webgpu/commit/88837b4e9f531bbe7b8b157803747708d169e88d",
        ):
            self.assertIn(link, page.links)
        self.assertIn(
            "cargo run --example multisamplingalphatocoverage", document)
        self.assertIn(
            "scripts/build-wasm.sh --release multisamplingalphatocoverage",
            document,
        )
        self.assertIn("cargo run --bin serve", document)
        for fact in (
            "Rasterization samples</dt><dd>4&times;",
            "Tree instances</dt><dd>25",
            "Submitted triangles</dt><dd>132,550",
            "1,706,820", "1,275,656", "207,780",
            "5,499", "15,906", "5,302", "397,650",
            "44-byte vertex stream", "241,956", "63,624", "300",
            "2,097,156", "2,403,180", "29,491,200", "30.417 MiB",
            "210,634", "48,915", "2,595",
            "alpha_to_coverage_enabled: <span class=\"code-keyword\">true</span>",
            "2 indexed instanced draws", "132,550 triangles",
        ):
            self.assertIn(fact, document)
        for behavior in (
            "Transparent areas of the rectangular leaf cards preserve the background and depth already stored in uncovered samples",
            "The leaf material declares glTF <code>alphaMode: BLEND</code>, but this focused loader does not implement glTF blending.",
            "Alpha-to-coverage is a fixed-function step after the fragment shader produces target-zero alpha.",
            "Both multisampled attachments use discard stores because no later pass reads them.",
            "The second pass loads the resolved surface and draws the text and any active joystick rings without depth or another resolve.",
        ):
            self.assertIn(behavior, document)
        for limitation in (
            "The exact threshold and sample pattern are implementation dependent",
            "The sample count is fixed at four, with no runtime capability branch, selector, 1x comparison, or lower-sample fallback.",
            "There is no conventional alpha blend, sorting, stochastic mask, temporal accumulation, foliage-specific alpha cutoff, or depth resolve.",
            "The missing mip chain can still make distant leaf cards shimmer even when polygon and alpha edges use MSAA.",
            "the effective light therefore follows camera translation instead of behaving as the configured world-space point.",
        ):
            self.assertIn(limitation, document)
        self.assertIn(
            "./multisamplingalphatocoverage.js?build=seo-test",
            document,
        )
        self.assertIn(
            "./multisamplingalphatocoverage_bg.wasm?build=seo-test",
            document,
        )
        self.assertIn("screenshots/multisamplingalphatocoverage.jpg", document)
        self.assertNotIn("screenshots/multisampling.jpg", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/multisamplingalphatocoverage.rs">Read the Alpha-to-coverage source &nearr;</a>\n'
            '      <a href="../multisampling/">&larr; Previous: WebGPU 4x MSAA multisampling</a>\n'
            '      <a href="../deferred/">Next: WebGPU deferred shading &rarr;</a>\n'
            f'      <p class="copyright">&copy; <span data-current-year>{renderer.date.today().year}</span> <a href="https://pooya.ai">Pooya Eimandar</a>. All rights reserved.</p>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 35, "name": "WebGPU alpha-to-coverage foliage", '
            '"url": "https://pooya.ai/webgpu/multisamplingalphatocoverage/"',
            gallery,
        )
        self.assertIn(
            "<strong>WebGPU alpha-to-coverage foliage</strong>", gallery)
        self.assertIn(
            'alt="Dense grove of alpha-textured oak trees rendered with 4x WebGPU MSAA and alpha-to-coverage against a dark blue background."',
            gallery,
        )
        self.assertIn(
            "Render 25 instanced oak trees in two indexed draws, converting leaf alpha into four-sample coverage before resolving to the surface.",
            gallery,
        )
        self.assertLess(
            gallery.index('href="./multisampling/"'),
            gallery.index('href="./multisamplingalphatocoverage/"'),
        )

    def test_repository_deferred_has_its_own_article_and_navigation(self):
        document = renderer.render_example("deferred", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "defer-lighting", "load-jax-assets", "animate-jax-skin",
            "fill-gbuffer", "budget-gbuffer", "compose-six-lights",
            "record-two-passes", "control-camera", "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"], "deferred")
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: deferred shading",
        )
        self.assertEqual(
            self.json_ld(document)["headline"],
            "WebGPU Deferred Shading in Rust: G-Buffer with wgpu",
        )
        self.assertEqual(page.canonical, "https://pooya.ai/webgpu/deferred/")
        self.assertEqual(
            page.metadata["description"],
            "Build deferred shading in WebGPU with Rust and wgpu: fill position, normal, and albedo G-buffer targets, then light animated Jax with six point lights.",
        )
        self.assertIn(
            '<p class="eyebrow">WebGPU notes &nbsp; / &nbsp; 30</p>',
            document,
        )
        self.assertIn(
            '<h1 id="article-title">WebGPU Deferred Shading in Rust: G-Buffer with wgpu</h1>',
            document,
        )
        for caption, caption_text in (
            ("deferred-assets-caption",
             "Deferred shading runtime assets and compiled inputs"),
            ("deferred-geometry-caption",
             "Deferred shading geometry and explicit GPU buffers"),
            ("deferred-gbuffer-caption",
             "Full-resolution Deferred G-buffer attachments"),
            ("deferred-lights-caption",
             "Six lights evaluated by every deferred composition fragment"),
            ("deferred-passes-caption",
             "Deferred shading render passes and explicit draw work"),
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
            self.assertIn(
                f'<caption id="{caption}">{caption_text}</caption>', document)
        self.assertIn("../multisamplingalphatocoverage/#about", page.links)
        self.assertIn("../multisamplingalphatocoverage/", page.links)
        self.assertIn("../deferredmultisampling/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/deferred.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/deferred.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_skin.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/joystick.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.gltf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.bin",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/jax_base_color.png",
            "https://github.com/PooyaEimandar/sib",
            "https://github.com/PooyaEimandar/webgpu/commit/5f1bc5ac180630a7bc3b5da71fb57d15b77193ff",
            "https://github.com/PooyaEimandar/webgpu/commit/a65ec584ad446059707f4c0c9d1f14171540aa7a",
            "https://github.com/PooyaEimandar/webgpu/commit/550d035fd99724bb9da48c041c06feac5aee3c80",
            "https://github.com/PooyaEimandar/webgpu/commit/c46fad85c97dc6995740d02cf54629dbe7bb5a44",
            "https://github.com/PooyaEimandar/webgpu/commit/b5c162571514843771ceb98d7e50861aaedfff68",
            "https://github.com/PooyaEimandar/webgpu/commit/34b0e557b82aefaf30a5ab426386a5a613f2acd7",
            "https://github.com/PooyaEimandar/webgpu/commit/27ff1c98d5f2a353a18376377ee70d47d9b14975",
        ):
            self.assertIn(link, page.links)
        self.assertIn("cargo run --example deferred", document)
        self.assertIn("scripts/build-wasm.sh --release deferred", document)
        self.assertIn("cargo run --bin serve", document)
        for fact in (
            "G-buffer color targets</dt><dd>3",
            "Deferred lights</dt><dd>6",
            "Geometry-pass triangles</dt><dd>11,962",
            "68,324", "1,957,752", "13,008", "2,039,084",
            "35,880", "76 bytes", "2,726,880", "143,520",
            "11,960 triangles", "8,192", "46 used", "2,879,368",
            "4,194,304", "7,073,672", "Rgba16Float", "Rgba8Unorm",
            "Depth32Float", "24 logical bytes per physical pixel",
            "22,118,400", "21.09375 MiB", "18,432,000", "3,686,400",
            "2 + 11,960 = 11,962", "11,963 submitted triangles",
            "five-second cycle", "radius / (distance&sup2; + 1)",
        ):
            self.assertIn(fact, document)
        for limitation in (
            "Rust always uploads zero and exposes no UI for changing it, so the live demo only shows composed lighting.",
            "The second pass clears the surface dark blue, but the fullscreen triangle covers it and returns black for empty G-buffer pixels.",
            "Transparent geometry is not supported: the G-buffer pipelines do not blend, Jax writes a fixed 0.45 specular value instead of material or texture alpha, and composition always outputs alpha one.",
            "The material base-color factor is baked into loader vertex color and multiplied again in the deferred character shader; Jax's white factor hides that double application.",
            "Specular is not gated by a positive <code>N dot L</code>.",
        ):
            self.assertIn(limitation, document)
        self.assertIn("./deferred.js?build=seo-test", document)
        self.assertIn("./deferred_bg.wasm?build=seo-test", document)
        self.assertIn("screenshots/deferred.jpg", document)
        self.assertNotIn("screenshots/multisampling.jpg", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/deferred.rs">Read the Deferred shading source &nearr;</a>\n'
            '      <a href="../multisamplingalphatocoverage/">&larr; Previous: WebGPU alpha-to-coverage</a>\n'
            '      <a href="../deferredmultisampling/">Next: WebGPU deferred multisampling &rarr;</a>\n'
            f'      <p class="copyright">&copy; <span data-current-year>{renderer.date.today().year}</span> <a href="https://pooya.ai">Pooya Eimandar</a>. All rights reserved.</p>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 29, "name": "WebGPU deferred shading", '
            '"url": "https://pooya.ai/webgpu/deferred/"',
            gallery,
        )
        self.assertIn("<strong>WebGPU deferred shading</strong>", gallery)
        self.assertIn(
            'alt="Animated rabbit-like Jax character walking across a checkerboard floor under red, yellow, green, blue, and white deferred lights"',
            gallery,
        )
        self.assertIn(
            "Write world position, normal, and albedo into three G-buffer targets, then light animated Jax and the floor with six lights in one fullscreen pass.",
            gallery,
        )
        self.assertLess(gallery.index('href="./bloom/"'),
                        gallery.index('href="./deferred/"'))
        self.assertLess(
            gallery.index('href="./deferred/"'),
            gallery.index('href="./deferredmultisampling/"'),
        )

    def test_repository_deferred_multisampling_has_its_own_article_and_navigation(self):
        document = renderer.render_example("deferredmultisampling", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "resolve-after-lighting", "load-jax-assets", "animate-scene",
            "store-four-samples", "resolve-four-samples", "record-two-passes",
            "control-camera", "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"],
            "deferredmultisampling",
        )
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: deferred multisampling",
        )
        self.assertEqual(
            self.json_ld(document)["headline"],
            "WebGPU Deferred MSAA in Rust: Manual 4x Resolve with wgpu",
        )
        self.assertEqual(
            page.canonical,
            "https://pooya.ai/webgpu/deferredmultisampling/",
        )
        self.assertEqual(
            page.metadata["description"],
            "Build 4x MSAA deferred shading in WebGPU with Rust and wgpu: store four G-buffer samples, shade each with six lights, and manually average the result.",
        )
        self.assertIn(
            '<p class="eyebrow">WebGPU notes &nbsp; / &nbsp; 31</p>',
            document,
        )
        self.assertIn(
            '<h1 id="article-title">WebGPU Deferred MSAA in Rust: Manual 4x Resolve with wgpu</h1>',
            document,
        )
        for caption, caption_text in (
            ("deferred-msaa-assets-caption",
             "Deferred multisampling runtime assets and compiled inputs"),
            ("deferred-msaa-geometry-caption",
             "Deferred multisampling scene geometry and explicit GPU buffers"),
            ("deferred-msaa-gbuffer-caption",
             "Four-sample Deferred MSAA G-buffer attachments at 1280 by 720"),
            ("deferred-msaa-resolve-caption",
             "Manual Deferred MSAA resolve work in the normal composition path"),
            ("deferred-msaa-passes-caption",
             "Deferred multisampling render passes and submitted work"),
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
            self.assertIn(
                f'<caption id="{caption}">{caption_text}</caption>', document)
        self.assertIn("../deferred/#about", page.links)
        self.assertIn("../multisampling/#about", page.links)
        self.assertIn("../deferred/", page.links)
        self.assertIn("../deferredshadows/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/deferredmultisampling.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/deferredmultisampling.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_skin.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/joystick.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.gltf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.bin",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/jax_base_color.png",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/deferredmultisampling.jpg",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/deferredmultisampling.webp",
            "https://github.com/PooyaEimandar/sib",
            "https://github.com/PooyaEimandar/webgpu/commit/836a56bad389e0d9c4e6bead9c07545f288fee55",
            "https://github.com/PooyaEimandar/webgpu/commit/24b478ba1c91b8957c25aa6ddac7b87a29c681e5",
            "https://github.com/PooyaEimandar/webgpu/commit/550d035fd99724bb9da48c041c06feac5aee3c80",
            "https://github.com/PooyaEimandar/webgpu/commit/c46fad85c97dc6995740d02cf54629dbe7bb5a44",
            "https://github.com/PooyaEimandar/webgpu/commit/a1783f9ce64b2e6e59c8eb1f45f0423117ee8365",
            "https://github.com/PooyaEimandar/webgpu/commit/a65ec584ad446059707f4c0c9d1f14171540aa7a",
            "https://github.com/PooyaEimandar/webgpu/commit/34b0e557b82aefaf30a5ab426386a5a613f2acd7",
            "https://github.com/PooyaEimandar/webgpu/commit/27ff1c98d5f2a353a18376377ee70d47d9b14975",
        ):
            self.assertIn(link, page.links)
        self.assertIn("cargo run --example deferredmultisampling", document)
        self.assertIn(
            "scripts/build-wasm.sh --release deferredmultisampling",
            document,
        )
        self.assertIn("cargo run --bin serve", document)
        for fact in (
            "G-buffer samples</dt><dd>4&times;",
            "Light evaluations per pixel</dt><dd>24",
            "G-buffer bytes per pixel</dt><dd>96",
            "68,324", "1,957,752", "13,008", "2,039,084",
            "35,880 positions", "35,880 source <code>u16</code> indices",
            "46-joint", "Walking_1", "138 channels", "91 STEP", "47 LINEAR",
            "2,726,880", "143,520", "2,879,368", "4,194,304",
            "7,073,672", "8,784 bytes", "128 matrices; 46 used",
            "144-byte block", "224-byte block", "96-byte four-sample G-buffer",
            "Rgba16Float", "Rgba8Unorm", "Depth32Float", "LessEqual",
            "29,491,200", "14,745,600", "88,473,600", "84.375 MiB",
            "70.3125 MiB", "14.0625 MiB", "95,547,272", "91.121 MiB",
            "texture_multisampled_2d&lt;f32&gt;", "textureLoad",
            "First-sample debug prefetch", "Ambient albedo resolve",
            "Per-sample lighting", "<strong>19</strong>", "<strong>116</strong>",
            "106,905,600", "22,118,400", "24 light evaluations",
            "2 + 11,960 = 11,962", "11,963 submitted triangles",
            "two passes, three explicit draws", "MSAA samples: 4x",
        ):
            self.assertIn(fact, document)
        for behavior in (
            "Here all three G-buffer colors have <code>resolve_target: None</code>; the shader reads their stored samples explicitly.",
            "The geometry shaders do not use <code>sample_index</code>, sample-qualified interpolation, or a sample mask output, so the code does not request four geometry fragment invocations per output pixel.",
            "The 116-byte figure follows source-level texel values: 8-byte position, 8-byte normal, and 4-byte albedo records. It is not measured physical bandwidth; a compiler, cache, or texture implementation may remove or hide repeated reads.",
            "Its ambient term is 15%, versus 2.5% in the single-sample shader, and its specular exponent is 8 instead of 16.",
            "The composition pipeline is single-sample, has no depth attachment, and returns alpha one.",
            "A resize recreates all four attachments and the composition bind group.",
        ):
            self.assertIn(behavior, document)
        for limitation in (
            "The sample count is hard-coded independently as four in Rust and WGSL.",
            "there is no runtime capability query, selector, 1x comparison, or fallback.",
            "It has no tiled or clustered light culling, hardware color resolve, depth resolve, or later use for the depth texture.",
            "MSAA improves polygon coverage boundaries; it does not generate texture mipmaps or solve shader and texture minification aliasing.",
            "Diagnostic targets show one sample rather than resolved data, and the single-sample overlay remains outside the antialiased scene.",
            "The loader bakes base-color factor into vertex color and the shader multiplies that factor again; Jax's white factor hides the duplicate application.",
            "Normals are not transformed by an inverse-transpose matrix, and specular is not gated by a positive <code>N dot L</code>.",
        ):
            self.assertIn(limitation, document)
        self.assertIn("./deferredmultisampling.js?build=seo-test", document)
        self.assertIn(
            "./deferredmultisampling_bg.wasm?build=seo-test", document)
        self.assertIn("screenshots/deferredmultisampling.jpg", document)
        self.assertNotIn("screenshots/deferred.jpg", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/deferredmultisampling.rs">Read the Deferred multisampling source &nearr;</a>\n'
            '      <a href="../deferred/">&larr; Previous: WebGPU deferred shading</a>\n'
            '      <a href="../deferredshadows/">Next: WebGPU deferred shadows &rarr;</a>\n'
            f'      <p class="copyright">&copy; <span data-current-year>{renderer.date.today().year}</span> <a href="https://pooya.ai">Pooya Eimandar</a>. All rights reserved.</p>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 30, "name": "WebGPU deferred multisampling", '
            '"url": "https://pooya.ai/webgpu/deferredmultisampling/"',
            gallery,
        )
        self.assertIn(
            "<strong>WebGPU deferred multisampling</strong>", gallery)
        self.assertIn(
            'alt="Animated Jax character walking across a checkerboard floor under colored deferred lights rendered with 4x MSAA"',
            gallery,
        )
        self.assertIn(
            "Store four samples for every position, normal, albedo, and depth G-buffer pixel, then shade and average them manually in one fullscreen pass.",
            gallery,
        )
        self.assertLess(
            gallery.index('href="./deferred/"'),
            gallery.index('href="./deferredmultisampling/"'),
        )
        self.assertLess(
            gallery.index('href="./deferredmultisampling/"'),
            gallery.index('href="./deferredshadows/"'),
        )

    def test_repository_deferred_shadows_has_its_own_article_and_navigation(self):
        document = renderer.render_example("deferredshadows", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "add-shadows", "load-jax-assets", "animate-shadow-caster",
            "render-shadow-maps", "fill-gbuffer", "filter-shadows",
            "record-five-passes", "control-camera", "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"],
            "deferredshadows",
        )
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: deferred shadow mapping",
        )
        self.assertEqual(
            self.json_ld(document)["headline"],
            "WebGPU Deferred Shadow Mapping in Rust: 3x3 PCF with wgpu",
        )
        self.assertEqual(
            page.canonical,
            "https://pooya.ai/webgpu/deferredshadows/",
        )
        self.assertEqual(
            page.metadata["description"],
            "Build deferred shadow mapping in WebGPU with Rust and wgpu: render animated Jax into three 1024x1024 depth maps and filter three spotlights with 3x3 PCF.",
        )
        self.assertIn(
            '<p class="eyebrow">WebGPU notes &nbsp; / &nbsp; 32</p>',
            document,
        )
        self.assertIn(
            '<h1 id="article-title">WebGPU Deferred Shadow Mapping in Rust: 3x3 PCF with wgpu</h1>',
            document,
        )
        for caption, caption_text in (
            ("deferred-shadows-assets-caption",
             "Deferred Shadows runtime assets and compiled inputs"),
            ("deferred-shadows-buffers-caption",
             "Deferred Shadows scene geometry and explicit GPU buffers"),
            ("deferred-shadows-maps-caption",
             "Deferred shadow-map resources and filtering rules"),
            ("deferred-shadows-gbuffer-caption",
             "Single-sample Deferred Shadows G-buffer at 1280 by 720"),
            ("deferred-shadows-lights-caption",
             "Eight moving spotlights evaluated by Deferred Shadows"),
            ("deferred-shadows-passes-caption",
             "Deferred Shadows render passes and submitted work"),
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
            self.assertIn(
                f'<caption id="{caption}">{caption_text}</caption>', document)
        self.assertIn("../deferredmultisampling/#about", page.links)
        self.assertIn("../deferredmultisampling/", page.links)
        self.assertIn("../ssao/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/deferredshadows.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/deferredshadows.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/web/index.html",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_skin.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/joystick.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.gltf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.bin",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/jax_base_color.png",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/deferredshadows.jpg",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/deferredshadows.webp",
            "https://github.com/PooyaEimandar/sib",
            "https://github.com/PooyaEimandar/webgpu/commit/24b478ba1c91b8957c25aa6ddac7b87a29c681e5",
            "https://github.com/PooyaEimandar/webgpu/commit/d097b099d109c48f0dc7028d837f4d7e2bd65a43",
            "https://github.com/PooyaEimandar/webgpu/commit/550d035fd99724bb9da48c041c06feac5aee3c80",
            "https://github.com/PooyaEimandar/webgpu/commit/c46fad85c97dc6995740d02cf54629dbe7bb5a44",
            "https://github.com/PooyaEimandar/webgpu/commit/a1783f9ce64b2e6e59c8eb1f45f0423117ee8365",
            "https://github.com/PooyaEimandar/webgpu/commit/a65ec584ad446059707f4c0c9d1f14171540aa7a",
            "https://github.com/PooyaEimandar/webgpu/commit/34b0e557b82aefaf30a5ab426386a5a613f2acd7",
            "https://github.com/PooyaEimandar/webgpu/commit/27ff1c98d5f2a353a18376377ee70d47d9b14975",
        ):
            self.assertIn(link, page.links)
        self.assertIn("cargo run --example deferredshadows", document)
        self.assertIn(
            "scripts/build-wasm.sh --release deferredshadows", document)
        self.assertIn("cargo run --bin serve", document)
        for fact in (
            "Shadow maps</dt><dd>3 &times; 1024&sup2;",
            "Spotlights</dt><dd>8",
            "PCF taps per shadowed light</dt><dd>9",
            "68,324", "1,957,752", "13,008", "2,039,084",
            "12,553", "350 lines of WGSL", "122,752",
            "35,880 vertices", "35,880 source <code>u16</code> indices",
            "Walking_1", "138 channels", "46 joints", "128 reserved matrices",
            "2,726,880", "143,520", "184", "8,192",
            "144 + 224 bytes", "432", "928-byte block",
            "2,880,504", "7,074,808", "9,920 bytes per frame",
            "12,582,912", "4,194,304", "16 MiB",
            "Depth32Float", "Rgba8Unorm", "Rgba16Float",
            "100&deg;; aspect 1; 0.1&ndash;64", "Slope 0.25",
            "max(0.00022(1-N&middot;L), 0.00008)",
            "3&times;3 = 9 taps", "Offsets 1.5 shadow texels apart",
            "<strong>24</strong>", "<strong>22,118,400</strong>",
            "7,372,800", "3,686,400", "21.09375 MiB",
            "45,970,424", "43.841 MiB", "3.5% ambient",
            "15&deg; inner cone", "28&deg; outer cone", "exponent 18",
            "18,432,000", "24,883,200", "47,843",
            "3 &times; 11,960 = 35,880", "2 + 11,960 = 11,962",
            "5 render passes", "6 explicit scene draws",
            "shadow maps: 3 x 1024",
        ):
            self.assertIn(fact, document)
        for behavior in (
            "Each shadowed spotlight gets a separate 1024&times;1024 <code>Depth32Float</code> texture with one sample, one mip, and one layer.",
            "The floor is a receiver but is not submitted as a shadow caster.",
            "Because the current shadow pipeline includes <code>fs_shadow</code>, it also binds one shared 1024&times;1024 <code>Rgba8Unorm</code> color target.",
            "The comparison sampler clamps at texture edges, uses nearest filtering, and applies <code>LessEqual</code>; softness comes from nine explicit comparison samples rather than sampler filtering.",
            "Each nine-tap result clamps to at least 0.2, so a shadow suppresses at most 80% of that light rather than becoming fully black.",
            "These are source-level operation counts rather than measured memory bandwidth or GPU invocation totals; compiler optimization and texture caches can change the physical cost.",
            "This is not a controlled shadows-only comparison with the earlier Deferred pages.",
            "Resizing recreates only the full-resolution G-buffer and its composition bind group; the three fixed-resolution shadow maps remain allocated.",
        ):
            self.assertIn(behavior, document)
        for limitation in (
            "they do not return to their starting values before the forced wrap and jump at the cycle boundary.",
            "Rust always uploads debug target zero and shadows enabled, and the UI exposes neither setting.",
            "Setting the internal <code>RENDER_SHADOW_MAPS</code> constant false is not a complete toggle because composition still enables shadows and samples the unwritten maps.",
            "those lookups still run inside a valid light frustum when the spotlight cone, <code>N dot L</code>, or albedo would make the contribution zero",
            "The shared throwaway color target and zero-output fragment stage are unnecessary for opaque depth-only shadow rendering.",
            "There is no transparency or alpha-tested shadow casting, although the current RGB texture has no alpha channel.",
            "Its white base-color factor hides a loader/shader double multiplication that would affect non-white materials.",
            "Normals are not inverse-transpose transformed, and specular is not gated by positive <code>N dot L</code>.",
            "the retained animated scene keeps CPU mesh and cloned image data after GPU upload.",
        ):
            self.assertIn(limitation, document)
        self.assertIn("./deferredshadows.js?build=seo-test", document)
        self.assertIn("./deferredshadows_bg.wasm?build=seo-test", document)
        self.assertIn("screenshots/deferredshadows.jpg", document)
        self.assertNotIn("screenshots/deferredmultisampling.jpg", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/deferredshadows.rs">Read the Deferred Shadows source &nearr;</a>\n'
            '      <a href="../deferredmultisampling/">&larr; Previous: WebGPU deferred multisampling</a>\n'
            '      <a href="../ssao/">Next: WebGPU screen-space ambient occlusion &rarr;</a>\n'
            f'      <p class="copyright">&copy; <span data-current-year>{renderer.date.today().year}</span> <a href="https://pooya.ai">Pooya Eimandar</a>. All rights reserved.</p>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 31, "name": "WebGPU deferred shadows", '
            '"url": "https://pooya.ai/webgpu/deferredshadows/"',
            gallery,
        )
        self.assertIn("<strong>WebGPU deferred shadows</strong>", gallery)
        self.assertIn(
            'alt="Animated Jax character walking across a checkerboard floor under colorful WebGPU spotlights casting multiple shadows"',
            gallery,
        )
        self.assertIn(
            "Render animated Jax into three 1024 &times; 1024 depth maps, then use up to 27 comparison samples while shading eight spotlights in a fullscreen deferred pass.",
            gallery,
        )
        self.assertLess(
            gallery.index('href="./deferredmultisampling/"'),
            gallery.index('href="./deferredshadows/"'),
        )
        self.assertLess(
            gallery.index('href="./deferredshadows/"'),
            gallery.index('href="./ssao/"'),
        )

    def test_repository_ssao_has_its_own_article_and_navigation(self):
        document = renderer.render_example("ssao", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "add-ssao", "load-jax-assets", "preserve-gbuffer", "sample-spiral",
            "blur-mask", "compose-lighting", "tune-live", "record-eight-passes",
            "control-camera", "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(page.elements_by_id["demo"]["data-example"], "ssao")
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: screen-space ambient occlusion",
        )
        self.assertEqual(
            self.json_ld(document)["headline"],
            "WebGPU SSAO in Rust: 32-Sample Ambient Occlusion with wgpu",
        )
        self.assertEqual(page.canonical, "https://pooya.ai/webgpu/ssao/")
        self.assertEqual(
            page.metadata["description"],
            "Build screen-space ambient occlusion in WebGPU with Rust and wgpu: sample 32 spiral neighbors, blur a full-resolution AO mask, and tune it live with egui.",
        )
        self.assertEqual(
            page.metadata["og:image"],
            "https://pooya.ai/webgpu/screenshots/ssao.jpg",
        )
        self.assertEqual(
            page.metadata["og:image:alt"],
            "WebGPU SSAO demo showing animated Jax on a checkerboard floor under colored spotlights, with ambient occlusion controls open.",
        )
        self.assertEqual(page.metadata["og:image:width"], "1280")
        self.assertEqual(page.metadata["og:image:height"], "720")
        self.assertIn(
            '<p class="eyebrow">WebGPU notes &nbsp; / &nbsp; 33</p>',
            document,
        )
        self.assertIn(
            '<h1 id="article-title">WebGPU SSAO in Rust: 32-Sample Ambient Occlusion with wgpu</h1>',
            document,
        )
        for caption, caption_text in (
            ("ssao-assets-caption", "SSAO runtime assets and compiled inputs"),
            ("ssao-targets-caption",
             "Full-resolution G-buffer and SSAO targets at 1280 by 720"),
            ("ssao-sampling-caption",
             "Normal-path SSAO and blur source work per output pixel"),
            ("ssao-controls-caption", "Live SSAO controls, defaults, and shader effects"),
            ("ssao-passes-caption",
             "SSAO frame passes and explicit scene or fullscreen draws"),
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
            self.assertIn(
                f'<caption id="{caption}">{caption_text}</caption>', document)
        self.assertEqual(document.count('<div class="article-table"'), 5)
        self.assertIn("../deferredshadows/#about", page.links)
        self.assertIn("../deferredshadows/", page.links)
        self.assertIn("../computeparticles/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/ssao.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/ssao.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/web/index.html",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/gltf_skin.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/joystick.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.gltf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/models/jax.bin",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/jax_base_color.png",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/fonts/Vazirmatn-Regular.ttf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/fonts/Vazirmatn-LICENSE.txt",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/ssao.jpg",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/ssao.webp",
            "https://github.com/PooyaEimandar/sib",
            "https://github.com/PooyaEimandar/webgpu/commit/fd8bf279df47abbdbd4579f2945aeb23a2a58709",
            "https://github.com/PooyaEimandar/webgpu/commit/550d035fd99724bb9da48c041c06feac5aee3c80",
            "https://github.com/PooyaEimandar/webgpu/commit/c46fad85c97dc6995740d02cf54629dbe7bb5a44",
            "https://github.com/PooyaEimandar/webgpu/commit/c73b19eceb187417370e82ebeedb8446554bce38",
            "https://github.com/PooyaEimandar/webgpu/commit/0a6f61003b053d261bf2e264063527aad3d8ba14",
            "https://github.com/PooyaEimandar/webgpu/commit/a65ec584ad446059707f4c0c9d1f14171540aa7a",
        ):
            self.assertIn(link, page.links)
        self.assertIn("cargo run --example ssao", document)
        self.assertIn("scripts/build-wasm.sh --release ssao", document)
        self.assertIn("cargo run --bin serve", document)
        for fact in (
            "SSAO neighbors</dt><dd>32",
            "Blur taps</dt><dd>25",
            "Render passes</dt><dd>8",
            "68,324", "1,957,752", "13,008", "2,039,084",
            "146 accessor declarations", "59 nodes", "16,832", "452 lines of WGSL",
            "Six graphics pipelines", "122,752", "Walking_1", "138 channels",
            "46-joint skin", "35,880 source <code>u16</code> indices", "11,960 triangles",
            "2,726,880", "143,520", "8,192", "46 of 128 matrices used", "184 bytes",
            "2,880,536", "7,074,840 fixed logical bytes", "4,194,304 RGBA8 bytes",
            "Rgba16Float", "Rgba8Unorm", "Depth32Float",
            "<strong>32</strong>", "<strong>29,491,200</strong>", "28.125 MiB",
            "8 bytes per physical pixel", "7.03125 MiB",
            "2.39996323 radians", "67 <code>textureLoad</code> calls", "404",
            "61,747,200 texture loads", "372,326,400 logical bytes", "355.078 MiB",
            "25 <code>textureLoad</code> calls", "100 logical bytes", "92,160,000 bytes",
            "12 MiB", "another 4 MiB", "0.56 + 0.44 * ao", "0.582",
            "44 px", "8&ndash;96", "2.15", "0.015", "0.35&ndash;4",
            "shader minimum 0.36", "0&ndash;2", "7 choices",
            "35,880", "11,962", "8 explicit scene/fullscreen draws", "47,845",
            "16,777,216", "22,118,400", "7,372,800", "53,343,256",
            "50.8721 MiB", "eight queue writes totaling 9,952 bytes", "1,760 bytes",
        ):
            self.assertIn(fact, document)
        for behavior in (
            "It does not upload a hemisphere kernel or a noise texture, and it does not reconstruct positions from depth.",
            "Despite the shader variable name <code>linear_depth</code>, this value is normalized device depth; the algorithm uses it only to reject cleared background and neighbor samples.",
            "The normalized sample index is squared before multiplying the screen-pixel radius, concentrating probes near the center while still reaching the selected outer radius.",
            "The blur is spatial rather than bilateral: it does not compare depth, position, or normal.",
            "The factor multiplies the complete lit color, including ambient, diffuse, specular, and already shadowed direct light.",
            "Turning SSAO off makes the raw pass return white before its 32-sample loop and removes AO from final composition, yet the raw fullscreen pass, 25-tap blur pass, blurred-texture load, and two AO targets remain active.",
            "“Reset SSAO” restores the SSAO toggle and six numeric defaults; it does not reset the shadow toggle or selected debug view.",
            "Shadow Mask can perform the normal 27 comparison taps and then repeat another 27 while rebuilding the diagnostic mask.",
            "Resizing recreates both masks, the four G-buffer attachments, and their bind groups.",
            "On a controls-stable frame, the listed uniform and joint buffers receive eight queue writes totaling 9,952 bytes before joystick or egui uploads.",
            "camera movement and Jax animation each cap their delta at 1/15 second. The five-second light phase uses the uncapped frame delta.",
        ):
            self.assertIn(behavior, document)
        for limitation in (
            "Off-screen occluders cannot contribute, disoccluded regions can change suddenly, and a radius measured in screen pixels changes its world-space reach with depth and resolution.",
            "Edge coordinates clamp, so samples beyond the viewport repeat edge texels.",
            "The 5&times;5 blur is not depth- or normal-aware, so it can bleed across silhouettes.",
            "Both AO targets store four channels although one is used.",
            "The composition shader normalizes the cleared zero background normal without the safe guard used by newer deferred shaders",
            "AO multiplies direct and specular lighting as well as ambient",
            "only Jax casts, three of eight lights are shadowed, each PCF kernel costs up to nine comparisons",
            "Its white base-color factor hides a loader/shader double multiplication that would affect non-white materials.",
            "Normals are not inverse-transpose transformed, specular is not gated by positive <code>N dot L</code>",
            "The animation helper also interpolates the file's STEP channels rather than preserving their declared transitions.",
        ):
            self.assertIn(limitation, document)
        self.assertIn("./ssao.js?build=seo-test", document)
        self.assertIn("./ssao_bg.wasm?build=seo-test", document)
        self.assertIn("screenshots/ssao.jpg", document)
        self.assertNotIn("screenshots/deferredshadows.jpg", document)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/ssao.rs">Read the SSAO source &nearr;</a>\n'
            '      <a href="../deferredshadows/">&larr; Previous: WebGPU deferred shadows</a>\n'
            '      <a href="../computeparticles/">Next: WebGPU compute particles &rarr;</a>\n'
            f'      <p class="copyright">&copy; <span data-current-year>{renderer.date.today().year}</span> <a href="https://pooya.ai">Pooya Eimandar</a>. All rights reserved.</p>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 32, "name": "WebGPU screen-space ambient occlusion", '
            '"url": "https://pooya.ai/webgpu/ssao/"',
            gallery,
        )
        self.assertIn(
            "<strong>WebGPU screen-space ambient occlusion</strong>", gallery)
        self.assertIn(
            'alt="Animated Jax character on a colorful checkerboard beside WebGPU SSAO controls"',
            gallery,
        )
        self.assertIn(
            "Generate a full-resolution AO mask with up to 32 spiral samples, soften it with a weighted 5 &times; 5 blur, and combine it with deferred shadows.",
            gallery,
        )
        self.assertLess(
            gallery.index('href="./deferredshadows/"'),
            gallery.index('href="./ssao/"'),
        )
        self.assertLess(
            gallery.index('href="./ssao/"'),
            gallery.index('href="./parallaxmapping/"'),
        )

    def test_repository_compute_particles_has_its_own_article_and_navigation(self):
        document = renderer.render_example("computeparticles", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "move-state-to-compute", "load-two-ktx-textures", "seed-particles",
            "dispatch-simulation", "render-billboards", "steer-repulsor",
            "record-three-passes", "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"],
            "computeparticles",
        )
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: compute particles",
        )
        self.assertEqual(
            self.json_ld(document)["headline"],
            "WebGPU Compute Particles in Rust: 262,144 GPU Billboards",
        )
        self.assertEqual(
            page.canonical,
            "https://pooya.ai/webgpu/computeparticles/",
        )
        self.assertEqual(
            page.metadata["description"],
            "Build a WebGPU compute particle system in Rust and wgpu: update 262,144 particles, ping-pong storage buffers, and render additive textured billboards.",
        )
        self.assertEqual(
            page.metadata["og:image"],
            "https://pooya.ai/webgpu/screenshots/computeparticles.jpg",
        )
        self.assertEqual(
            page.metadata["og:image:alt"],
            "WebGPU compute particles forming dense cyan, magenta, yellow, and green ribbons around a moving repulsor.",
        )
        self.assertEqual(page.metadata["og:image:width"], "1280")
        self.assertEqual(page.metadata["og:image:height"], "720")
        self.assertIn(
            '<p class="eyebrow">WebGPU notes &nbsp; / &nbsp; 34</p>',
            document,
        )
        self.assertIn(
            '<h1 id="article-title">WebGPU Compute Particles in Rust: 262,144 GPU Billboards</h1>',
            document,
        )
        for caption, caption_text in (
            ("compute-particles-assets-caption",
             "Compute Particles runtime assets and embedded inputs"),
            ("compute-particles-resources-caption",
             "Fixed logical GPU resources owned by Compute Particles"),
            ("compute-particles-simulation-caption",
             "Compute simulation constants and state transitions"),
            ("compute-particles-render-caption",
             "Billboard expansion and fragment composition"),
            ("compute-particles-passes-caption",
             "Compute Particles frame graph and submitted work"),
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
            self.assertIn(
                f'<caption id="{caption}">{caption_text}</caption>', document)
        self.assertEqual(document.count('<div class="article-table"'), 5)
        self.assertIn("../ssao/#about", page.links)
        self.assertIn("../ssao/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/computeparticles.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/computeparticles.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/web/index.html",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/particle01_rgba.ktx",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/particle_gradient_rgba.ktx",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/fonts/Vazirmatn-Regular.ttf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/fonts/Vazirmatn-LICENSE.txt",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/computeparticles.jpg",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/computeparticles.webp",
            "https://github.com/PooyaEimandar/sib",
            "https://github.com/PooyaEimandar/webgpu/commit/c5a40b3e41a72f0d2b29e8899777fcd47781a511",
            "https://github.com/PooyaEimandar/webgpu/commit/550d035fd99724bb9da48c041c06feac5aee3c80",
            "https://github.com/PooyaEimandar/webgpu/commit/7bb1e1b93fad1cf7683913cfa79394e50b137f98",
            "https://github.com/PooyaEimandar/webgpu/commit/7f54cadf3f26d9e07c19d16d9251c7832b91a44e",
            "https://github.com/SaschaWillems/Vulkan-Assets/blob/main/textures/particle01_rgba.ktx",
            "https://github.com/SaschaWillems/Vulkan-Assets/blob/main/textures/particle_gradient_rgba.ktx",
            "https://github.com/SaschaWillems/Vulkan-Assets/blob/main/README.md",
        ):
            self.assertIn(link, page.links)
        self.assertIn("cargo run --example computeparticles", document)
        self.assertIn(
            "scripts/build-wasm.sh --release computeparticles", document)
        self.assertIn("cargo run --bin serve", document)
        for fact in (
            "Particles</dt><dd>262,144",
            "Compute workgroups</dt><dd>1,024",
            "Ping-pong storage</dt><dd>16 MiB",
            "16,484", "64&times;64 RGBA8; 1 mip", "16,384 bytes",
            "1,124", "256&times;0 header; RGBA8; 1 mip",
            "1,024 bytes at 256&times;1",
            "4,258", "145 lines of WGSL", "122,752",
            "17,608 stored bytes", "17,408 texel bytes",
            "0x5eed_cafe", "32 bytes per record",
            "8,388,608 bytes per 262,144-particle array",
            "16,794,656", "16.0166 MiB",
            "ceil(262144 / 256)", "1,024 workgroups", "262,144 invocations",
            "Distance squared &ge; 0.0004", "clamp to &plusmn;0.998",
            "16 MiB of particle payload", "1,572,864 vertex invocations",
            "524,288 triangles", "eight-surface-pixel square", "brightened by 1.75",
            "12.5-second loop", "one compute pass and two render passes",
            "16,777,216 fragment candidates",
        ):
            self.assertIn(fact, document)
        for behavior in (
            "There is no per-frame particle upload or readback.",
            "The target is a repulsor.",
            "When either coordinate leaves the clip-space square, the particle is clamped to &plusmn;0.998 and its velocity is reflected at one tenth of its previous magnitude plus an attraction back toward the target.",
            "The first frame reads A, writes B, renders B, and marks B active. The next reads B, writes A, renders A, and repeats.",
            "Sprite alpha is one throughout, so coverage comes from a smoothstep over the maximum RGB component.",
            "After the first valid pointer update, <code>pointer_active</code> remains true.",
            "Releasing the mouse, ending a touch, or leaving the canvas stops further updates but does not return to automatic movement",
            "On resize, Rust rewrites the uniform with the new physical surface dimensions and rebuilds the text overlay. The particle state remains intact.",
            "The fixed simulation performs one 32-byte uniform queue write per frame; <code>params1.w</code> is currently unused.",
            "Native loading gives each request a worker thread. WebAssembly sends both URLs to one temporary Worker, whose <code>Promise.all</code> fetches them concurrently.",
            "Its displayed frame rate is capture text, not a benchmark.",
        ):
            self.assertIn(behavior, document)
        for limitation in (
            "This is not an N-body simulation. Particles do not observe one another, exchange momentum, collide, spawn, die, compact, or sort.",
            "The integration is frame-rate dependent because repulsive acceleration is added to velocity once per frame without multiplying by the time step.",
            "The example has no camera, depth attachment, depth test, culling, or multisampling.",
            "Billboards use normalized device positions rather than a world-space camera.",
            "They have no depth ordering, soft-particle intersection, culling, LOD, indirect draw, or MSAA.",
            "Both textures have one mip and use linear <code>Rgba8Unorm</code>, so the gradient is not decoded as sRGB and mip filtering has no lower level to select.",
            "The automatic target has no visual marker, and pointer mode has no way back to automatic motion without restarting.",
            "Boundary response is a visual rule rather than a physically derived collision.",
            "the apparent trails come from dense instantaneous overlap, not accumulated frames.",
        ):
            self.assertIn(limitation, document)
        self.assertIn("./computeparticles.js?build=seo-test", document)
        self.assertIn("./computeparticles_bg.wasm?build=seo-test", document)
        self.assertIn("screenshots/computeparticles.jpg", document)
        self.assertNotIn("screenshots/ssao.jpg", document)
        self.assertIn("../computecloth/", page.links)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/computeparticles.rs">Read the Compute Particles source &nearr;</a>\n'
            '      <a href="../ssao/">&larr; Previous: WebGPU screen-space ambient occlusion</a>\n'
            '      <a href="../computecloth/">Next: WebGPU compute cloth simulation &rarr;</a>\n'
            f'      <p class="copyright">&copy; <span data-current-year>{renderer.date.today().year}</span> <a href="https://pooya.ai">Pooya Eimandar</a>. All rights reserved.</p>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 4, "name": "WebGPU compute particles", '
            '"url": "https://pooya.ai/webgpu/computeparticles/"',
            gallery,
        )
        self.assertIn("<strong>WebGPU compute particles</strong>", gallery)
        self.assertIn(
            'alt="Rainbow WebGPU compute particles forming flowing bands across a black background"',
            gallery,
        )
        self.assertIn(
            "Update 262,144 particles in 1,024 compute workgroups, ping-pong two storage buffers, and render the GPU-written state as additive billboards.",
            gallery,
        )
        self.assertLess(
            gallery.index('href="./particlesystem/"'),
            gallery.index('href="./computeparticles/"'),
        )
        self.assertLess(
            gallery.index('href="./computeparticles/"'),
            gallery.index('href="./computecloth/"'),
        )

    def test_repository_compute_nbody_has_its_own_article_and_navigation(self):
        document = renderer.render_example("computenbody", "seo-test")
        page = HeadParser(document)
        self.assertEqual(page.h2_ids, [
            "tile-all-pairs", "load-particle-assets", "seed-six-clusters",
            "calculate-forces", "integrate-in-place", "render-particles",
            "tune-simulation", "record-four-passes", "run-the-example",
        ])
        self.assertNotIn("data-static-demo", page.elements_by_id["about"])
        self.assertEqual(
            page.elements_by_id["demo"]["data-example"],
            "computenbody",
        )
        self.assertEqual(
            page.elements_by_id["demo"]["aria-label"],
            "Live WebGPU demo: N-body simulation",
        )
        self.assertEqual(
            self.json_ld(document)["headline"],
            "WebGPU N-Body Simulation in Rust: 12,288 GPU Particles",
        )
        self.assertEqual(
            page.canonical,
            "https://pooya.ai/webgpu/computenbody/",
        )
        self.assertEqual(
            page.metadata["description"],
            "Build a WebGPU N-body simulation in Rust and wgpu: calculate all-pairs gravity with tiled WGSL compute shaders, then render 12,288 additive billboards.",
        )
        self.assertEqual(
            page.metadata["og:image"],
            "https://pooya.ai/webgpu/screenshots/computenbody.jpg",
        )
        self.assertEqual(
            page.metadata["og:image:alt"],
            "WebGPU N-body simulation showing six colorful particle clusters interacting in 3D space, with live controls open.",
        )
        self.assertEqual(page.metadata["og:image:width"], "1280")
        self.assertEqual(page.metadata["og:image:height"], "720")
        self.assertIn(
            '<p class="eyebrow">WebGPU notes &nbsp; / &nbsp; 37</p>',
            document,
        )
        self.assertIn(
            '<h1 id="article-title">WebGPU N-Body Simulation in Rust: 12,288 GPU Particles</h1>',
            document,
        )
        for caption, caption_text in (
            ("nbody-assets-caption", "Compute N-body runtime assets and embedded inputs"),
            ("nbody-resources-caption",
             "Compute N-body initialization and fixed logical GPU resources"),
            ("nbody-tiling-caption",
             "Source-level calculate-pass workload for 12,288 bodies"),
            ("nbody-controls-caption",
             "Compute N-body live controls and current defaults"),
            ("nbody-passes-caption", "Compute N-body frame graph and submitted work"),
        ):
            self.assertIn(
                f'role="region" aria-labelledby="{caption}" tabindex="0"',
                document,
            )
            self.assertIn(
                f'<caption id="{caption}">{caption_text}</caption>', document)
        self.assertEqual(document.count('<div class="article-table"'), 5)
        self.assertIn("../computecullandlod/#about", page.links)
        self.assertIn("../computecullandlod/", page.links)
        for link in (
            "https://github.com/PooyaEimandar/webgpu/blob/main/examples/computenbody.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/computenbody_compute.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/shaders/computenbody_render.wgsl",
            "https://github.com/PooyaEimandar/webgpu/blob/main/web/index.html",
            "https://github.com/PooyaEimandar/webgpu/blob/main/src/asset.rs",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/particle01_rgba.ktx",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/textures/particle_gradient_rgba.ktx",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/fonts/Vazirmatn-Regular.ttf",
            "https://github.com/PooyaEimandar/webgpu/blob/main/assets/fonts/Vazirmatn-LICENSE.txt",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/computenbody.jpg",
            "https://github.com/PooyaEimandar/webgpu/blob/main/screenshots/computenbody.webp",
            "https://github.com/PooyaEimandar/sib",
            "https://github.com/PooyaEimandar/webgpu/commit/301b7920be243360dc51f24ecb4b08c2dba24e14",
            "https://github.com/PooyaEimandar/webgpu/commit/550d035fd99724bb9da48c041c06feac5aee3c80",
            "https://github.com/PooyaEimandar/webgpu/commit/c46fad85c97dc6995740d02cf54629dbe7bb5a44",
            "https://github.com/PooyaEimandar/webgpu/commit/de16e06e35263457f2b71db9ab55b4f2376f21f8",
            "https://github.com/PooyaEimandar/webgpu/commit/c73b19eceb187417370e82ebeedb8446554bce38",
            "https://github.com/PooyaEimandar/webgpu/commit/7bb1e1b93fad1cf7683913cfa79394e50b137f98",
            "https://github.com/SaschaWillems/Vulkan-Assets/blob/main/textures/particle01_rgba.ktx",
            "https://github.com/SaschaWillems/Vulkan-Assets/blob/main/textures/particle_gradient_rgba.ktx",
            "https://github.com/SaschaWillems/Vulkan-Assets/blob/main/README.md",
        ):
            self.assertIn(link, page.links)
        self.assertIn("cargo run --example computenbody", document)
        self.assertIn("scripts/build-wasm.sh --release computenbody", document)
        self.assertIn("cargo run --bin serve", document)
        for fact in (
            "Particles</dt><dd>12,288",
            "Pair terms per frame</dt><dd>150,994,944",
            "Shared tile per workgroup</dt><dd>4 KiB",
            "16,484", "64&times;64 RGBA8; 1 mip", "16,384-byte",
            "1,124", "256&times;0 header; RGBA8; 1 mip",
            "1,024 bytes at 256&times;1", "2,445", "85 lines",
            "2,807", "89 lines", "122,752", "17,608", "17,408",
            "0x4d2f_6a31", "2,048 particles", "2,047", "90,000",
            "37.5 to 75", "12,282 records", "393,216", "176 bytes",
            "410,800", "0.3918 MiB", "48 workgroups of 256 threads",
            "2,304 workgroup tiles", "589,824 loads", "9,437,184 bytes = 9 MiB",
            "2,415,919,104 bytes = 2.25 GiB", "4,608 workgroup barrier points",
            "gravity * delta * mass / (distance_squared + soften)^power",
            "default power 0.75", "exponent of 1.5", "1e&minus;6 floor",
            "clamp(frame_delta * 0.05 * time_scale, 0, 1/120)",
            "73,728 vertex invocations", "24,576 triangles", "60&deg;",
            "14 units", "75&deg; yaw", "26&deg; pitch", "0.1 and 512",
            "clamped from 1 to 128 surface pixels", "0.0001&ndash;0.006",
            "0.35&ndash;1.4", "0.005&ndash;0.35", "0.25&ndash;3",
            "0.2&ndash;4", "48 &times; 256 compute invocations",
            "one 176-byte example uniform queue write", "393,216-byte buffer write",
        ):
            self.assertIn(fact, document)
        for behavior in (
            "one storage buffer remains in place. A calculate pass reads positions and writes velocities; a second compute pass integrates those new velocities into positions.",
            "Tiling reduces source-position storage loads by a factor of 256 while preserving the full quadratic arithmetic.",
            "These six heavy bodies are not pinned attractors.",
            "Reset particles” regenerates the deterministic CPU vector and writes all 393,216 bytes back into the storage buffer without reallocating it.",
            "Includes zero-contribution self terms",
            "These are source-level counts, not measured memory traffic or execution time.",
            "That field separation makes one in-place buffer safe without ping-ponging.",
            "The phase therefore also increases mass by <code>delta_t * vel.w</code> every active frame.",
            "Pause and time scale zero freeze numerical state, but both compute passes still dispatch and the calculate pass still evaluates every pair.",
            "The six 90,000-mass bodies reach the upper clamp.",
            "KTX sprite alpha is opaque everywhere and the surface clears with alpha one, final surface alpha remains one.",
            "Changing a control rewrites the same 176-byte uniform without rebuilding pipelines, bindings, textures, or buffers.",
            "The screenshot shows time scale 2.20, power 0.80, soften 0.070, and brightness 2.45, not the defaults above.",
            "The renderer records the same four passes in all three cases.",
        ):
            self.assertIn(behavior, document)
        for limitation in (
            "With default power 0.75 the law is not Newton's inverse-square acceleration",
            "The algorithm remains O(N&sup2;).",
            "There is no Barnes&ndash;Hut tree, fast multipole method, neighbor cutoff, adaptive body count, subgroup optimization, GPU timing, or capability-based workload selection.",
            "The force exponent is artistic rather than Newtonian, and single-precision semi-implicit Euler integration has no adaptive step or energy correction.",
            "Bodies do not collide, merge, conserve a fixed system energy, or leave trails in a history buffer.",
            "The full-<code>vec4</code> integrate operation couples gradient phase into mass and projected size.",
            "The fixed camera cannot orbit, pan, or zoom.",
            "Additive billboards have no depth test, sorting, culling, indirect draw, soft-particle intersection, HDR target, exposure, tone mapping, or MSAA.",
            "Their opaque sprite alpha means black corners still execute fragment work.",
        ):
            self.assertIn(limitation, document)
        self.assertIn("./computenbody.js?build=seo-test", document)
        self.assertIn("./computenbody_bg.wasm?build=seo-test", document)
        self.assertIn("screenshots/computenbody.jpg", document)
        self.assertNotIn("screenshots/computeparticles.jpg", document)
        self.assertIn("../computeraytracing/", page.links)
        self.assertIn(
            '<footer class="article-footer">\n'
            '      <a href="https://github.com/PooyaEimandar/webgpu/blob/main/examples/computenbody.rs">Read the Compute N-body source &nearr;</a>\n'
            '      <a href="../computecullandlod/">&larr; Previous: WebGPU compute culling and LOD</a>\n'
            '      <a href="../computeraytracing/">Next: WebGPU compute shader ray tracing &rarr;</a>\n'
            f'      <p class="copyright">&copy; <span data-current-year>{renderer.date.today().year}</span> <a href="https://pooya.ai">Pooya Eimandar</a>. All rights reserved.</p>',
            document,
        )
        gallery = (renderer.WEB_ROOT /
                   "index.html").read_text(encoding="utf-8")
        self.assertIn(
            '"position": 7, "name": "WebGPU N-body simulation", '
            '"url": "https://pooya.ai/webgpu/computenbody/"',
            gallery,
        )
        self.assertIn("<strong>WebGPU N-body simulation</strong>", gallery)
        self.assertIn(
            'alt="Luminous colored particle clusters evolving in the WebGPU N-body simulation beside live controls"',
            gallery,
        )
        self.assertIn(
            "Evaluate 151 million pairwise interactions with tiled workgroup memory, then render 12,288 additive billboards with live egui controls.",
            gallery,
        )
        self.assertLess(
            gallery.index('href="./metropolis/"'),
            gallery.index('href="./computenbody/"'),
        )
        self.assertLess(
            gallery.index('href="./computenbody/"'),
            gallery.index('href="./computeraytracing/"'),
        )


if __name__ == "__main__":
    unittest.main()
