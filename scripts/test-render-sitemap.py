#!/usr/bin/env python3
"""Test sitemap discovery without building Rust or WebAssembly examples."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


sys.dont_write_bytecode = True
SCRIPT = Path(__file__).with_name("render-sitemap.py")
SPEC = importlib.util.spec_from_file_location("render_sitemap", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)
NAMESPACES = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
CANONICAL_ROOT = "https://pooya.ai/webgpu/"


class RenderSitemapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="webgpu-sitemap-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.web_root = self.root / "web"
        self.examples_root = self.root / "examples"
        self.web_root.mkdir()
        self.examples_root.mkdir()
        self.add_page("index.html")

    def add_page(self, relative_path):
        page = self.web_root / relative_path
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("<!doctype html><title>Test page</title>", encoding="utf-8")
        return page

    def add_source(self, slug):
        source = self.examples_root / f"{slug}.rs"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fn main() {}", encoding="utf-8")
        return source

    def render(self):
        return renderer.render_sitemap(self.web_root, self.examples_root)

    def urls(self, document):
        root = ET.fromstring(document)
        self.assertEqual(root.tag, f"{{{NAMESPACES['s']}}}urlset")
        return [entry.text for entry in root.findall("s:url/s:loc", NAMESPACES)]

    def test_gallery_is_included_with_no_built_demos(self):
        self.add_source("triangle")
        document = self.render()
        self.assertTrue(document.startswith("<?xml "))
        self.assertEqual(self.urls(document), [CANONICAL_ROOT])

    def test_only_built_demo_pages_are_included_in_deterministic_order(self):
        for slug in ("vertexattributes", "triangle", "texture"):
            self.add_source(slug)
        self.add_page("vertexattributes/index.html")
        self.add_page("triangle/index.html")
        self.assertEqual(self.urls(self.render()), [
            CANONICAL_ROOT,
            CANONICAL_ROOT + "triangle/",
            CANONICAL_ROOT + "vertexattributes/",
        ])

    def test_assets_unknown_folders_nested_pages_and_unbuilt_demos_are_excluded(self):
        self.add_source("triangle")
        self.add_source("texture")
        self.add_source("nested/helper")
        self.add_page("triangle/index.html")
        for path in (
            "assets/index.html", "screenshots/index.html", "unknown/index.html",
            "triangle/nested/index.html", "texture/nested/index.html",
            "nested/helper/index.html", "_article_qa_preview.html",
        ):
            self.add_page(path)
        self.assertEqual(self.urls(self.render()), [CANONICAL_ROOT, CANONICAL_ROOT + "triangle/"])

    def test_invalid_slugs_cannot_produce_unsafe_or_noncanonical_urls(self):
        for slug in ("Triangle", "two words", "x&y", "x#part", "x?query", "_preview", ".hidden"):
            self.add_source(slug)
            self.add_page(f"{slug}/index.html")
        self.add_source("demo-2_test")
        self.add_page("demo-2_test/index.html")
        self.assertEqual(self.urls(self.render()), [CANONICAL_ROOT, CANONICAL_ROOT + "demo-2_test/"])

    def test_directory_named_index_html_is_not_a_page(self):
        self.add_source("triangle")
        (self.web_root / "triangle" / "index.html").mkdir(parents=True)
        self.assertEqual(self.urls(self.render()), [CANONICAL_ROOT])

    def test_symlink_demo_directories_and_pages_are_excluded(self):
        outside = self.root / "outside"
        outside.mkdir()
        outside_page = outside / "index.html"
        outside_page.write_text("External page", encoding="utf-8")
        self.add_source("triangle")
        self.add_source("texture")
        (self.web_root / "triangle").symlink_to(outside, target_is_directory=True)
        (self.web_root / "texture").mkdir()
        (self.web_root / "texture" / "index.html").symlink_to(outside_page)
        self.assertEqual(self.urls(self.render()), [CANONICAL_ROOT])

    def test_sitemap_contains_only_canonical_locations_without_invented_metadata(self):
        self.add_source("triangle")
        self.add_page("triangle/index.html")
        root = ET.fromstring(self.render())
        for entry in root:
            self.assertEqual([child.tag for child in entry], [f"{{{NAMESPACES['s']}}}loc"])
            self.assertTrue(entry[0].text.startswith(CANONICAL_ROOT))
            self.assertTrue(entry[0].text.endswith("/"))
            self.assertNotIn("index.html", entry[0].text)

    def test_missing_build_output_and_gallery_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Build output directory not found"):
            renderer.render_sitemap(self.root / "missing", self.examples_root)
        (self.web_root / "index.html").unlink()
        with self.assertRaisesRegex(ValueError, "Gallery page not found"):
            self.render()

    def test_missing_examples_directory_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Examples directory not found"):
            renderer.render_sitemap(self.web_root, self.root / "missing")

    def test_cli_renders_xml_to_stdout_using_repository_example_names(self):
        self.add_page("triangle/index.html")
        self.add_page("unknown/index.html")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.web_root)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(self.urls(result.stdout), [CANONICAL_ROOT, CANONICAL_ROOT + "triangle/"])

    def test_cli_failure_has_a_useful_error_and_no_partial_xml(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.root / "missing")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("render-sitemap: Build output directory not found", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
