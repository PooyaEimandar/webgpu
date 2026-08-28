# Demo articles

Articles are optional, authored HTML rendered into the demo page at build time.
Readers and search engines receive the complete text in the initial HTML, even
when WebGPU is unavailable. Demos without an article keep their existing
fullscreen layout.

The triangle, vertex attributes, texture, texture cubemap, texture array, and
texture mipmap generation examples include articles. Use their HTML as
references for code snippets, screenshots, section links, and comparison
tables. Keep the footer links in reading order: triangle, vertex attributes,
texture loading, texture cubemap, texture array, then texture mipmap generation.

## Add an article

Create both `web/articles/EXAMPLE.json` and `web/articles/EXAMPLE.html`, where
`EXAMPLE` matches the Rust example name. Use a lowercase slug starting with a
letter; subsequent characters may include numbers, hyphens, and underscores.
The renderer fails if only one of the two files exists.

The metadata file requires four nonempty strings:

```json
{
  "title": "WebGPU Triangle in Rust: wgpu and WGSL",
  "description": "A short, specific summary of what the example teaches.",
  "breadcrumbName": "Triangle",
  "image": "triangle.jpg",
  "imageWidth": 1280,
  "imageHeight": 732,
  "imageAlt": "A triangle with a smooth RGB gradient on a black background"
}
```

`image` is a filename in `screenshots/`, not a URL or a path. Use an existing
`.jpg` or `.webp` screenshot; the WebAssembly build copies those formats to the
published site. Write a clear headline and description in natural language.
Keep explanations grounded in the actual Rust and WGSL code.

`breadcrumbName` is optional and defaults to the title. When supplied,
`imageWidth` and `imageHeight` must both be positive integers matching the
actual screenshot; they are included in Open Graph image metadata.

The HTML file is a trusted, repository-authored fragment, not a full page:

```html
<article class="example-article" id="about" tabindex="-1" aria-labelledby="article-title">
  <div class="article-inner">
    __ARTICLE_BREADCRUMBS__
    <h1 id="article-title">__ARTICLE_TITLE__</h1>
    <p>__ARTICLE_DESCRIPTION__</p>
    <h2>How the example works</h2>
    <p>Explain the visible result and the code that produces it.</p>
  </div>
</article>
```

Use the shared article styles in `web/example.html` and the triangle article as
the layout reference. Keep `id="about"` for the More info anchor and
`tabindex="-1"` for keyboard focus after navigation. The title and description
placeholders are HTML-escaped from metadata; `__EXAMPLE__` is also available in
article fragments. Escape code snippets as HTML. Do not add a second document
head or duplicate the generated metadata.

`__ARTICLE_BREADCRUMBS__` renders a visible link to the gallery and the current
article name, with matching `BreadcrumbList` structured data. If the token is
omitted, no breadcrumb schema is emitted. Add stable IDs to section headings
and ordinary anchor links for an optional table of contents.

Use `__ARTICLE_IMAGE__` for the screenshot path and `__ARTICLE_IMAGE_ALT__` for
its escaped alternative text. A visible `<img>` makes the result available
without WebGPU and discoverable without running JavaScript. Set its real
`width` and `height` to reserve layout space; the triangle article demonstrates
a WebP `<picture>` with JPG fallback, lazy loading, and a caption.

Use `&copy; <span data-current-year>__CURRENT_YEAR__</span>` in a copyright
notice to render the current year at build time and refresh it in the browser.

For a demo with no mouse, touch, or keyboard controls, add `data-static-demo`
to its `<article>` element, as the triangle article does. This disables pointer
events on the canvas and removes it from sequential keyboard focus, allowing
wheel and touch gestures over the demo to scroll the page. Leave this attribute
off demos that require canvas input.

This flag controls input handling, not animation: the vertex attributes demo
keeps rotating while allowing the reader to scroll over its canvas.

The renderer adds the More info link and SVG, article layout class, page title,
description, canonical URL, author, Open Graph and Twitter metadata, and
`BlogPosting` JSON-LD. The author is Pooya Eimandar and canonical URLs use
`https://pooya.ai/webgpu/EXAMPLE/`. No publication or modification dates are
invented.

The article allows large search image previews. Loading and WebGPU error
messages are excluded from snippets with `data-nosnippet`; the article itself
remains available for indexing.

## Build and test

Run the renderer tests from the repository root; they need only Python 3:

```sh
python3 scripts/test-render-example.py
python3 scripts/test-render-sitemap.py
```

Render HTML without rebuilding WebAssembly:

```sh
python3 scripts/render-example.py triangle preview > /tmp/triangle-preview.html
```

Build the complete demo, including its article:

```sh
scripts/build-wasm.sh --release triangle
```

The Python API is `render_example(example, build_id, web_root=WEB_ROOT) -> str`.
The optional `web_root` is useful for isolated test fixtures. Errors go to
stderr with a nonzero exit status when the renderer runs as a command.

## Search discovery after deployment

The WebAssembly build writes `target/web/sitemap.xml` (or the configured
`WEBGPU_WEB_ROOT`). It includes the gallery and actual built demo pages with
canonical `https://pooya.ai/webgpu/` URLs. It does not invent `lastmod` dates.
To regenerate the sitemap without rebuilding WebAssembly:

```sh
python3 scripts/render-sitemap.py target/web > target/web/sitemap.xml
```

After publishing, submit `https://pooya.ai/webgpu/sitemap.xml` in Google Search
Console or reference it from `https://pooya.ai/robots.txt` in the personal-site
repository. A `robots.txt` under `/webgpu/` cannot control crawling of this site.
Use Search Console URL Inspection to request a recrawl of changed articles.
The build does not submit URLs or change search engine settings automatically.

See Google's guidance on [sitemaps](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap),
[robots.txt location](https://developers.google.com/crawling/docs/robots-txt/create-robots-txt),
and [article structured data](https://developers.google.com/search/docs/appearance/structured-data/article).
