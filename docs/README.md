# Building the Documentation

## Prerequisites

Install the project and documentation dependency group from the committed lockfile:

```bash
uv sync --group docs
```

## Building

To build the HTML documentation:

```bash
uv run make -C docs html
```

The documentation will be generated in `docs/_build/html/`. Open `_build/html/index.html` in your browser to view it.

## Other Build Formats

- `uv run make -C docs latexpdf` - Build PDF documentation
- `uv run make -C docs epub` - Build EPUB documentation
- `uv run make -C docs clean` - Clean build directory

## Viewing Locally

After building, you can view the documentation by opening `_build/html/index.html` in your web browser, or by using a simple HTTP server:

```bash
cd docs/_build/html
uv run python -m http.server 8000
```

Then visit http://localhost:8000 in your browser.
