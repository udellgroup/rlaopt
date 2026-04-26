# Building the Documentation

## Prerequisites

Install the required dependencies:

```bash
pip install -r ../requirements-dev.txt
```

Or install just the documentation dependencies:

```bash
pip install sphinx sphinx-book-theme myst-parser
```

## Building

To build the HTML documentation:

```bash
cd docs
make html
```

The documentation will be generated in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

## Other Build Formats

- `make latexpdf` - Build PDF documentation
- `make epub` - Build EPUB documentation
- `make clean` - Clean build directory

## Viewing Locally

After building, you can view the documentation by opening `_build/html/index.html` in your web browser, or by using a simple HTTP server:

```bash
cd _build/html
python -m http.server 8000
```

Then visit http://localhost:8000 in your browser.
