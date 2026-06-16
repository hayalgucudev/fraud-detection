"""Convert REPORT.md to REPORT.pdf with embedded figures."""

import base64
import re
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "REPORT.md"
REPORT_PDF = ROOT / "REPORT.pdf"

CSS = """
@page { size: A4; margin: 2cm; }
body {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.45;
  color: #1a1a1a;
}
h1 { font-size: 20pt; color: #0d47a1; border-bottom: 2px solid #0d47a1; padding-bottom: 6px; }
h2 { font-size: 15pt; color: #1565c0; margin-top: 22px; }
h3 { font-size: 12pt; color: #1976d2; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; }
th { background: #e3f2fd; }
code { background: #f5f5f5; padding: 1px 4px; font-size: 9pt; }
pre { background: #f5f5f5; padding: 10px; font-size: 9pt; overflow: wrap; }
img { max-width: 100%; height: auto; margin: 10px 0; display: block; }
hr { border: none; border-top: 1px solid #ddd; margin: 20px 0; }
strong { color: #0d47a1; }
"""


def embed_images(md_text: str, base_dir: Path) -> str:
    """Replace markdown image paths with base64 data URIs for PDF rendering."""

    def replacer(match: re.Match) -> str:
        alt, path = match.group(1), match.group(2)
        img_path = (base_dir / path).resolve()
        if not img_path.exists():
            return f"*[Figure not found: {path}]*"
        mime = "png" if img_path.suffix.lower() == ".png" else "jpeg"
        data = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return f"![{alt}](data:image/{mime};base64,{data})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replacer, md_text)


def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    md_text = embed_images(md_text, md_path.parent)
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>{CSS}</style></head>
<body>{html_body}</body></html>"""

    with pdf_path.open("wb") as out:
        status = pisa.CreatePDF(html, dest=out, encoding="utf-8")
    if status.err:
        raise RuntimeError(f"PDF generation failed with {status.err} error(s)")
    print(f"Created {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    md_to_pdf(REPORT_MD, REPORT_PDF)
