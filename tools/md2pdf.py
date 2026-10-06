#!/usr/bin/env python3
"""
md2pdf.py – Gera os PDFs do white paper a partir do Markdown de docs/.

Markdown -> HTML (python-markdown; blocos ```math renderizados com KaTeX) -> PDF pelo Chrome/Chromium
headless (Playwright). Sem LaTeX nem pandoc.

Uso:
    pip install markdown playwright          # e um Chrome/Chromium instalado
    python tools/md2pdf.py                   # docs/WHITE_PAPER_{PT,EN}.md -> docs/pdf/*.pdf
    python tools/md2pdf.py --chrome /usr/bin/google-chrome
"""
import argparse
import html
import re
import time
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = {"pt": ROOT / "docs" / "WHITE_PAPER_PT.md", "en": ROOT / "docs" / "WHITE_PAPER_EN.md"}
FOOTER = {"pt": "Página", "en": "Page"}
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

CSS = """
 @page { size: A4; margin: 22mm 20mm 20mm 20mm; }
 body { font-family: 'Source Serif 4','DejaVu Serif',Georgia,serif; font-size: 10.5pt; line-height: 1.5; color:#1a1a1a; }
 h1 { font-family: 'DejaVu Sans',Arial,sans-serif; font-size: 22pt; color:#2b3a67; margin: 0 0 4mm; }
 h2 { font-family: 'DejaVu Sans',Arial,sans-serif; font-size: 14pt; color:#2b3a67; border-bottom: 1.5px solid #c9d1e6;
      padding-bottom: 2px; margin-top: 9mm; page-break-after: avoid; }
 h3 { font-family: 'DejaVu Sans',Arial,sans-serif; font-size: 11.5pt; color:#34477a; margin-top: 6mm; page-break-after: avoid; }
 table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 8.8pt; page-break-inside: avoid; }
 th, td { border: 1px solid #c9ccd6; padding: 3px 5px; vertical-align: top; text-align: left; }
 th { background: #eef1f8; }
 tr:nth-child(even) td { background: #fafbfd; }
 code { font-family: 'DejaVu Sans Mono',monospace; font-size: 8.8pt; background:#f3f4f7; padding: 0 2px; border-radius: 2px; }
 pre { background:#f3f4f7; padding: 3mm; border-radius: 3px; font-size: 8.5pt; white-space: pre-wrap; page-break-inside: avoid; }
 pre code { background: none; padding: 0; }
 blockquote { border-left: 3px solid #9aa8d1; margin: 3mm 0; padding: 1mm 4mm; color:#444; background:#f6f7fb; }
 .math { margin: 3mm 0; page-break-inside: avoid; overflow-x: auto; }
 .katex-display { margin: 0.4em 0; }
 a { color:#2b4fa8; text-decoration: none; }
"""


def to_html(md_text: str, lang: str) -> str:
    blocks = []

    def keep(m):
        blocks.append(m.group(1).strip())
        return f"\n\nMATHBLOCK{len(blocks) - 1}END\n\n"

    md_text = re.sub(r"```math\n(.*?)```", keep, md_text, flags=re.S)
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    for i, b in enumerate(blocks):
        body = body.replace(f"<p>MATHBLOCK{i}END</p>", '<div class="math">$$' + html.escape(b) + "$$</div>")
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}}]}});document.body.dataset.done=1"></script>
<style>{CSS}</style></head><body>{body}</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "pdf"), help="pasta de saída")
    ap.add_argument("--chrome", default=None, help="executável do Chrome/Chromium (padrão: o do Playwright)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        kw = {"executable_path": args.chrome, "args": ["--no-sandbox"]} if args.chrome else {}
        browser = p.chromium.launch(**kw)
        for lang, src in DOCS.items():
            page_html = out / f"TextureLab_White_Paper_{lang.upper()}.html"
            page_html.write_text(to_html(src.read_text(encoding="utf-8"), lang), encoding="utf-8")
            pg = browser.new_page()
            pg.goto(page_html.resolve().as_uri())
            pg.wait_for_function("document.body.dataset.done == '1'", timeout=60000)
            time.sleep(1)
            errors = pg.evaluate("document.querySelectorAll('.katex-error').length")
            pdf = page_html.with_suffix(".pdf")
            pg.pdf(path=str(pdf), format="A4", print_background=True, display_header_footer=True,
                   header_template="<span></span>",
                   footer_template="<div style='width:100%;font-size:8px;color:#777;text-align:center'>"
                                   f"TextureLab — White Paper · {FOOTER[lang]} <span class='pageNumber'></span>/"
                                   "<span class='totalPages'></span></div>",
                   margin={"top": "20mm", "bottom": "18mm", "left": "18mm", "right": "18mm"})
            page_html.unlink()
            print(f"{pdf}  (fórmulas com erro: {errors})")
        browser.close()


if __name__ == "__main__":
    main()
