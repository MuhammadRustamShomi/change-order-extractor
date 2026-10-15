"""
Render a few of the raw text samples into PDFs so the pipeline is tested on
real PDF input, not just .txt files.

  - co_001, co_003, co_005 -> PDFs with a normal text layer
  - co_008                 -> an IMAGE-ONLY PDF (simulates a phone scan with
                              no text layer). The pipeline must detect this
                              and refuse to guess, instead of returning junk.

Run:  python scripts/make_samples.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

RAW = Path(__file__).resolve().parent.parent / "samples" / "raw"


def text_to_pdf(txt_path: Path, pdf_path: Path) -> None:
    """Write each line of the text file in a monospace font (keeps columns)."""
    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    c.setFont("Courier", 9)
    y = 750
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        c.drawString(40, y, line)
        y -= 12
    c.save()


def text_to_image_pdf(txt_path: Path, pdf_path: Path) -> None:
    """Draw the text onto an image and save it as a PDF with no text layer."""
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    img = Image.new("RGB", (1275, 1650), "white")  # 8.5x11in at 150dpi
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 60
    for line in lines:
        draw.text((60, y), line, fill="black", font=font)
        y += 22
    # Slight rotation, like a crooked phone scan.
    img = img.rotate(0.8, expand=False, fillcolor="white")
    img.save(pdf_path, "PDF", resolution=150)


def main() -> None:
    for stem in ["co_001_formal_form", "co_003_ocr_noisy", "co_005_aia_contract_sum"]:
        text_to_pdf(RAW / f"{stem}.txt", RAW / f"{stem}.pdf")
        print("wrote", f"{stem}.pdf")
    text_to_image_pdf(RAW / "co_001_formal_form.txt", RAW / "co_008_scanned_image.pdf")
    print("wrote co_008_scanned_image.pdf (image only)")


if __name__ == "__main__":
    main()
