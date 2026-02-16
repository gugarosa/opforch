"""Build PDF report from MIGRATION_REPORT.md with embedded benchmark images.

Uses fpdf2 with Unicode TrueType fonts, auto-sized table columns,
zebra-striped rows, and proper markdown rendering.
"""

import os
import re
from pathlib import Path
from typing import List

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
MD_FILE = ROOT / "MIGRATION_REPORT.md"
OUT_PDF = ROOT / "MIGRATION_REPORT.pdf"

# Layout
PAGE_W = 210  # A4 width in mm
MARGIN_L = 12
MARGIN_R = 12
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

# Table colors (RGB)
TBL_HEADER_BG = (37, 60, 110)      # dark navy
TBL_HEADER_FG = (255, 255, 255)    # white text
TBL_ROW_EVEN = (245, 247, 252)     # light blue-grey
TBL_ROW_ODD = (255, 255, 255)      # white
TBL_BORDER = (200, 210, 225)       # soft blue border
TBL_ACCENT_FG = (25, 60, 120)      # accent for bold cells


class ReportPDF(FPDF):

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_left_margin(MARGIN_L)
        self.set_right_margin(MARGIN_R)
        self.set_auto_page_break(auto=True, margin=18)

        font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        try:
            self.add_font("Segoe", "", str(font_dir / "segoeui.ttf"))
            self.add_font("Segoe", "B", str(font_dir / "segoeuib.ttf"))
            self.add_font("Segoe", "I", str(font_dir / "segoeuii.ttf"))
            self.add_font("Segoe", "BI", str(font_dir / "segoeuiz.ttf"))
            self.add_font("Consolas", "", str(font_dir / "consola.ttf"))
            self.add_font("Consolas", "B", str(font_dir / "consolab.ttf"))
            self._bf = "Segoe"
            self._mf = "Consolas"
        except Exception:
            self._bf = "Helvetica"
            self._mf = "Courier"

        self.add_page()

    def header(self):
        self.set_font(self._bf, "I", 7)
        self.set_text_color(160, 160, 160)
        self.set_x(MARGIN_L)
        self.cell(CONTENT_W, 4, "OPFython \u2192 OPForch  |  Migration Report",
                  align="L", new_x="LMARGIN", new_y="NEXT")
        # Thin header line
        self.set_draw_color(200, 210, 225)
        self.line(MARGIN_L, self.get_y(), PAGE_W - MARGIN_R, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(200, 210, 225)
        self.line(MARGIN_L, self.get_y(), PAGE_W - MARGIN_R, self.get_y())
        self.set_font(self._bf, "", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()} / {{nb}}", align="C")


def clean(text: str) -> str:
    """Strip inline markdown formatting."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return text.strip()


def _measure_col_widths(pdf: ReportPDF, rows: List[List[str]],
                        font_size_header: float = 7.5,
                        font_size_body: float = 7.0) -> List[float]:
    """Compute column widths proportional to max content width, fitting CONTENT_W."""
    n_cols = len(rows[0])
    max_widths = [0.0] * n_cols

    for r_idx, row in enumerate(rows):
        is_header = (r_idx == 0)
        pdf.set_font(pdf._bf, "B" if is_header else "", 
                     font_size_header if is_header else font_size_body)
        for c_idx, cell in enumerate(row):
            w = pdf.get_string_width(clean(cell)) + 5  # padding
            if c_idx < n_cols:
                max_widths[c_idx] = max(max_widths[c_idx], w)

    total = sum(max_widths)
    if total <= 0:
        return [CONTENT_W / n_cols] * n_cols

    # Scale proportionally to fill CONTENT_W
    scale = CONTENT_W / total
    widths = [w * scale for w in max_widths]

    # Enforce minimum column width of 12mm
    min_w = 12.0
    for i in range(n_cols):
        if widths[i] < min_w:
            widths[i] = min_w

    # Re-scale if we exceeded
    total2 = sum(widths)
    if total2 > CONTENT_W:
        scale2 = CONTENT_W / total2
        widths = [w * scale2 for w in widths]

    return widths


def _cell_is_bold(cell_text: str) -> bool:
    """Check if the original cell text was bold markdown."""
    return cell_text.strip().startswith("**") and cell_text.strip().endswith("**")


def render_table(pdf: ReportPDF, rows: List[List[str]]) -> None:
    """Render a markdown table with auto-sized columns, zebra striping, and styling."""
    if not rows or not rows[0]:
        return

    n_cols = len(rows[0])
    # Normalize row lengths
    for row in rows:
        while len(row) < n_cols:
            row.append("")

    col_widths = _measure_col_widths(pdf, rows)
    table_w = sum(col_widths)
    x_offset = MARGIN_L + (CONTENT_W - table_w) / 2  # center table

    row_h_header = 7
    row_h_body = 6
    font_header = 7.5
    font_body = 7.0

    # Check if table fits, else page break
    approx_height = row_h_header + row_h_body * (len(rows) - 1)
    if pdf.get_y() + approx_height > 270:
        pdf.add_page()

    # --- Header ---
    y = pdf.get_y()
    pdf.set_font(pdf._bf, "B", font_header)
    pdf.set_fill_color(*TBL_HEADER_BG)
    pdf.set_text_color(*TBL_HEADER_FG)
    pdf.set_draw_color(*TBL_HEADER_BG)

    for j, cell in enumerate(rows[0]):
        pdf.set_xy(x_offset + sum(col_widths[:j]), y)
        pdf.cell(col_widths[j], row_h_header, clean(cell),
                 border=1, fill=True, align="C")

    y += row_h_header

    # --- Body rows ---
    pdf.set_draw_color(*TBL_BORDER)
    for r_idx, row in enumerate(rows[1:], start=1):
        if y + row_h_body > 275:
            pdf.add_page()
            y = pdf.get_y()
            # Repeat header on new page
            pdf.set_font(pdf._bf, "B", font_header)
            pdf.set_fill_color(*TBL_HEADER_BG)
            pdf.set_text_color(*TBL_HEADER_FG)
            pdf.set_draw_color(*TBL_HEADER_BG)
            for j, cell in enumerate(rows[0]):
                pdf.set_xy(x_offset + sum(col_widths[:j]), y)
                pdf.cell(col_widths[j], row_h_header, clean(cell),
                         border=1, fill=True, align="C")
            y += row_h_header
            pdf.set_draw_color(*TBL_BORDER)

        # Zebra stripe
        if r_idx % 2 == 0:
            pdf.set_fill_color(*TBL_ROW_EVEN)
        else:
            pdf.set_fill_color(*TBL_ROW_ODD)

        for j, cell in enumerate(row):
            is_bold = _cell_is_bold(cell)
            cleaned = clean(cell)

            if is_bold:
                pdf.set_font(pdf._bf, "B", font_body)
                pdf.set_text_color(*TBL_ACCENT_FG)
            else:
                pdf.set_font(pdf._bf, "", font_body)
                pdf.set_text_color(40, 40, 40)

            # First column left-aligned, rest center-aligned
            align = "L" if j == 0 else "C"

            pdf.set_xy(x_offset + sum(col_widths[:j]), y)
            # Add left padding for first column
            if j == 0:
                pdf.cell(col_widths[j], row_h_body, "  " + cleaned,
                         border=1, fill=True, align=align)
            else:
                pdf.cell(col_widths[j], row_h_body, cleaned,
                         border=1, fill=True, align=align)

        y += row_h_body

    # Bottom border emphasis
    pdf.set_draw_color(*TBL_HEADER_BG)
    pdf.line(x_offset, y, x_offset + table_w, y)

    pdf.set_xy(MARGIN_L, y + 3)
    pdf.set_text_color(30, 30, 30)
    pdf.set_draw_color(0, 0, 0)


def build_pdf():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    lines = MD_FILE.read_text(encoding="utf-8").splitlines()

    in_code = False
    code_buf: List[str] = []
    tbl_buf: List[List[str]] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        pdf.set_x(MARGIN_L)

        # --- Code blocks ---
        if line.strip().startswith("```"):
            if in_code:
                # Render code block with background
                if pdf.get_y() + len(code_buf) * 3.5 > 270:
                    pdf.add_page()
                pdf.set_font(pdf._mf, "", 6.5)
                pdf.set_fill_color(240, 242, 245)
                pdf.set_draw_color(200, 210, 225)
                y_start = pdf.get_y()
                for cl in code_buf:
                    if pdf.get_y() > 270:
                        pdf.add_page()
                    pdf.set_x(MARGIN_L + 2)
                    pdf.set_text_color(50, 50, 50)
                    pdf.cell(CONTENT_W - 4, 3.5, cl, new_x="LMARGIN",
                             new_y="NEXT", fill=True)
                # Draw border around code block
                y_end = pdf.get_y()
                pdf.set_draw_color(200, 210, 225)
                pdf.rect(MARGIN_L, y_start - 0.5,
                         CONTENT_W, y_end - y_start + 1)
                pdf.ln(2)
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # --- Tables ---
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Skip separator rows
            if all(re.match(r"^[-:]+$", c) for c in cells):
                i += 1
                continue
            tbl_buf.append(cells)
            # Accumulate consecutive table rows
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1
                continue
            else:
                render_table(pdf, tbl_buf)
                tbl_buf = []
                i += 1
                continue

        # --- Images ---
        img_m = re.match(r"!\[.*?\]\((.+?)\)", line.strip())
        if img_m:
            img_path = ROOT / img_m.group(1).replace("/", os.sep)
            if img_path.exists():
                if pdf.get_y() > 150:
                    pdf.add_page()
                pdf.image(str(img_path), x=MARGIN_L, w=CONTENT_W)
                pdf.ln(4)
            i += 1
            continue

        # --- Headings ---
        if line.startswith("# ") and not line.startswith("## "):
            pdf.add_page()
            pdf.set_font(pdf._bf, "B", 22)
            pdf.set_text_color(25, 50, 100)
            pdf.multi_cell(0, 12, clean(line[2:]))
            # Underline
            pdf.set_draw_color(37, 60, 110)
            pdf.set_line_width(0.6)
            pdf.line(MARGIN_L, pdf.get_y() + 1,
                     MARGIN_L + 80, pdf.get_y() + 1)
            pdf.set_line_width(0.2)
            pdf.ln(5)
            i += 1
            continue

        if line.startswith("## "):
            pdf.ln(4)
            if pdf.get_y() > 250:
                pdf.add_page()
            pdf.set_font(pdf._bf, "B", 14)
            pdf.set_text_color(37, 60, 110)
            pdf.multi_cell(0, 8, clean(line[3:]))
            # Subtle line
            pdf.set_draw_color(200, 210, 225)
            pdf.line(MARGIN_L, pdf.get_y() + 1,
                     MARGIN_L + CONTENT_W, pdf.get_y() + 1)
            pdf.ln(3)
            i += 1
            continue

        if line.startswith("### "):
            pdf.ln(2)
            if pdf.get_y() > 260:
                pdf.add_page()
            pdf.set_font(pdf._bf, "B", 11)
            pdf.set_text_color(50, 60, 80)
            pdf.multi_cell(0, 7, clean(line[4:]))
            pdf.ln(2)
            i += 1
            continue

        if line.startswith("#### "):
            pdf.ln(1)
            pdf.set_font(pdf._bf, "B", 9.5)
            pdf.set_text_color(60, 70, 90)
            pdf.multi_cell(0, 6, clean(line[5:]))
            pdf.ln(1)
            i += 1
            continue

        # --- Horizontal rule ---
        if line.strip() == "---":
            pdf.ln(2)
            pdf.set_draw_color(200, 210, 225)
            pdf.line(MARGIN_L + 20, pdf.get_y(),
                     PAGE_W - MARGIN_R - 20, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # --- Bullet points ---
        if line.strip().startswith("- "):
            pdf.set_font(pdf._bf, "", 9)
            pdf.set_text_color(30, 30, 30)
            if pdf.get_y() > 275:
                pdf.add_page()
            indent = 6
            bullet_text = clean(line.strip()[2:])
            pdf.set_x(MARGIN_L + indent)
            pdf.cell(4, 5, "\u2022", new_x="END")
            pdf.multi_cell(CONTENT_W - indent - 4, 5, bullet_text)
            i += 1
            continue

        # --- Numbered lists ---
        nm = re.match(r"^(\d+)\.\s+(.+)", line.strip())
        if nm:
            pdf.set_font(pdf._bf, "", 9)
            pdf.set_text_color(30, 30, 30)
            if pdf.get_y() > 275:
                pdf.add_page()
            indent = 6
            pdf.set_x(MARGIN_L + indent)
            pdf.cell(6, 5, f"{nm.group(1)}.", new_x="END")
            pdf.multi_cell(CONTENT_W - indent - 6, 5, clean(nm.group(2)))
            i += 1
            continue

        # --- Normal text ---
        stripped = line.strip()
        if stripped:
            pdf.set_font(pdf._bf, "", 9)
            pdf.set_text_color(30, 30, 30)
            if pdf.get_y() > 275:
                pdf.add_page()
            pdf.multi_cell(0, 5, clean(stripped))
        else:
            pdf.ln(2)

        i += 1

    pdf.output(str(OUT_PDF))
    print(f"PDF saved: {OUT_PDF}")
    print(f"Pages: {pdf.page_no()}")
    sz = OUT_PDF.stat().st_size / 1024
    print(f"Size: {sz:.0f} KB")


if __name__ == "__main__":
    build_pdf()
