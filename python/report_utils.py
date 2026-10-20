# report_utils.py
# Small helpers shared by the analysis script:
#   * show_table()  - prints a table neatly in the terminal AND saves it as a PNG image
#   * Report        - collects headings, notes, tables and figures, then writes one
#                     HTML results page and opens it in the browser automatically

import os
import base64
import html
import webbrowser
import pathlib
import matplotlib
matplotlib.use("Agg")          # draw to files only, no pop-up windows
import matplotlib.pyplot as plt

ACCENT = "#2a5fa8"


def _fmt(v):
    if isinstance(v, float):
        if v != 0 and (abs(v) < 0.001 or abs(v) >= 1e6):
            return f"{v:.3e}"
        return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.2f}"
    return str(v)


def table_png(df, title, path, accent=ACCENT):
    """Render a pandas DataFrame as a clean table image."""
    show_index = not (df.index.dtype == "int64" and list(df.index) == list(range(len(df))))
    cells = [[_fmt(v) for v in row] for row in df.values]
    cols = [str(c) for c in df.columns]
    if show_index:
        cells = [[str(i)] + r for i, r in zip(df.index, cells)]
        cols = [str(df.index.name or "")] + cols

    ncol, nrow = len(cols), len(cells)
    widths = [max(len(cols[j]), *(len(r[j]) for r in cells)) for j in range(ncol)]
    fig_w = min(max(sum(widths) * 0.11 + 0.6, 5), 13)
    fig_h = 0.42 * (nrow + 1) + 0.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                   colWidths=[w / sum(widths) for w in widths])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d6d4cc")
        if r == 0:
            cell.set_facecolor(accent)
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f3ef")
        if show_index and c == 0 and r > 0:
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=12, weight="bold", loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _trim(path)
    return path


def _trim(path, pad=12):
    """Cut away the empty white space matplotlib leaves around a table."""
    try:
        from PIL import Image, ImageChops
        im = Image.open(path).convert("RGB")
        box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
        if box:
            l, t, r, b = box
            im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height))).save(path)
    except ImportError:
        pass


def show_table(df, title, png_path=None, report=None, note=None):
    """Print in the terminal, save as PNG, and add to the HTML report."""
    df = df.copy()
    df.columns.name = None
    line = "=" * max(60, len(title) + 4)
    print(f"\n{line}\n  {title}\n{line}")
    print(df.to_string())
    if note:
        print(f"\n  -> {note}")
    if png_path:
        table_png(df, title, png_path)
    if report is not None:
        report.table(df, title, note)


class Report:
    def __init__(self, title, subtitle=""):
        self.title, self.subtitle, self.parts = title, subtitle, []

    def heading(self, text):
        self.parts.append(f"<h2>{html.escape(text)}</h2>")

    def text(self, text):
        self.parts.append(f"<p>{text}</p>")

    def callout(self, text):
        self.parts.append(f"<div class='callout'>{text}</div>")

    def pre(self, text, label=""):
        self.parts.append(f"<details><summary>{html.escape(label)}</summary><pre>{html.escape(text)}</pre></details>")

    def table(self, df, title, note=None):
        default_index = list(df.index) == list(range(len(df)))
        d = df if default_index else df.reset_index()
        t = d.to_html(classes="tbl", border=0, index=False, float_format=lambda v: _fmt(float(v)))
        n = f"<p class='note'>{note}</p>" if note else ""
        self.parts.append(f"<h3>{html.escape(title)}</h3>{t}{n}")

    def image(self, path, caption):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        self.parts.append(f"<figure><img src='data:image/png;base64,{b64}'/><figcaption>{html.escape(caption)}</figcaption></figure>")

    def save_and_open(self, path):
        page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(self.title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{{font-family:Georgia,serif;background:#fcfcfb;color:#1f1f1d;max-width:980px;margin:0 auto;padding:28px 18px 60px;line-height:1.55}}
 header{{border-bottom:3px solid {ACCENT};padding-bottom:12px;margin-bottom:10px}}
 h1{{font-family:Arial,sans-serif;margin:0;font-size:28px}} .sub{{color:#5b5a56;font-style:italic;margin-top:6px}}
 h2{{font-family:Arial,sans-serif;color:{ACCENT};margin-top:38px;border-bottom:1px solid #d6d4cc;padding-bottom:4px}}
 h3{{font-family:Arial,sans-serif;font-size:15px;margin:22px 0 6px}}
 .tbl{{border-collapse:collapse;font-family:Arial,sans-serif;font-size:13.5px;margin:4px 0 6px;min-width:50%}}
 .tbl th{{background:{ACCENT};color:#fff;padding:7px 12px;text-align:center}}
 .tbl td{{padding:6px 12px;text-align:center;border-bottom:1px solid #e4e2da}}
 .tbl tbody tr:nth-child(even){{background:#f4f3ef}} .tbl td:first-child{{font-weight:bold;text-align:left}}
 .note{{color:#5b5a56;font-size:14px;margin-top:4px}}
 .callout{{background:#f4f3ef;border-left:4px solid {ACCENT};padding:10px 14px;margin:12px 0}}
 figure{{margin:18px 0;text-align:center}} img{{max-width:min(100%,780px);border:1px solid #e4e2da}}
 figcaption{{color:#5b5a56;font-style:italic;font-size:14px;margin-top:6px}}
 pre{{background:#f6f5f1;padding:12px;overflow-x:auto;font-size:12px}} summary{{cursor:pointer;color:{ACCENT};font-family:Arial,sans-serif}}
</style></head><body><header><h1>{html.escape(self.title)}</h1><div class="sub">{html.escape(self.subtitle)}</div></header>
{''.join(self.parts)}</body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"\nFull results page saved to: {os.path.abspath(path)}")
        try:
            webbrowser.open(pathlib.Path(path).resolve().as_uri())
            print("Opening it in your browser...")
        except Exception:
            print("Open the file above in any browser to see all tables and graphs.")
