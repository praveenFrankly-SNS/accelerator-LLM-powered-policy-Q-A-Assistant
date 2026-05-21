#!/usr/bin/env python3
"""
Convert Databricks .py notebooks to HTML for GitHub Pages publishing.
Parses # MAGIC %md cells as markdown and code cells with syntax highlighting.
"""

import os
import re
import html
import glob
import markdown


def parse_databricks_notebook(filepath):
    """Parse a Databricks .py notebook into a list of typed cells."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"# COMMAND ----------", content)
    cells = []

    for section in sections:
        if not section.strip():
            continue

        if "# MAGIC %md" in section:
            lines = section.split("\n")
            md_lines = []
            for line in lines:
                if line.startswith("# MAGIC %md"):
                    md_lines.append(line[11:].strip())
                elif line.startswith("# MAGIC "):
                    md_lines.append(line[8:])
                elif line.startswith("# MAGIC"):
                    md_lines.append(line[7:])
            cells.append({"type": "markdown", "content": "\n".join(md_lines)})
        else:
            lines = section.split("\n")
            code_lines = [l for l in lines if not l.startswith("# DBTITLE")]
            code_content = "\n".join(code_lines).strip()
            if code_content:
                cells.append({"type": "code", "content": code_content})

    return cells


def convert_to_html(filepath, output_dir="site"):
    """Convert a single Databricks .py notebook to an HTML file."""
    filename          = os.path.basename(filepath)
    name_without_ext  = os.path.splitext(filename)[0]
    cells             = parse_databricks_notebook(filepath)
    html_cells        = []

    for cell in cells:
        if cell["type"] == "markdown":
            md_html = markdown.markdown(
                cell["content"], extensions=["fenced_code", "tables"]
            )
            html_cells.append(f"""
            <div class="cell text-cell">
                <div class="text-cell-render">{md_html}</div>
            </div>""")
        else:
            escaped = html.escape(cell["content"])
            html_cells.append(f"""
            <div class="cell code-cell">
                <div class="input-area">
                    <pre><code class="language-python">{escaped}</code></pre>
                </div>
            </div>""")

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{name_without_ext}</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css" rel="stylesheet"/>
    <style>
        body {{ font-family: "Helvetica Neue", Arial, sans-serif; font-size: 13px;
               line-height: 1.5; color: #1F2937; background: #fff; margin: 0; padding: 0; }}
        .container {{ width: 100%; padding: 15px; }}
        .cell {{ margin-bottom: 16px; }}
        .text-cell-render {{ padding: 6px 0; }}
        .text-cell-render h1 {{ font-size: 1.8em; color: #1F2937; }}
        .text-cell-render h2 {{ font-size: 1.4em; color: #1F2937; margin-top: 24px; }}
        .text-cell-render h3 {{ font-size: 1.2em; color: #374151; }}
        .text-cell-render code {{ background: #f3f4f6; padding: 2px 5px;
                                  border-radius: 3px; color: #e91e63; }}
        .text-cell-render pre {{ background: #f3f4f6; padding: 12px;
                                 border-radius: 6px; overflow-x: auto; }}
        .code-cell {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 4px; }}
        .input-area pre {{ margin: 0; padding: 10px;
                           font-family: "Monaco", "Consolas", monospace; font-size: 11px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/plugins/autoloader/prism-autoloader.min.js"></script>
</head>
<body>
    <div class="container">
        {"".join(html_cells)}
    </div>
</body>
</html>"""

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name_without_ext}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    return name_without_ext


if __name__ == "__main__":
    for py_file in sorted(glob.glob("notebooks/*.py")):
        name = convert_to_html(py_file)
        print(f"Converted {py_file} → site/{name}.html")
