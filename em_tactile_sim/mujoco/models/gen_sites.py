"""
Generate 49 <site> XML elements (7×7 @ 1mm) for visual overlay on sensor_pad.
Run this script once to produce em_sensor_flat_with_sites.xml.
"""
import os
import re


def generate_sites_xml(rows: int = 7, cols: int = 7,
                        cell_spacing: float = 1e-3,
                        z_offset: float = 0.0015) -> str:
    """Return XML string of <site> elements covering the sensing grid."""
    lines = []
    span_x = (cols - 1) * cell_spacing / 2.0
    span_y = (rows - 1) * cell_spacing / 2.0
    for i in range(rows):
        for j in range(cols):
            x = -span_x + j * cell_spacing
            y = -span_y + i * cell_spacing
            lines.append(
                f'      <site name="cell_{i}_{j}" '
                f'pos="{x:.6f} {y:.6f} {z_offset:.6f}" '
                f'size="0.0003" rgba="1 0.5 0 0.8"/>'
            )
    return "\n".join(lines)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "em_sensor_flat.xml")
    dst = os.path.join(here, "em_sensor_flat_with_sites.xml")

    with open(src) as f:
        content = f.read()

    rows, cols = 7, 7
    marker = "      <!-- 49 个 cell site（gen_sites.py 生成后插入此处，仅可视化） -->"
    sites = generate_sites_xml(rows=rows, cols=cols)
    if marker not in content:
        raise RuntimeError("Marker not found in XML — check em_sensor_flat.xml")

    new_content = content.replace(marker, sites)
    with open(dst, "w") as f:
        f.write(new_content)
    print(f"Written {rows*cols} sites → {dst}")
