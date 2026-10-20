"""The documentation chart renderer uses only the standard library.

Parakeet's text helpers are deliberately not imported: their module requires
onnxruntime at import time. These checks render strings without writing files.
"""

from xml.etree import ElementTree

import pytest

from docs.make_charts import CHARTS, THEMES, svg


@pytest.mark.parametrize("chart_name", sorted(CHARTS))
@pytest.mark.parametrize("theme_name", sorted(THEMES))
def test_chart_svg_contains_accessible_title_and_all_rows(chart_name, theme_name):
    chart = CHARTS[chart_name]
    root = ElementTree.fromstring(svg(chart, THEMES[theme_name]))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib["role"] == "img"
    assert root.attrib["aria-label"] == chart["title"]
    labels = [node.text for node in root.findall("svg:text", namespace)]
    assert chart["title"] in labels
    assert all(label in labels for label, _, _ in chart["rows"])
    assert len(root.findall("svg:path", namespace)) == len(chart["rows"])
