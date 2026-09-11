import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from nag import figstyle


def test_widths_are_the_journal_widths():
    assert figstyle.WIDTH_DOUBLE == pytest.approx(183 / 25.4, abs=0.01)
    assert figstyle.WIDTH_SINGLE == pytest.approx(89 / 25.4, abs=0.01)


def test_palette_has_exactly_three_policy_colours():
    """A fourth policy colour is how the manuscript lost its visual language."""
    assert figstyle.POLICY_COLOURS == {
        "deterministic": figstyle.BLUE,
        "advisory": figstyle.SALMON,
        "enforced": figstyle.GREY,
    }


def test_apply_style_disables_the_grid():
    figstyle.apply_style()
    assert plt.rcParams["axes.grid"] is False


def test_apply_style_embeds_editable_text():
    figstyle.apply_style()
    assert plt.rcParams["pdf.fonttype"] == 42


def test_no_rcparam_font_size_is_below_five_point():
    """Figures are authored at final width, so these sizes are what print."""
    figstyle.apply_style()
    for key in ("font.size", "axes.labelsize", "xtick.labelsize",
                "ytick.labelsize", "legend.fontsize"):
        assert float(plt.rcParams[key]) >= 5.0, key


def test_panel_label_is_bold_lowercase_eight_point():
    figstyle.apply_style()
    fig, ax = plt.subplots()
    t = figstyle.panel_label(ax, "a")
    assert t.get_text() == "a"
    assert t.get_fontweight() == "bold"
    assert t.get_fontsize() == pytest.approx(8.0)
    plt.close(fig)


def test_panel_label_refuses_uppercase():
    figstyle.apply_style()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        figstyle.panel_label(ax, "A")
    plt.close(fig)
