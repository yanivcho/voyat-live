"""The edit of one video. Anchors: 12 = word #12 in the words table, "שלום" = that word, 1.5 = source seconds."""
from vedit import *  # noqa: F401,F403


def build(p):
    p.set_format("vertical")   # vertical | horizontal | square ; fit="contain" keeps the whole frame (blurred fill)
    p.set_style("clean")       # clean | bold  (see vedit/styles.py)
    p.cut_silences()           # min_silence=0.40, keep_gap=0.20, xfade=0.12
    p.captions()
