"""Formats, brand palette and caption styles. Colors are RGB tuples."""
from dataclasses import dataclass, field

FORMATS = {"vertical": (1080, 1920), "horizontal": (1920, 1080), "square": (1080, 1080)}

# Taken from the Voyat site in this repository (index.html CSS variables).
BRAND = dict(navy=(0x1B, 0x21, 0x48), ink=(0x0B, 0x0F, 0x22), green=(0x7F, 0xD6, 0xA2),
             amber=(0xF0, 0xC0, 0x60), slate=(0x9A, 0xA0, 0xBD), white=(255, 255, 255))


@dataclass
class CaptionStyle:
    font: str = "Heebo-ExtraBold"
    size: float = 0.060                 # font size, fraction of canvas width
    fill: tuple = BRAND["white"]
    active: tuple = BRAND["green"]       # color of the word being spoken
    active_box: tuple = None            # color of a box behind the spoken word (None = no box)
    active_scale: float = 1.0
    upcoming_alpha: float = 1.0         # opacity of words not yet spoken
    stroke: float = 0.0                 # outline width, fraction of font size
    stroke_fill: tuple = BRAND["ink"]
    shadow: tuple = (0.0, 0.035, 0.07, 0.55)  # dx, dy, blur (fractions of font size), opacity
    pill: tuple = None                  # (rgb, opacity): rounded panel behind each line
    max_words: int = 4
    max_chars: int = 24
    max_width: float = 0.80             # line width limit, fraction of canvas width
    y: dict = field(default_factory=lambda: dict(vertical=0.70, horizontal=0.84, square=0.80))
    anim: str = "rise"                  # rise: fade + slide up | pop: scale with overshoot
    enter: float = 0.18
    exit: float = 0.12
    lead: float = 0.05                  # show a page this much before its first word
    hold: float = 0.35                  # keep a page this long after its last word (if nothing follows)
    gap_break: float = 0.45             # a pause longer than this always starts a new page
    soft_break: float = 0.18            # ...and this one does too once the page has 2+ words (phrase boundary)
    line_spacing: float = 1.15
    strip: str = ",.:;…\"'״׳"           # punctuation removed from caption words


STYLES = {
    # Direction "clean / minimal": calm phrases on a brand-navy panel, spoken word in brand green.
    "clean": CaptionStyle(pill=(BRAND["navy"], 0.78), shadow=None),
    # Direction "bold / energetic": 1-2 huge words that pop in, spoken word on a brand-amber box.
    "bold": CaptionStyle(font="Rubik-Black", size=0.092, fill=BRAND["white"], active=BRAND["ink"],
                         active_box=BRAND["amber"], active_scale=1.06, stroke=0.10, stroke_fill=BRAND["ink"],
                         shadow=(0.0, 0.06, 0.02, 0.85), max_words=2, max_chars=14, anim="pop", enter=0.22,
                         exit=0.0, hold=0.25),
}
