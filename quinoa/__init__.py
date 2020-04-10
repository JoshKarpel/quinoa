import os as _os

# see https://github.com/ContinuumIO/anaconda-issues/issues/905
_os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from .io import read_image, write_image
from .colors import (
    GREEN,
    YELLOW,
    RED,
    BLUE,
    WHITE,
    TEAL,
    PINK,
    BLACK,
    fractions,
    convert_colorspace,
)
from .figs import (
    show_image,
    draw_text,
    draw_arrow,
    draw_circle,
    draw_ellipse,
    draw_rectangle,
)
