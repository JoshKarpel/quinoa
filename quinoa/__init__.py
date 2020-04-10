import os as _os
import logging as _logging

# see https://github.com/ContinuumIO/anaconda-issues/issues/905
_os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

_logger = _logging.getLogger(__name__)
_handler = _logging.NullHandler()
_handler.setLevel(_logging.DEBUG)
_logger.addHandler(_handler)
_logger.setLevel(_logging.DEBUG)

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
    BGR_COLORS,
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
