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
    CYAN,
    MAGENTA,
    BLACK,
    HTML_COLORS_8,
    HTML_COLORS_12,
    RGB_COLORS_8,
    RGB_COLORS_12,
    BGR_COLORS_8,
    BGR_COLORS_12,
    fractions,
    convert_colorspace,
)
from .figs import (
    show_image,
    color_labels,
    overlay_image,
    draw_text,
    draw_line,
    draw_arrow,
    draw_circle,
    draw_ellipse,
    draw_rectangle,
)
from .card import (
    find_card_corners,
    determine_new_corners,
    get_rectifier,
    corners_to_slice,
    HoughLine,
)
from .seeds import find_not_shadows, find_seed_mask
from .segment_isolated_seeds import find_seed_blobs_with_one_seed_rough
from .utils import window
