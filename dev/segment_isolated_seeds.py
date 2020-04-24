import logging

from pathlib import Path

from tqdm import tqdm

from quinoa.segment_isolated_seeds import process

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HERE = Path(__file__).parent.absolute()
DATA = HERE.parent / "data"
OUT = HERE / "out" / Path(__file__).stem

if __name__ == "__main__":
    image_paths = [
        path for path in (DATA / "aus").iterdir() if path.suffix.lower() == ".jpg"
    ]
    # image_paths = [DATA / "aus" / "104.JPG"]

    for path in tqdm(image_paths):
        process(path, OUT)
