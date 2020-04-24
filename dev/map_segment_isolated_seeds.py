import logging

from pathlib import Path
import  sys
import subprocess

from quinoa.segment_isolated_seeds import process

import htmap

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HERE = Path(__file__).parent.absolute()
DATA = HERE.parent / "data"
OUT = HERE / "out" / Path(__file__).stem

if __name__ == "__main__":
    docker_image, bucket = sys.argv[1:]

    image_names = [line.split()[-1] for line in subprocess.run(['mc', 'ls', bucket], capture_output = True, text = True).stdout.splitlines()]
    image_paths = [f's3://s3dev.chtc.wisc.edu/{bucket}/{im}' for im in image_names]

    htmap.settings['DOCKER.IMAGE'] = docker_image

    m = htmap.map(
        process,
        image_names,
        map_options = htmap.MapOptions(
            request_memory = '2GB',
            request_disk = '1GB',
            input_files = image_paths,
        )
    )

    print(m, len(m))
