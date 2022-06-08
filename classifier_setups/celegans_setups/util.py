from __future__ import print_function
import logging
try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print(e)

import gunpowder as gp

from linajea import (parse_tracks_file)

logger = logging.getLogger(__name__)


def parse_tracks_file_by_class(filename, num_classes=None,
                               scale=1.0, limit_to_roi=None):
    locations, track_info = parse_tracks_file(
        filename, scale=scale,
        limit_to_roi=limit_to_roi)

    locationsByClass = [[] for _ in range(num_classes)]

    labelsByClass = [[] for _ in range(num_classes)]
    for idx, cell in enumerate(track_info):
        if cell[-1] == 0:
            locationsByClass[0].append(locations[idx])
            labelsByClass[0].append(cell[-1])
        elif cell[-1] == 1:
            locationsByClass[1].append(locations[idx])
            labelsByClass[1].append(cell[-1])
        elif cell[-1] == 2:
            locationsByClass[2].append(locations[idx])
            labelsByClass[2].append(cell[-1])
        else:
            raise RuntimeError("invalid class {} ({})".format(
                cell[-1], cell))
    logger.info("%d normal cells, %d mother cells, %d daughter cells",
                len(locationsByClass[0]), len(locationsByClass[1]),
                len(locationsByClass[2]))

    return locations, track_info, locationsByClass, labelsByClass
