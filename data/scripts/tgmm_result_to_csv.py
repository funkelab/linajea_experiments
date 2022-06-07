from __future__ import print_function
import argparse
import os
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def tgmm_to_txt(tgmm_dir, xml, output_file, frames, scale=1):
    prev_cell_id_map = {}
    curr_cell_id = 0
    xml = xml.replace("*", "%03d")
    with open(output_file, 'w') as out:
        for t in range(*frames):
            curr_xml = xml % t
            xml_file = os.path.join(tgmm_dir, curr_xml)
            logger.info("Reading file %s" % xml_file)
            with open(xml_file) as f:
                tree = ET.parse(f)
                root = tree.getroot()
                cell_id_map = {}
                for cell in root.iter('GaussianMixtureModel'):
                    try:
                        pos = [float(p) for p in
                               cell.attrib['m'].strip().split(' ')]
                        logger.debug(pos)
                    except ValueError as e:
                        logger.warn(e)
                        continue
                    x, y, z = [round(p) for p in pos]
                    z = z * scale
                    local_cell_id = int(cell.attrib['id'])
                    cell_id_map[local_cell_id] = curr_cell_id
                    logger.debug("Local cell id: %d" % local_cell_id)
                    local_parent_id = int(cell.attrib['parent'])
                    parent_id = prev_cell_id_map[local_parent_id]\
                        if local_parent_id in prev_cell_id_map else -1
                    track_id = -1
                    to_write = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        t, z, y, x, curr_cell_id, parent_id, track_id)
                    logger.debug(to_write)
                    out.write(to_write)
                    curr_cell_id += 1
                prev_cell_id_map = cell_id_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgmm_dir', type=str,
                        help='Directory of TGMM xml files')
    parser.add_argument('--xml', type=str,
                        help="Template for filenames of tgmm xml files"
                        "(replace last 3 digits of time with a star)")
    parser.add_argument('--output', type=str,
                        help='output file')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument("--frames", type=int, nargs=2,
                        help="start and end frame", required=True)
    args = parser.parse_args()
    tgmm_dir = args.tgmm_dir
    xml = args.xml
    output_file = args.output
    scale = args.scale
    # only_daughters = args.daughters
    frames = args.frames
    tgmm_to_txt(tgmm_dir, xml, output_file, frames, scale=scale)
