from __future__ import print_function
import argparse
import logging
import xml.etree.ElementTree as ET
from check_ground_truth import (get_edge_distance_stats,
                                check_valid_edges,
                                check_unattached_points)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def xml_to_track_graph(
        xml_file,
        output_file,
        scale=1.0,
        frame_limit=None,
        name=False,
        header=False,
        unattached=False):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    cells = {}
    edges = []

    for frame_with_nodes in root.iter('SpotsInFrame'):
        for node in frame_with_nodes.iter('Spot'):
            cell = {}
            _id = int(node.attrib['ID'])
            cell['id'] = _id
            if name:
                name = node.attrib['name'].split("_")[-1]
                cell['name'] = name
            t = int(float(node.attrib['POSITION_T']))
            if frame_limit is not None and t >= frame_limit:
                continue
            x = float(node.attrib['POSITION_X'])*scale
            y = float(node.attrib['POSITION_Y'])*scale
            z = float(node.attrib['POSITION_Z'])*scale
            if 'RADIUS' in node.attrib:
                cell['radius'] = node.attrib['RADIUS']
            #     r = node.attrib['RADIUS']
            # else:
            #     r = 1.0
            cell['position'] = (t, z, y, x)
            # cell['radius'] = r
            cells[_id] = cell

    for track in root.iter('Track'):
        track_id = track.attrib['TRACK_ID']
        for edge in track.iter('Edge'):
            edge_dict = {}
            target = int(edge.attrib['SPOT_SOURCE_ID'])
            source = int(edge.attrib['SPOT_TARGET_ID'])
            if target in cells and source in cells:
                edge_dict['source'] = source
                edge_dict['target'] = target
                edge_dict['track_id'] = track_id
                edges.append(edge_dict)

    print("Found %d cells and %d edges in XML file" % (len(cells), len(edges)))

    # print(cells)
    # print(edges)
    get_edge_distance_stats(cells, edges)
    check_valid_edges(cells, edges)
    unattached = check_unattached_points(cells, edges)
    if not unattached:
        print("removing unattached cells {}".format(unattached))
        for node in unattached:
            del cells[node]
    write_txt_file(cells, edges, output_file, header=header)


def write_txt_file(cells, edges, txt_file, header=False):
    with open(txt_file, 'w') as out:
        for edge in edges:
            u = edge['target']
            v = edge['source']
            assert u in cells, "Referenced cell %d not in file" % u
            assert v in cells, "Referenced cell %d not in file" % v

            frame_u = cells[u]['position'][0]
            frame_v = cells[v]['position'][0]

            # let u be the earlier cell
            if frame_u > frame_v:
                (u, v) = (v, u)
            elif frame_u == frame_v:
                raise RuntimeError(
                    "edge between %d and %d within frame %d" % (u, v, frame_u))

            # set parent of v to u
            if 'parent' in cells[v]:
                print(
                    "track does not form a tree in time, cell %d has "
                    "(at least) parents %d and %d" % (
                        v,
                        cells[v]['parent'],
                        u))
            cells[v]['parent'] = u
            cells[v]['track_id'] = edge['track_id']
            cells[u]['track_id'] = edge['track_id']

        fields = ["t", "z", "y", "x",
                  "cell_id", "parent_id", "track_id"]
        if "radius" in next(iter(cells.values())):
            fields.append("radius")
        if "name" in next(iter(cells.values())):
            fields.append("name")
        fmt = "{}" + "\t{}"*(len(fields)-1) + "\n"

        if header:
            out.write(fmt.format(*fields))

        for cell_id, cell in cells.items():
            entries = []
            t, z, y, x = cell['position']
            entries.extend([t, z, y, x])

            entries.append(cell_id)

            parent_id = cell['parent'] if 'parent' in cell else -1
            entries.append(parent_id)

            if 'track_id' not in cell:
                print(cell)
                track_id = -1
            else:
                track_id = cell['track_id']
            entries.append(track_id)

            if "radius" in cell:
                entries.append(cell["radius"])

            if "name" in cell:
                entries.append(cell['name'])
            out.write(fmt.format(*entries))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=str,
                        help='input MaMuT xml file')
    parser.add_argument('--output', type=str,
                        help='output file')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument("--frame_limit", type=int, help="")
    parser.add_argument("--name", action="store_true", help="")
    parser.add_argument("--header", action="store_true", help="")
    parser.add_argument("-u" "--unattached", action='store_true',
                        help="Include unattached nodes")
    args = parser.parse_args()
    xml_file = args.xml
    output_file = args.output
    scale = args.scale
    # only_daughters = args.daughters
    frame_limit = args.frame_limit
    print(args)
    xml_to_track_graph(
            xml_file, output_file, scale=scale,
            frame_limit=frame_limit, name=args.name, header=args.header,
            unattached=args.unattached)
