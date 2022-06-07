import argparse


def filter_division_tracks(division_tracks_file, daughters_only_file):
    parent_child_dict = {}
    header = ""
    with open(division_tracks_file, 'r') as fl:
        ln = fl.readline()
        if "parent" in ln and "track" in ln:
            header = ln
        else:
            fl.seek(0)

        for cell in fl:
            tokens = cell.strip().split('\t')
            parent_id = int(tokens[5])
            if parent_id != -1:
                if parent_id not in parent_child_dict:
                    parent_child_dict[parent_id] = []
                parent_child_dict[parent_id].append(cell)

    daughters = []
    for children in list(parent_child_dict.values()):
        if len(children) == 2:
            daughters += children
        elif len(children) > 2:
            print("Error! Cell has more than two children: %d" % len(children))
    print("Found %d divisions" % (len(daughters) / 2))
    with open(daughters_only_file, 'w') as out:
        out.write(header)
        for daughter in daughters:
            out.write(daughter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--div_tracks', type=str,
                        help="Path to division tracks file")
    parser.add_argument('--out', type=str,
                        help="File to write daughter cells to")
    args = parser.parse_args()
    filter_division_tracks(args.div_tracks, args.out)
