import argparse


def filter_division_tracks(division_tracks_file, parents_only_file):
    parent_child_dict = {}
    cell_dict = {}
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
            cell_id = int(tokens[4])
            cell_dict[cell_id] = tokens
            if parent_id != -1:
                if parent_id not in parent_child_dict:
                    parent_child_dict[parent_id] = []
                parent_child_dict[parent_id].append(cell)

    parents = []
    for parent, children in parent_child_dict.items():
        if len(children) == 2:
            parents.append(parent)
        elif len(children) > 2:
            print("Error! Cell has more than two children: %d" % len(children))
    print("Found %d divisions" % len(parents))
    with open(parents_only_file, 'w') as out:
        out.write(header)
        for parent_id in parents:
            loc_and_id = cell_dict[parent_id]
            out.write('\t'.join(loc_and_id))
            out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--div_tracks', type=str,
                        help="Path to division tracks file")
    parser.add_argument('--out', type=str, default=None,
                        help="File to write parents to")
    args = parser.parse_args()
    if args.out is None:
        args.out = args.div_tracks.split("div_state.txt")[0] + "parents.txt"
    filter_division_tracks(args.div_tracks, args.out)
