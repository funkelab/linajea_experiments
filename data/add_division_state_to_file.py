import os
import sys


def parse_division_tracks(tracks_file):
    parent_child_dict = {}
    cells_dict = {}
    header = ""
    with open(tracks_file, 'r') as fl:
        ln = fl.readline()
        if "parent" in ln and "track" in ln:
            header = ln[:-1] + "\tdiv_state\n"
        else:
            fl.seek(0)

        for cell in fl:
            tokens = cell.strip().split('\t')
            parent_id = int(tokens[5])
            cell_id = int(tokens[4])
            cells_dict[cell_id] = cell
            if parent_id == -1:
                continue
            if parent_id not in parent_child_dict:
                parent_child_dict[parent_id] = []
            parent_child_dict[parent_id].append(cell)

    fn = os.path.splitext(tracks_file)[0] + "_div_state.txt"
    with open(fn, 'w') as out:
        out.write(header)
        for cell_id, cell in cells_dict.items():
            tokens = cell.strip().split('\t')
            parent_id = int(tokens[5])
            #print("Cell: %d" % cell_id)
            #print("Parent: %d" % parent_id)
            if parent_id == -1:
                siblings = [cell]
            else:
                siblings = parent_child_dict[parent_id]
            #print("Siblings: %s" % siblings)
            try:
                children = parent_child_dict[cell_id]
                #print("Chidlren: %s" % children)
            except:
                children = []
            assert len(children) <= 2, \
                "Error! Cell has more than two children: %d" % cell_id
            assert len(siblings) <= 2, \
                "Error! Cell has more than two siblings: %d" % parent_id
            assert len(siblings) > 0, \
                "Error! Cell has no children but should have one: %d" % parent_id
            assert not(parent_id != -1 and len(siblings) == 2 and len(children) == 2), \
                "Error! No timepoint between to successive divisions: %d %d" % (parent_id, cell_id)

            if len(children) == 2:
                out.write(cell.strip() + "\t" + "1" + "\n")
            elif parent_id == -1:
                out.write(cell.strip() + "\t" + "0" + "\n")
            elif len(siblings) == 2:
                out.write(cell.strip() + "\t" + "2" + "\n")
            else: #if len(siblings) == 1 and  len(children) == 1:
                out.write(cell.strip() + "\t" + "0" + "\n")


if __name__ == '__main__':
    assert len(sys.argv) == 2, "Usage: python add_division_state_to_file.py"\
        " <tracks_file>"
    tracks_file = sys.argv[1]
    parse_division_tracks(tracks_file)
