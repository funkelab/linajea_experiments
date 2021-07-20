import sys
import logging
import queue as q

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TrackEditor:

    def __init__(self, track_file):
        self.track_file = track_file
        with open(track_file, 'r') as f:
            lines = f.readlines()
        lines = list(map(lambda x: x.strip().split(), lines))

        self.entries = {}
        self.id_to_index = {}
        self.parent_id_to_index = {}

        # t, z, y, x, cell_id, parent_id, track_id
        for index, line in enumerate(lines):
            _id = line[4]
            parent_id = line[5]
            assert _id not in self.id_to_index,\
                "Duplicate ID found: %s already in dictionary" % _id
            self.entries[index] = line
            self.id_to_index[_id] = index
            self.parent_id_to_index.setdefault(parent_id, [])
            self.parent_id_to_index[parent_id].append(index)

    def delete_after(self, _id):
        """Will remove the point with _id and all following points in the track
        from the internal tracks representation. This includes any branches.
        """
        line = self.entries.pop(self.id_to_index.pop(_id))
        logger.info("Deleting %s and all points in track after" % line)

        parent_ids = q.Queue()
        parent_ids.put(_id)

        while not parent_ids.empty():
            parent = parent_ids.get()
            children = self.parent_id_to_index.pop(parent, [])
            if len(children) == 2:
                logger.debug("Found two children")
            for child in children:
                logger.debug("Deleting %s" % self.entries[child])
                _id = self.entries[child][4]
                parent_ids.put(_id)
                del self.entries[child]
                del self.id_to_index[_id]

    def delete_before(self, _id):
        """Will remove the point with _id and all preceeding points in the track
        from the internal tracks representation. This does not trace any
        sibling branches, and it may leave sibling branches pointing
        to non-existant parents.
        """
        line = self.entries.get(self.id_to_index[_id])
        logger.info("Deleting %s and all points in track before" % line)
        for child_index in self.parent_id_to_index[_id]:
            self.entries[child_index][5] = '-1'

        while _id != '-1':
            del self.parent_id_to_index[_id]
            index = self.id_to_index.pop(_id)
            entry = self.entries.pop(index)
            logger.debug("Deleting %s" % entry)
            _id = entry[5]

    def merge_tracks(self, parent, child):
        """Will set the parent of child to parent and change all the track
        ids for child and all following connected cells to parent's track_id
        """
        assert parent in self.id_to_index,\
            "Parent id %s not found" % parent
        assert child in self.id_to_index, "Child id %s not found" % child
        logger.info("Changing parent of %s to %s and merging tracks"
                    % (child, parent))
        track_id = self.entries[self.id_to_index[parent]][6]
        child_entry = self.entries[self.id_to_index[child]]
        self.parent_id_to_index[child_entry[5]].remove(self.id_to_index[child])
        self.parent_id_to_index.setdefault(parent, [])
        self.parent_id_to_index[parent].append(self.id_to_index[child])
        child_entry[5] = parent
        child_entry[6] = track_id
        self.entries[self.id_to_index[child]] = child_entry

        parent_ids = q.Queue()
        parent_ids.put(child)

        while not parent_ids.empty():
            parent = parent_ids.get()
            children = self.parent_id_to_index.get(parent, [])
            if len(children) == 2:
                logger.debug("Found two children")
            for child in children:
                logger.debug("Changing track id of %s to %s"
                             % (self.entries[child], track_id))
                _id = self.entries[child][4]
                parent_ids.put(_id)
                self.entries[child][6] = track_id

    def write(self, outfile):
        with open(outfile, 'w') as f:
            for i in range(max(self.entries.keys())):
                if i in self.entries:
                    f.write('\t'.join(self.entries[i]) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("args: <input track file> <output track file>")
        sys.exit()

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    te = TrackEditor(input_file)

    # put beginning of track 4 on track 5 and delete end
    te.merge_tracks('18815', '231924')
    te.delete_after('18814')

    # delete track 18 and end of one branch of track 5
    te.delete_after('18719')
    te.delete_after('87916')

    # delete track 51
    te.delete_after('255038')

    # put beginning and end of 12 onto 27 and delete middle
    te.merge_tracks('255089', '24383')
    te.merge_tracks('24429', '255209')
    te.delete_after('255088')

    # Put end of track 100 onto 19 and delete beginning
    te.merge_tracks('251608', '24066')
    te.delete_before('24065')

    te.write(output_file)
