import sys
import os


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python cleanup_snapshots.py <setup> <skip_by>\n"
              "Filters snapshots to those with number 1 + n*<skip_by>")
        sys.exit(1)

    setup = sys.argv[1]
    skip_by = int(sys.argv[2])

    snapshot_dir = './' + setup + '/snapshots/'
    i = 0
    c = 0
    for snapshot_name in os.listdir(snapshot_dir):
        print(snapshot_name)
        end_index = -1 * len('.hdf')
        if "snapshot" in snapshot_name:
            start_index = len('snapshot_')
            snapshot_num = int(snapshot_name[start_index:end_index])
        elif "batch" in snapshot_name:
            start_index = len('batch_')
            snapshot_num = int(snapshot_name[start_index:end_index])
        else:
            print("unknown snapshot prefix, trying num extraction")
            snapshot_num = int(snapshot_name.split("_")[-1].split(".")[0])
        if (snapshot_num - 1) % skip_by == 0:
            # print("Keeping snapshot %d" % snapshot_num)
            pass
        else:
            print("Trashing snapshot %d" % snapshot_num)
            os.remove(snapshot_dir + snapshot_name)
            c += 1

    print("removed %d snapshots" % c)
