import sys


def load_error_counts(filename):
    error_counts = {}
    with open(filename) as f:
        for line in f.readlines():
            split = line.strip().split(':')
            error_counts[split[0]] = int(split[1])
    return error_counts


if __name__ == '__main__':
    old_file, new_file = sys.argv[1:3]
    old_error_counts = load_error_counts(old_file)
    new_error_counts = load_error_counts(new_file)

    for fname, new_error_count in new_error_counts.items():
        old_error_count = old_error_counts.get(fname, 0)
        if new_error_count > old_error_count:
            print("File %s has %d new errors (old: %d, new: %d)"
                  % (fname, new_error_count - old_error_count,
                     old_error_count, new_error_count))
