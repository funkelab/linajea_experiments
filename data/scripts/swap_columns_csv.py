import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help="Tracks file to adjust")
    parser.add_argument('outfile', help="File to write adjusted tracks to")
    parser.add_argument('-i', help="First index to swap", type=int, default=1)
    parser.add_argument('-j', help="Second index to swap", type=int, default=3)
    args = parser.parse_args()
    f = args.infile
    out = args.outfile
    with open(f, 'r') as f_in:
        lines = f_in.readlines()
        lines = [l.strip().split('\t') for l in lines]
    with open(out, 'w') as f_out:
        for line in lines:
            temp = line[args.i]
            line[args.i] = line[args.j]
            line[args.j] = temp
            f_out.write('\t'.join(line) + '\n')
