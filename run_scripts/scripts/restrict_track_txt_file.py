import os
import sys

in_fn = sys.argv[1]
out_fn = sys.argv[2]

lmt = int(sys.argv[3])
with open(out_fn, 'w') as out_fl:
    with open(in_fn, 'r') as in_fl:
        for ln in in_fl:
            nrs = ln.split()
            nrs = [int(n) for n in nrs]
            if nrs[1] > lmt:
                continue
            nrs[2] = min(lmt, nrs[2])
            out_fl.write("{} {} {} {}\n".format(*nrs))
