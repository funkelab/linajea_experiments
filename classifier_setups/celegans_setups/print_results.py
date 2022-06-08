import os
import sys
import toml

import numpy as np
from natsort import natsorted

metrics = []
params = []
only_th = None if len(sys.argv) == 2 else float(sys.argv[2])
if len(sys.argv) == 4:
    sampleT = sys.argv[3]
else:
    sampleT = None
for chkpt_dir in natsorted(os.listdir(sys.argv[1])):
    if "event" in chkpt_dir:
        continue
    chkpt = int(chkpt_dir)
    for th_dir in natsorted(os.listdir(os.path.join(sys.argv[1], chkpt_dir))):
        th = float(th_dir.split("prob_threshold_")[-1].replace("_", "."))
        if only_th is not None and th != 0.1:
            continue
        for kind in os.listdir(os.path.join(sys.argv[1], chkpt_dir, th_dir)):
            sample = sampleT
            if sample is None:
                if "lightsheet" in kind:
                    sample = "_".join(kind.split("_")[4:6])
                else:
                    sample = kind.split("_")[4]
            elif sample not in kind.split("_"):
                continue

            swa = "swa" in kind
            if "ttr" in kind:
                ttr = int(kind.split("ttr")[-1].split("_")[0])
            else:
                ttr = 1
            params.append((sample, chkpt, th,
                           "w_swa" if swa else "wo_swa",
                           "ttr {}".format(ttr)))
            with open(os.path.join(sys.argv[1], chkpt_dir, th_dir, kind), 'r') as f:
                results = toml.load(f)
            metric = results['mixed']['AP']
            metrics.append(metric)
            print("{:.7f} {}".format(metric, params[-1]))

if len(metrics) == 0:
    print("nothing found")
else:
    best_metrics = metrics[np.argmax(metrics)]
    best_params = params[np.argmax(metrics)]

    print("best: {} {}".format(best_metrics, best_params))
