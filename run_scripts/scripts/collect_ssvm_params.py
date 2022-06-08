import os
import glob
import logging
import sys
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=10)


def load_ssvm_params(struct_svm_dir, ssvm_type=None):
    logger.debug("running load ssvm, dir: %s", struct_svm_dir)
    params = {}
    if ssvm_type is None or ssvm_type == "default":
        with open(os.path.join(struct_svm_dir, "ssvm.txt"), 'r') as f:
            output = list(f)
        for line in output[::-1]:
            if "ε" in line and "INFO" in line and "is:" in line:
                eps = float(line.split()[-1])
                if abs(eps) > 1000:
                    logger.warning("unable to compute valid ssvm params")
                    return False
                else:
                    logger.info("found valid ssvm params in dir %s", struct_svm_dir)
                break
    else:
        fls = glob.glob(os.path.join(struct_svm_dir, "logs/*.out"))
        print(fls, ssvm_type)
        for fl in fls:
            with open(fl, 'r') as f:
                ln = next(f)
                print(fl, ln)
                if ln[:-1].split(" ")[-1] != ssvm_type:
                    continue
                output = list(f)
                for idx, ln in enumerate(output):
                    if "Subject: Job" in ln:
                        break
                if "Done" not in output[idx]:
                    continue
                logger.info("found valid ssvm params in file %s", os.path.join(struct_svm_dir, fl))
                output = output[idx-12:idx-3]
                break
        else:
            raise RuntimeError("unable to compute valid ssvm params")
            return False

    params["weight_node_score"] =   float(output[-9])
    params["selection_constant"] =  float(output[-8])
    params["track_cost"] =          float(output[-7])
    # params["disappear_cost"] =      float(line[-6]) # = 0.0/not used
    params["weight_division"] =     float(output[-5])
    params["division_constant"] =   float(output[-4])
    params["weight_child"] =        float(output[-3])
    params["weight_continuation"] = float(output[-2])
    params["weight_edge_score"] =   float(output[-1])

    logger.debug("ssvm params: %s", params)
    return params


if __name__ == "__main__":
    ind = sys.argv[1]
    ssvm_type = None if len(sys.argv) <= 2 else sys.argv[2]

    params = {}
    for exp in glob.glob(ind + "*"):
        print(exp)
        for sample in glob.glob(os.path.join(exp, "ssvm_*")):
            p = load_ssvm_params(sample, ssvm_type=ssvm_type)
            print(sample, p)
            params[sample] = p

    # print(params)
    p_header = True
    for k, v in params.items():
        if p_header:
            print(list(v.keys()))
            p_header = False
        print(list(v.values()))
