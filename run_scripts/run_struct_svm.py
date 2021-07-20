import os
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_MAX_THREADS"] = num_threads

import argparse
import structsvm as ssvm
import pylp
import logging
import numpy as np

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        help='path to files')
    parser.add_argument('--db_name', type=str,
                        help='db_name')
    parser.add_argument('--db_host', type=str,
                        default="mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin?replicaSet=rsLinajea",
                        help='db host')
    args = parser.parse_args()


    # simple ILP to chose exactly one indiciator variable

    features = np.loadtxt("features.txt").T
    logger.info("features %s", features.shape)
    num_variables = features.shape[1]
    num_features = features.shape[0]
    # ground_truth = np.zeros((num_variables,))
    # ground_truth[0] = 1
    best_effort = np.loadtxt("best_effort.txt", dtype=np.float32)
    # best_effort = np.expand_dims(best_effort, 1)
    logger.info("best effort %s", best_effort.shape)

    constraints = pylp.LinearConstraints()
    cnt_cnstr = 0
    with open("constraints.txt", 'r') as f:
        for ln in f:
            cnstr = pylp.LinearConstraint()
            terms = ln.split()
            for t in terms:
                if "=" in t or "<" in t or ">" in t:
                    if t == "=" or t == "==":
                        cnstr.set_relation(pylp.Relation.Equal)
                    elif t == "<=":
                        cnstr.set_relation(pylp.Relation.LessEqual)
                    elif t == ">=":
                        cnstr.set_relation(pylp.Relation.GreaterEqual)
                    else:
                        raise RuntimeError("invalid relation {}".format(t))
                elif "*" not in t:
                    cnstr.set_value(int(t))
                    assert t == terms[-1], "found value like term not at the end"
                else:
                    coeff, ind = t.split("*")
                    cnstr.set_coefficient(int(ind), int(coeff))
            cnt_cnstr += 1
            constraints.add(cnstr)
    logger.info("constraints %s", cnt_cnstr)

    logger.info("setting up")
    loss = ssvm.SoftMarginLoss(constraints, features, best_effort)
    bundle_method = ssvm.BundleMethod(
        loss.value_and_gradient,
        dims=num_features,
        regularizer_weight=0.001,
        eps=1e-6)

    logger.info("starting optimization")
    w = bundle_method.optimize(1000)
    logger.info("weights: %s", w)
    for wp in w:
        print(wp)
    # costs = features.T@w

    # costs for first variable should be minimal
    # for i in range(1, num_variables):
    #     assert costs[i] >= costs[0]
