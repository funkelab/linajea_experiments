import argparse
import logging
import numpy as np
import csv
from sklearn import metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_prediction_from_file(filename):
    with open(filename) as f:
        reader = csv.reader(f, delimiter=' ')
        results = []
        for row in reader:
            results.append(row)

    results = np.array(results)
    true_labels = results[:, 0]
    predicted_labels = results[:, 1]
    return true_labels, predicted_labels


def read_predictions_from_mongo(config):
    # TODO: implement
    pass


def evaluate(true_labels, predicted_labels):
    # Constants
    key = {0: "division",
           1: "daughter",
           2: "continuation"}
    logger.info("Key: %s" % str(key))

    # Print the confusion matrix
    logger.info("Confusion Matrix: (pred across top, actual down size)")
    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)
    logger.info(confusion_matrix)

    logger.info("Classification Report:")
    # Print the precision and recall, among other metrics
    logger.info(metrics.classification_report(
        true_labels,
        predicted_labels,
        digits=3))

    balanced_acc = metrics.balanced_accuracy_score(
            true_labels, predicted_labels)
    logger.info("Balanced accuracy: %s" % balanced_acc)
    return confusion_matrix, balanced_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-file', type=str, required=True,
                        help='text file with true labels in first column'
                             'and predicted labels in second column')
    args = parser.parse_args()

    eval(args.pred_file)
