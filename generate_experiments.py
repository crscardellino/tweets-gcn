# -*- coding: utf-8 -*-

import argparse
import json
import os

from datetime import datetime
from random import sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",
                        help="Path to the directory to store the configurations.")
    parser.add_argument("--experiments", "-e",
                        default=10,
                        help="Number of experiment configurations to generate",
                        type=int)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    PARAMETERS_GRID = {
        "activation": ["relu", "sigmoid", "tanh"],
        "dropout": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "edge_type": ["hashtags", "mentions", "toptfidf"] +
                     ["{}-grams".format(n) for n in range(2, 6)],
        "epochs": [50, 100, 250, 500],
        "filter_sizes": [[16], [32], [64], [16, 16], [32, 32], [64, 64]],
        "learning_rate": [0.1, 0.01, 0.001, 0.0001],
        "reg_parameter": [0.1, 0.01, 0.001, 0.0001, 0],
        "use_bias": [True, False],
        "weighted_edges": [True, False]
    }

    experiment_base_name = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    for i in range(args.experiments):
        experiment_fname = os.path.join(args.output, "{}_{}.json".format(experiment_base_name, i))

        experiment_config = {}

        for param, param_grid in PARAMETERS_GRID.items():
            experiment_config[param] = sample(param_grid, 1)[0]

        with open(experiment_fname, "w") as fh:
            json.dump(experiment_config, fh)
