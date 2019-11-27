# -*- coding: utf-8 -*-

import argparse
import yaml
import os

from datetime import datetime
from random import sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-random-search",
                        default="./base_random_search.yml",
                        help="Path to the yaml file with the base configuration.")
    parser.add_argument("--experiments",
                        default=10,
                        help="Number of experiment configurations to generate.",
                        type=int)
    parser.add_argument("--output",
                        default="./experiments",
                        help="Path to the directory to store the configurations.")

    args = parser.parse_args()

    with open(args.base_random_search, "r") as fh:
        base_experiment_grid = yaml.load(fh, Loader=yaml.SafeLoader)

    os.makedirs(args.output, exist_ok=True)

    experiment_base_name = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    for i in range(args.experiments):
        experiment_fname = os.path.join(
            args.output,
            "random-{}_{}.yml".format(experiment_base_name, i)
        )

        experiment_config = {}

        for param, param_grid in base_experiment_grid.items():
            experiment_config[param] = sample(param_grid, 1)[0]

        with open(experiment_fname, "w") as fh:
            yaml.dump(experiment_config, fh)
