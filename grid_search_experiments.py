# -*- coding: utf-8 -*-

import argparse
import yaml
import os
import sys
import warnings

from datetime import datetime
from functools import reduce
from itertools import product
from operator import itemgetter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-grid-search",
                        default="./base_grid_search.yml",
                        help="Path to the yaml file with the base configuration.")
    parser.add_argument("--output",
                        default="./experiments",
                        help="Path to the directory to store the configurations.")

    args = parser.parse_args()

    with open(args.base_grid_search, "r") as fh:
        base_experiment_grid = yaml.load(fh, Loader=yaml.SafeLoader)

    experiments_no = reduce(lambda x, y: x * y, map(len, base_experiment_grid.values()))
    if 100 < experiments_no < 500:
        warnings.warn("There are more than 100 possible experiments")
    elif experiments_no >= 500:
        print("There are more than 500 experiments", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    experiment_base_name = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    sorted_parameters = sorted(base_experiment_grid.keys())
    parameters_values = product(*(
        p[1] for p in sorted(base_experiment_grid.items(), key=itemgetter(0))
    ))

    for i, parameters_values in enumerate(parameters_values):
        experiment_fname = os.path.join(
            args.output,
            "{}_{}.yml".format(experiment_base_name, i)
        )

        experiment_config = dict(zip(sorted_parameters, parameters_values))

        with open(experiment_fname, "w") as fh:
            yaml.dump(experiment_config, fh)
