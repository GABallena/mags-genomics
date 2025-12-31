#!/usr/bin/env python3
# Portfolio-safe copy: extract best-fit model from a text report (e.g., ModelFinder output)
# Identifiers generalized; no private data included.

import re

def parse_best_model(input_file, output_file):
    with open(input_file, "r") as infile:
        for line in infile:
            if line.startswith("Best-fit model:"):
                model = line.split(":")[1].strip()
                with open(output_file, "w") as outfile:
                    outfile.write(model)
                break

if __name__ == "__main__":
    import sys
    parse_best_model(sys.argv[1], sys.argv[2])
