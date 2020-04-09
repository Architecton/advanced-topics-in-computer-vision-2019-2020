import argparse

# Script used to parse results for different datasets
# to evaluate effects of different parameter values.

# Parse argument value.
parser = argparse.ArgumentParser()
parser.add_argument("--param")
args = parser.parse_args()

# Open results file and accumulate failures.
with open('../results/results.txt', 'r') as f1:
    failure_cumsum = 0
    for line in f1.readlines():
        failure_cumsum += int(line[line.index(":")+2:].strip())

    # Write aggregated failures for parameter value.
    with open('../results/res_param.txt', 'a') as f2:
        f2.write("{0},{1}\n".format(args.param, failure_cumsum))
 
