from matplotlib import pyplot as plt
import sys
from pathlib import Path
import json

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE, family='sans-serif')  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE - 1)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

MIN_FAC_TWO = None
MIN_FAC = None

plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{xfrac}")

files = sys.argv[1:]

for file in files:
    with open(file) as f:
        data = json.load(f)

    general = data['general']

    proj, datapoints = next(iter(general.items()))
    # datapoints = data['detailed']['Chart']['10']
    times = [v['n_total'] for v in datapoints]
    n_comp_success = [v['n_comp_success'] for v in datapoints]
    plt.plot(times, n_comp_success, label=file)

plt.xlabel('\\textbf{Number of unique patches}')
plt.ylabel('\\textbf{Number of compilable patches}')
plt.legend()
plt.savefig('plot.png')
