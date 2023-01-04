import json
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

folder = Path(sys.argv[1])
all_files = folder.glob("**/*.json")
times = []
times_dict = {}
completion_times = []

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE, family="sans-serif")  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE - 1)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

MIN_FAC_TWO = None
MIN_FAC = None

plt.rc("text", usetex=True)
# TODO: fix `i` alignment
for file in all_files:
    if "time.json" == file.name:
        with open(file) as f:
            data = json.load(f)
        for i in range(1, len(data["times"]) + 1):
            if i not in times_dict:
                times_dict[i] = []
            times_dict[i].extend(data["times"][:i])
        times.extend(data["times"])
        completion_times.extend(data["completion"])
print("Total Patch Gen:", sum(times))
time_points = [
    sum(times_dict[i]) / len(times_dict[i]) for i in sorted(list(times_dict.keys()))
]
plt.xlabel("\\textbf{Number of samples}")
plt.ylabel("\\textbf{Average patch generation time (seconds)}")
plt.yticks(np.arange(0, np.ceil(max(time_points)), 0.5))
plt.plot(time_points, label=folder.name)
plt.legend()
plt.savefig("times.png")
print("Avg Patch Gen:", time_points)
print("Avg Patch Gen:", sum(times) / len(times))
if len(completion_times) > 0:
    print("Completion:", sum(completion_times) / len(completion_times))
