from matplotlib import pyplot as plt
import sys
from pathlib import Path
import json

files = sys.argv[1:]

for file in files:
    with open(file) as f:
        data = json.load(f)

    general = data['general']

    proj, datapoints = next(iter(general.items()))
    times = [v['time_consumed'] for v in datapoints]
    n_comp_success = [v['n_comp_success'] for v in datapoints]
    plt.plot(times, n_comp_success, label=file)
plt.legend()
plt.savefig('plot.png')
