import json
import sys
from pathlib import Path
from matplotlib import pyplot as plt

folder = Path(sys.argv[1])
all_files = folder.glob('**/*.json')
times = []
times_dict = {}
completion_times = []
for file in all_files:
    if 'time.json' == file.name:
        with open(file) as f:
            data = json.load(f)
        for i in range(1, len(data['times']) + 1):
            if i not in times_dict:
                times_dict[i] = []
            times_dict[i].extend(data['times'][:i])
        times.extend(data['times'])
        completion_times.extend(data['completion'])
print('Total Patch Gen:', sum(times))
time_points = [sum(times_dict[i]) / len(times_dict[i]) for i in sorted(list(times_dict.keys()))]
plt.plot(time_points)
plt.savefig('times.png')
print('Avg Patch Gen:', time_points)
if len(completion_times) > 0:
    print('Completion:', sum(completion_times) / len(completion_times))
