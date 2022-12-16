import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
all_files = folder.glob('**/*.json')
times = []
completion_times = []
for file in all_files:
    if 'time.json' == file.name:
        with open(file) as f:
            data = json.load(f)
        times.extend(data['times'])
        completion_times.extend(data['completion'])
print('Total Patch Gen:', sum(times))
print('Avg Patch Gen:', sum(times) / len(times))
if len(completion_times) > 0:
    print('Completion:', sum(completion_times) / len(completion_times))
