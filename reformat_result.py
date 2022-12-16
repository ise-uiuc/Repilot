import json
import sys

with open(sys.argv[1]) as f:
    data: dict = json.load(f)


analysis: dict = {
    'general': {
        'n_fixed_bug (plausible)': 0,
        'n_total_bug': 0,
        'n_plausible': 0,
        'n_parse_success': 0,
        'n_comp_success': 0,
        'val_time': 0,
        'n_total': 0,
    },
    'details': {},
}


general = analysis['general']
details = analysis['details']
for bug_id, result in data.items():
    # if bug_id == 'Chart-11':
    #     continue
    general['n_total_bug'] += 1
    details[bug_id] = {}

    succeeded = len(result['succeeded'])
    parse_failed = len(result['parse_failed']) if 'parse_failed' in result else 0
    comp_failed = len(result['comp_failed'])
    test_failed = len(result['test_failed'])
    details[bug_id]['n_plausible'] = succeeded
    general['n_plausible'] += succeeded
    general['n_fixed_bug (plausible)'] += 1 if succeeded > 0 else 0

    if 'times' in result:
        details[bug_id]['val_time'] = sum(result['times'].values())
        general['val_time'] += details[bug_id]['val_time']

    details[bug_id]['n_parse_success'] = comp_failed + test_failed + succeeded
    details[bug_id]['n_comp_success'] = test_failed + succeeded

    general['n_parse_success'] += comp_failed + test_failed + succeeded
    general['n_comp_success'] += test_failed + succeeded

    details[bug_id]['n_total'] = succeeded + parse_failed + comp_failed + test_failed
    general['n_total'] += succeeded + parse_failed + comp_failed + test_failed
general['avg_val_time'] = general['val_time'] / general['n_total']

print(json.dumps(analysis, indent=2))

