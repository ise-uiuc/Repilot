import argparse
import time
import openai
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.join(sys.path[0], '../../'))) # Hack
sys.path.append(os.path.dirname(os.path.join(sys.path[0], '../../Dataset/')))

from Dataset.parse_quixbugs import parse_python, get_unified_diff
from Dataset.parse_d4j import clean_parse_d4j, clean_parse_d4j_single_hunk
from Dataset.parse_manybugs import clean_parse_manybugs
from Dataset.parse_codeflaws import clean_parse_codeflaws
from Dataset.utils import filter_single_line
from Repair.prompt import BASE_PROMPT, VARY_BASE_PROMPT, LOCATION_PROMPT, LONG_BASE_PROMPT, JAVA_BASE_PROMPT, JAVA_VARY_PROMPT, JAVA_LONG_VARY_PROMPT
from Repair.prompt import C_VARY_PROMPT, C_BASE_PROMPT
from Repair.util import pick_smallest_example_fix, _run_validation
from api_request import request_engine, create_openai_config, create_openai_config_suffix

API_KEY_FILE = 'api_key.txt'  # read private api key
openai.api_key = open(API_KEY_FILE, 'r').read().strip()


def repair_loop(prompt, file_name, folder, bug, t_chances, stop="# Provide a fix for the buggy function",
                skip_val=True) -> (bool, bool, list):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prompt)
    temperature = 0.8
    top_p = 0.95
    config = create_openai_config(prompt=prompt, stop=stop, temperature=temperature, top_p=top_p)
    total_times = 0
    while t_chances > 0:
        total_times += 1
        t_chances -= 1
        print("Try: {}".format(total_times))
        ret = request_engine(config)
        if ret is None:
            return False, False, []
        output = ret["choices"][0]['text'].strip()
        diff = get_unified_diff(bug['buggy'], output)
        finish_reason = ret["choices"][0]['finish_reason']
        if finish_reason != "stop":
            continue
        if diff in p_diff:
            repair_result[p_diff[diff]]['num'] += 1
            continue
        p_diff[diff] = len(repair_result)
        print(diff)
        repair_result.append({'output': output,
                              'diff': diff,
                              'finish_reason': finish_reason,
                              'entropy': (-sum(ret["choices"][0]['logprobs']["token_logprobs"]) / len(ret["choices"][0]['logprobs']["token_logprobs"]),
                                          -sum(ret["choices"][0]['logprobs']["token_logprobs"])),
                              'valid': _run_validation(file_name.split(".")[0],
                                                       file_name.split(".")[0] + "_" +
                                                       str(len(repair_result)) + "." + file_name.split(".")[1],
                                                       folder, output, skip_val=skip_val),
                              'num': 1})

    end = time.time()
    print("{} Unique Patches Generated in {}s".format(len(repair_result), end - start))

    return False, False, repair_result


def repair_codex_edits(bugs):
    result = {}
    for file_name, bug in bugs.items():
        print(file_name)
        ret = openai.Edit.create(
            engine="code-davinci-edit-001",
            top_p=1,
            temperature=0,
            input=bug['buggy'],
            instruction="Fix the buggy function"
        )
        time.sleep(3.1)  # need to wait for rate limit to finish Limit: 20 requests / minute
        result[file_name] = {'output': ret["choices"][0]['text'].strip(),
                             'diff': get_unified_diff(bug['buggy'], ret["choices"][0]['text'].strip())}
        print(get_unified_diff(bug['buggy'], ret["choices"][0]['text'].strip()))

    with open("Results/codex_repair_edits.json", "w") as f:  # write to file
        json.dump(result, f)


def location_codex(bugs):
    result = {}
    for file_name, bug in bugs.items():
        print(file_name)
        prompt = LOCATION_PROMPT.format(bug['buggy'])
        ret = openai.Completion.create(
            engine="code-davinci-002",
            prompt=prompt,
            max_tokens=100,
            top_p=1,
            temperature=0,
            stop="# Provide the location of the bug in the buggy function"
        )
        time.sleep(4.1)  # need to wait for rate limit to finish Limit: 20 requests / minute
        result[file_name] = {'output': ret["choices"][0]['text'].strip(),
                             'finish_reason': ret["choices"][0]['finish_reason']}
        print(ret["choices"][0]['text'].strip())

    with open("Results/codex_f_loc.json", "w") as f:  # write to file
        json.dump(result, f)


def suffix_repair_loop(prefix, suffix, file_name, folder, bug, chances, skip_val=False) -> (bool, bool, list):
    repair_result = []
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prefix)
    print(">>> [INSERT] <<<")
    print(suffix)
    # Start with initial config with top choices by model
    initial_config = create_openai_config_suffix(prompt=prefix, suffix=suffix)
    ret = request_engine(initial_config)
    if ret is None:
        return False, False, []
    output = prefix + ret["choices"][0]['text'] + suffix
    diff = get_unified_diff(bug['buggy'], output)
    finish_reason = ret["choices"][0]['finish_reason']

    print(finish_reason)
    print(diff)

    repair_result.append({'output': output,
                          'diff': diff,
                          'finish_reason': finish_reason,
                          'valid': _run_validation(file_name.split(".")[0], file_name, folder, output,
                                                   skip_val=skip_val)})
    if repair_result[0]['valid']:
        return True, True, repair_result
    if finish_reason != "stop":
        return False, False, repair_result

    temperature = 0.85
    config = create_openai_config_suffix(prompt=prefix, suffix=suffix, temperature=temperature)
    current_chances = 1
    total_times = 0
    while current_chances < chances and total_times < 20:
        total_times += 1
        print("Try {}".format(total_times))
        ret = request_engine(config)
        output = prefix + ret["choices"][0]['text'] + suffix
        diff = get_unified_diff(bug['buggy'], output)
        finish_reason = ret["choices"][0]['finish_reason']
        if diff.strip() == "":
            continue
        already = False
        for previous in repair_result:
            if output == previous['output'] or finish_reason == "length":
                already = True
                break
        if already:
            continue

        print(diff)
        repair_result.append({'output': output,
                              'diff': diff,
                              'finish_reason': finish_reason,
                              'valid': _run_validation(file_name.split(".")[0],
                                                       file_name.split(".")[0] + "_" + str(
                                                           current_chances + 1) + "." + file_name.split(".")[1],
                                                       folder, output, skip_val=skip_val)})
        if repair_result[-1]['valid']:
            return True, False, repair_result
        current_chances += 1

    return False, False, repair_result


def suffix_repair(args, bugs, folder, chances, skip_val=True):
    """
    Codex suffix repair, write each patch to corresponding file
    :param args: arguments
    :param bugs: dict of bugs
    :param folder: folder ot save the files
    :param chances: prompt as input to codex
    :param skip_val: number of chances to try to repair (0 means only try once with temp=1)
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    correct = 0
    incorrect = 0
    first_tries = 0
    for file_name, bug in bugs.items():
        prefix = bug['prefix']
        suffix = bug['suffix']
        valid, first_try, result[file_name] = suffix_repair_loop(prefix, suffix, file_name, folder, bug, chances, skip_val=skip_val)

        if first_try:
            first_tries += 1
        if valid:
            correct += 1
        else:
            incorrect += 1

    with open(folder + "/codex_repair.json", "w") as f:  # write to file
        json.dump(result, f)

    print("First Try: {} Correct: {} Incorrect: {}".format(first_tries, correct, incorrect))


def repair_codex(args, bugs, folder, used_prompt, chances, vary, stop, skip_val=True, only_same=False):
    """
    Codex repair loop, write each patch to corresponding file
    :param args: arguments
    :param bugs: dict of bugs
    :param folder: folder to save the files
    :param used_prompt: prompt as input to codex
    :param chances: number of chances to try to repair (0 means only try once with temp=1)
    :param vary: whether or not the prompt should be varied (specifically designed for d4j and complex bugs, where the
            we use the an example fix from the same project
    :param stop: stop condition for codex
    :param skip_val: if True, skip validation
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/prompt.txt", "w") as f:
        f.write(used_prompt)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()

    for file_name, bug in bugs.items():
        # if 'Time' in file_name or 'Closure' in file_name or 'Math' in file_name or 'Mockito' in file_name or 'Chart' in file_name or 'Lang' in file_name:
            if vary:
                example_bug, example_fix = pick_smallest_example_fix(bugs, file_name, only_same=only_same)
                prompt = used_prompt.format(example_bug=example_bug, example_fix=example_fix, bug=bug['buggy'])
            else:
                prompt = used_prompt.format(bug['buggy'])
            valid, first_try, result[file_name] = repair_loop(prompt, file_name, folder, bug, t_chances=chances,
                                                              stop=stop, skip_val=skip_val)
            if len(result[file_name]) != 0:
                t_generated += chances
                t_unique += len(result[file_name])

    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/codex_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def parse_codex_bug_report(bug_report):
    with open(bug_report, 'r') as f:
        result = json.load(f)

    count = 0
    for bug, patches in result.items():
        if patches[-1]['valid'] == True:
            # print(bug)
            # print(patches[-1]['diff'])
            count += 1
        else:
            print(bug)

    print(count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="defects4j",
                        help="Dataset to use, current support: defects4j, quixbug-python, manybugs")
    parser.add_argument("--chances", type=int, default=1)
    parser.add_argument("--vary", action="store_true", default=False)
    parser.add_argument("--skip_val", action="store_true", default=False)
    parser.add_argument("--folder", type=str, default="Results/test")
    parser.add_argument("--suffix", action="store_true", default=False)
    args = parser.parse_args()
    if args.dataset == "defects4j":
        if args.suffix:
            dataset = clean_parse_d4j_single_hunk(folder="../../")
        else:
            dataset = clean_parse_d4j(folder="../../")
        if args.vary:
            prompt = JAVA_LONG_VARY_PROMPT
        else:
            prompt = JAVA_BASE_PROMPT
        stop = "// Provide a fix for the buggy function"
    elif args.dataset == "quixbug-python":
        dataset = parse_python(folder='../../')
        if args.vary:
            prompt = VARY_BASE_PROMPT
        else:
            prompt = BASE_PROMPT
        stop = "# Provide a fix for the buggy function"
    elif args.dataset == "manybugs":
        dataset = clean_parse_manybugs(folder="../../")
        prompt = C_BASE_PROMPT
        stop = "/* Provide a fix for the buggy function */"
    elif args.dataset == "codeflaws":
        dataset = clean_parse_codeflaws(folder='../../')
        prompt = C_VARY_PROMPT
        stop = "/* Provide a fix for the buggy function */"
        args.language = "c"
    elif args.dataset in ['Bears', 'BugsInPy', 'BugsDotJar']:
        def clean_parse_single_hunk(dataset: dict, is_python: bool) -> dict:
            extension = '.py' if is_python else '.java'
            cleaned_result = {}
            for bug_id, data in dataset.items():
                lines = data['buggy'].splitlines()
                leading_white_space = len(lines[0]) - len(lines[0].lstrip())
                cleaned_result[bug_id + extension] = {"buggy": "\n".join(
                    [line[leading_white_space:] for line in lines])}
                lines = data["prefix"].splitlines()
                cleaned_result[bug_id + extension]["prefix"] = "\n".join(
                    [line[leading_white_space:] for line in lines])
                lines = data["suffix"].splitlines()
                cleaned_result[bug_id + extension]["suffix"] = "\n".join(
                    [line[leading_white_space:] for line in lines])
                lines = data['fix'].splitlines()
                leading_white_space = len(lines[0]) - len(lines[0].lstrip())
                cleaned_result[bug_id + extension]["fix"] = "\n".join(
                    [line[leading_white_space:] for line in lines])
            return cleaned_result

        def clean_parse_single_func(dataset: dict, is_python: bool) -> dict:
            extension = '.py' if is_python else '.java'
            cleaned_result = {}
            for bug_id, data in dataset.items():
                lines = data['buggy'].splitlines()
                leading_white_space = len(lines[0]) - len(lines[0].lstrip())
                cleaned_result[bug_id + extension] = {"buggy": "\n".join(
                    [line[leading_white_space:] for line in lines])}
                lines = data['fix'].splitlines()
                leading_white_space = len(lines[0]) - len(lines[0].lstrip())
                cleaned_result[bug_id + extension]["fix"] = "\n".join(
                    [line[leading_white_space:] for line in lines])
            return cleaned_result

        is_python = args.dataset == 'BugsInPy'
        func_save_path = 'single_function_repair.json'
        hunk_save_path = 'single_function_single_hunk_repair.json'
        if not 'single_line' in args.__dict__:
            args.single_line = False
        f_dataset = os.path.join(
            os.path.pardir,
            os.path.pardir,
            args.dataset,
            hunk_save_path if (
                args.suffix or args.single_line) else func_save_path
        )
        with open(f_dataset) as f:
            dataset = json.load(f)
        dataset = (clean_parse_single_hunk if (args.single_line or args.suffix) else clean_parse_single_func)(
            dataset, is_python)
        if args.single_line:
            dataset = filter_single_line(dataset)
        print('Size of dataset:', len(dataset))
        # Copied from Defects4J above
        if args.vary:
            prompt = VARY_BASE_PROMPT if is_python else JAVA_LONG_VARY_PROMPT
        else:
            prompt = BASE_PROMPT if is_python else JAVA_BASE_PROMPT
        stop = '# Provide a fix for the buggy function' if is_python else '// Provide a fix for the buggy function'
        if args.single_line:
            stop = "\n"
        args.language = 'python' if is_python else 'java'

    else:
        print("Unknown dataset: {}".format(args.dataset))
        return -1
    if args.suffix:
        suffix_repair(args, dataset, args.folder, chances=args.chances,skip_val=args.skip_val)
    else:
        repair_codex(args, dataset, args.folder, prompt, chances=args.chances,
                     vary=args.vary, stop=stop, skip_val=args.skip_val, only_same=args.dataset.startswith("defects4j"))


if __name__ == "__main__":
    main()
