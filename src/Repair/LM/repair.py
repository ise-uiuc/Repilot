import argparse
import sys
import torch
import os
import json
import time

from model import GPT2, SpanLM
from datasets.parse_quixbugs import parse_python, get_unified_diff, parse_java #, parse_java_single_line
from datasets.parse_d4j import clean_parse_d4j, clean_parse_d4j_single_hunk #, clean_parse_d4j_single_line
from datasets.parse_manybugs import clean_parse_manybugs, clean_parse_manybugs_single_hunk
from datasets.parse_codeflaws import clean_parse_codeflaws, clean_parse_codeflaws_single_hunk
from Repair.prompt import BASE_PROMPT, LOCATION_PROMPT, LONG_BASE_PROMPT, JAVA_BASE_PROMPT, JAVA_VARY_PROMPT, \
    JAVA_LONG_VARY_PROMPT, VARY_BASE_PROMPT, TESTCASE_BASE_PROMPT, EXPERT_PROMPT
from Repair.prompt import C_VARY_PROMPT
from Repair.prompt import JAVA_INFILL_BASE_PREFIX, JAVA_INFILL_BASE_SUFFIX, INFILL_BASE_PREFIX, INFILL_BASE_SUFFIX
from Repair.util import pick_smallest_example_fix, set_seed, _run_validation, build_example_fixes, \
    pick_smallest_example_fix_name, get_testcase
from datasets.utils import filter_single_line


def suffix_repair_loop(args, model: SpanLM, prefix, suffix, file_name, folder, bug, t_chances, skip_val=True,
                       ignore_prefix=False):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prefix)
    print(">>> [INSERT] <<<")
    print(suffix)
    if not model.check_input(prefix, suffix, bug['buggy'], ignore_prefix):
        return 0, False, False, repair_result

    total_times = 0
    while t_chances > 0:
        total_times += 1
        torch.cuda.empty_cache()
        print("Try :{}".format(total_times))
        well, early_stop, outputs, entropies = model.model_predict(prefix=prefix, suffix=suffix,
                                                                   do_sample=True, use_max_length=ignore_prefix,
                                                                   buggy=bug['buggy'],
                                                                   num_samples=t_chances)
        t_chances -= args.batch_size
        if well:
            for index, output in enumerate(outputs):
                if not ignore_prefix:
                    output = prefix + output + suffix
                diff = get_unified_diff(bug['buggy'], output)
                if diff in p_diff:
                    repair_result[p_diff[diff]]['num'] += 1
                    continue
                p_diff[diff] = len(repair_result)
                print(diff)
                repair_result.append({'output': output,
                                      'diff': diff,
                                      'finish_reason': 'stop',
                                      'entropy': entropies[index],
                                      'valid': _run_validation(file_name.split(".")[0],
                                                               file_name.split(".")[0] + "_" + str(
                                                                   len(repair_result)) + "." + file_name.split(".")[1],
                                                               folder, output, skip_val=skip_val),
                                      'num': 1})
    end = time.time()
    print("{} Unique Patches Generated in {}s".format(
        len(repair_result), end - start))

    return total_times, False, False, repair_result


def single_line_repair_loop(args, model, prefix, suffix, file_name, folder, bug, t_chances, skip_val=True):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prefix)
    if not model.check_input(prefix, ""):
        return 0, False, False, repair_result

    total_times = 0
    while t_chances > 0:
        total_times += 1
        torch.cuda.empty_cache()
        print("Try :{}".format(total_times))
        well, length, outputs, entropies = model.model_predict(prefix, bug['buggy'], do_sample=True,
                                                               num_samples=t_chances)
        t_chances -= args.batch_size
        if well:
            for index, output in enumerate(outputs):
                output = prefix + output + suffix
                diff = get_unified_diff(bug['buggy'], output)
                if diff in p_diff:
                    repair_result[p_diff[diff]]['num'] += 1
                    continue
                p_diff[diff] = len(repair_result)
                print(diff)
                repair_result.append({
                    'output': output,
                    'diff': diff,
                    'finish_reason': 'stop',
                    'entropy': entropies[index],
                    'valid': _run_validation(file_name.split(".")[0],
                                             file_name.split(".")[0] + "_" + str(
                                                 len(repair_result)) + "." + file_name.split(".")[1],
                                             folder, output, skip_val=skip_val),
                    'num': 1})

    end = time.time()

    print("{} Unique Patches Generated in {}s".format(
        len(repair_result), end - start))

    return len(repair_result), False, False, repair_result


def repair_loop(args, model, prompt, file_name, folder, bug, t_chances, skip_val=True):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prompt)
    if not model.check_input(prompt, bug['buggy']):
        return 0, False, False, repair_result

    total_times = 0
    while t_chances > 0:
        total_times += 1
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Try :{}".format(total_times))
        try:
            well, length, outputs, entropies = model.model_predict(prompt, bug['buggy'], do_sample=True,
                                                                   num_samples=t_chances)
        except RuntimeError as e:
            print(f'Cannot generate patches for {file_name}')
            with open('bad-bears-line', 'a') as f:
                f.write(file_name + '\n')
            break
        finally:
            t_chances -= args.batch_size
        if well:
            for index, output in enumerate(outputs):
                diff = get_unified_diff(bug['buggy'], output)
                if diff in p_diff:
                    repair_result[p_diff[diff]]['num'] += 1
                    continue
                p_diff[diff] = len(repair_result)
                print(diff)
                repair_result.append({  'output': output,
                    'diff': diff,
                    'finish_reason': 'stop',
                    'entropy': entropies[index],
                    'valid': _run_validation(file_name.split(".")[0],
                                             file_name.split(".")[0] + "_" + str(
                        len(repair_result)) + "." + file_name.split(".")[1],
                        folder, output, skip_val=skip_val),
                    'num': 1})

    end = time.time()

    print("{} Unique Patches Generated in {}s".format(
        len(repair_result), end - start))

    return len(repair_result), False, False, repair_result


def suffix_repair(args, model, bugs, folder, chances, skip_val=True, set_prefix="", set_suffix="", only_same=False):
    """
    Suffix LM repair loop
    :param args: input arguments
    :param model: model to use for repair
    :param bugs: dict of bugs
    :param folder: folder to save the files
    :param chances: number of chances to try to repair
    :param skip_val: if True, skip validation
    :param set_suffix: set prefix for infilling
    :param set_prefix: set suffix for infilling
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))
    if args.suffix_end:
        with open(folder + "/prompt.txt", "w") as f:
            f.write(set_prefix)
            f.write(set_suffix)

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()
    for file_name, bug in bugs.items():
        if 'suffix' not in bug:
            continue
        # if 'Time' in file_name or 'Closure' in file_name or 'Math' in file_name or 'Mockito' in file_name or 'Chart' in file_name or 'Lang' in file_name:
        # if 'Cli' in file_name or 'Codec' in file_name or 'Collections' in file_name or 'Compress' in file_name or 'Csv' \
        #         in file_name or 'Gson' in file_name or 'Jackson' in file_name or 'Jsoup' in file_name or 'JxPath' in file_name:
        if args.suffix:
            suffix = "\n" + bug['suffix']
            # leading white space removal is needed to help with codet5 prediction since it does not have concept of
            # white spaces
            #leading_white_space = len(bug['buggy'].splitlines()[bug['line_no']]) - len(bug['buggy'].splitlines()[bug['line_no']].lstrip())
            prefix = bug['prefix'] + "\n"  # + " "*leading_white_space
            n_generated, valid, first_try, result[file_name] = suffix_repair_loop(args, model, prefix, suffix,
                                                                                  file_name,
                                                                                  folder, bug,
                                                                                  chances, skip_val,
                                                                                  ignore_prefix=False)
        else:
            example_bug, example_fix = pick_smallest_example_fix(
                bugs, file_name, only_same=only_same)
            suffix = set_suffix.format(
                example_bug=example_bug, example_fix=example_fix, bug=bug['buggy'])
            prefix = set_prefix.format(bug['buggy'])
            n_generated, valid, first_try, result[file_name] = suffix_repair_loop(args, model, prefix, suffix,
                                                                                  file_name,
                                                                                  folder, bug,
                                                                                  chances, skip_val, ignore_prefix=True)
        if n_generated >= 1:
            t_generated += chances
            t_unique += len(result[file_name])

    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def single_line_repair(args, model, bugs, folder, chances, skip_val):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()
    for file_name, bug in bugs.items():
        # if not ('Time' in file_name or 'Closure' in file_name or 'Math' in file_name or 'Mockito' in file_name or 'Chart' in file_name or 'Lang' in file_name):
        # if 'Cli' in file_name or 'Codec' in file_name or 'Collections' in file_name or 'Compress' in file_name or 'Csv' \
        #         in file_name or 'Gson' in file_name or 'Jackson' in file_name or 'Jsoup' in file_name or 'JxPath' in file_name:
        if "suffix" not in bug:
            continue

        suffix = "\n" + bug['suffix']
        prefix = bug['prefix'] + "\n"
        n_generated, valid, first_try, result[file_name] = single_line_repair_loop(args, model, prefix, suffix, file_name, folder, bug,
                                                                                   chances, skip_val)
        if n_generated >= 1:
            t_generated += chances
            t_unique += len(result[file_name])

    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def repair(args, model, bugs, folder, used_prompt, chances, vary, skip_val=True, testcase=False, expert=False, only_same=True):
    """
    LM repair loop, write each patch to corresponding file
    :param args: input arguments
    :param model: model to use for repair
    :param bugs: dict of bugs
    :param folder: folder to save the files
    :param used_prompt: prompt as input to the model
    :param chances: number of chances to try to repair
    :param vary: whether or not the prompt should be varied (specifically designed for d4j and complex bugs, where the
            we use the an example fix from the same project
    :param skip_val: if True, skip validation
    :param testcase
    :param expert
    :param only_same
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
        # if file_name != 'Bears-137.java':
        #     continue
        # if 'Time' in file_name or 'Closure' in file_name or 'Math' in file_name or 'Mockito' in file_name or 'Chart' in file_name or 'Lang' in file_name:
        # if 'Cli' in file_name or 'Codec' in file_name or 'Collections' in file_name or 'Compress' in file_name or 'Csv' \
        #         in file_name or 'Gson' in file_name or 'Jackson' in file_name or 'Jsoup' in file_name or 'JxPath' in file_name:
        if testcase:
            example_bug, example_fix, example_name = pick_smallest_example_fix_name(
                bugs, file_name, only_same=only_same)
            example_testcases = get_testcase(example_name.split(
                ".py")[0], "../../QuixBugs/json_testcases")
            prompt = used_prompt.format(example_testcases=example_testcases, example_bug=example_bug, example_fix=example_fix,
                                        testcases=get_testcase(file_name.split(
                                            ".py")[0], "../../QuixBugs/json_testcases"),
                                        bug=bug['buggy'])
        elif expert:
            # example_bug, example_fix = pick_smallest_example_fix(bugs, file_name, only_same=only_same)
            prompt = used_prompt.format(bug=bug['buggy'])
        elif vary:
            if "Collections" in file_name:
                example_bug, example_fix = pick_smallest_example_fix(
                    bugs, file_name, only_same=False)
            else:
                example_bug, example_fix = pick_smallest_example_fix(
                    bugs, file_name, only_same=only_same)
            prompt = used_prompt.format(
                example_bug=example_bug, example_fix=example_fix, bug=bug['buggy'])
        else:
            prompt = build_example_fixes(
                bugs, file_name, model, only_same, language=args.language)
        # prompt = used_prompt.format(bug['buggy'])
        n_generated, valid, first_try, result[file_name] = repair_loop(args, model, prompt, file_name, folder, bug,
                                                                       chances, skip_val)
        if n_generated >= 1:
            t_generated += chances
            t_unique += len(result[file_name])

    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--datasets", type=str, default="defects4j",
                        help="datasets to use, current support: defects4j, quixbug-python, quixbugs-java, manybugs")
    parser.add_argument("--chances", type=int, default=1)
    parser.add_argument("--vary", action="store_true", default=False)
    parser.add_argument("--testcase", action="store_true", default=False)
    parser.add_argument("--expert", action="store_true", default=False)
    parser.add_argument("--skip_val", action="store_true", default=False)
    parser.add_argument("--folder", type=str, default="Results/test")
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--suffix", action="store_true", default=False)
    parser.add_argument("--single_line", action="store_true", default=False)
    parser.add_argument("--suffix_end", action="store_true", help="for SpanLM but without location, i.e. same as "
                                                                  "regular LM but ask the model to predict the entire "
                                                                  "replacement function", default=False)
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()
    prompt, set_suffix, set_prefix = "", "", ""
    if args.datasets == "defects4j":
        if args.suffix:
            datasets = clean_parse_d4j_single_hunk(folder="../../")
        elif args.single_line:
            datasets = clean_parse_d4j_single_line(folder="../../")
        else:
            datasets = clean_parse_d4j(folder="../../")
        if args.suffix_end:
            set_prefix = JAVA_INFILL_BASE_PREFIX
            set_suffix = JAVA_INFILL_BASE_SUFFIX
        else:
            if args.vary:
                prompt = JAVA_LONG_VARY_PROMPT
            else:
                prompt = JAVA_BASE_PROMPT
        stop = "// Provide a fix for the buggy function"
        if args.single_line:
            stop = "\n"
        args.language = "java"
    elif args.datasets == "quixbug-python":
        datasets = parse_python(folder='../../')
        if args.suffix_end:
            set_prefix = INFILL_BASE_PREFIX
            set_suffix = INFILL_BASE_SUFFIX
        else:
            if args.testcase:
                prompt = TESTCASE_BASE_PROMPT
            elif args.vary:
                prompt = VARY_BASE_PROMPT
            elif args.expert:
                prompt = EXPERT_PROMPT
            else:
                prompt = BASE_PROMPT
        stop = "# Provide a fix for the buggy function"
        if args.expert:
            stop = "# A buggy function"
        if args.single_line:
            stop = "\n"
        args.language = "python"
    elif args.datasets == "quixbug-java":
        if args.single_line:
            datasets = parse_java_single_line(folder="../../")
        else:
            datasets = parse_java(folder='../../')
        if args.suffix_end:
            set_prefix = INFILL_BASE_PREFIX
            set_suffix = INFILL_BASE_SUFFIX

        else:
            if args.vary:
                prompt = JAVA_LONG_VARY_PROMPT
            else:
                prompt = JAVA_BASE_PROMPT
        stop = "// Provide a fix for the buggy function"
        if args.single_line:
            stop = "\n"
        args.language = "java"
    elif args.datasets == "manybugs":
        if args.suffix:
            datasets = clean_parse_manybugs_single_hunk(folder='../../')
        else:
            datasets = clean_parse_manybugs(folder='../../')
        prompt = C_VARY_PROMPT
        stop = "/* Provide a fix for the buggy function */"
        args.language = "c"
    elif args.datasets == "codeflaws":
        if args.suffix:
            datasets = clean_parse_codeflaws_single_hunk(folder="../../")
        else:
            datasets = clean_parse_codeflaws(folder='../../')
        prompt = C_VARY_PROMPT
        stop = "/* Provide a fix for the buggy function */"
        args.language = "c"
    elif args.datasets in ['Bears', 'BugsInPy', 'BugsDotJar']:
        def clean_parse_single_hunk(datasets: dict, is_python: bool) -> dict:
            extension = '.py' if is_python else '.java'
            cleaned_result = {}
            for bug_id, data in datasets.items():
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

        def clean_parse_single_func(datasets: dict, is_python: bool) -> dict:
            extension = '.py' if is_python else '.java'
            cleaned_result = {}
            for bug_id, data in datasets.items():
                lines = data['buggy'].splitlines()
                leading_white_space = len(lines[0]) - len(lines[0].lstrip())
                cleaned_result[bug_id + extension] = {"buggy": "\n".join(
                    [line[leading_white_space:] for line in lines])}
                lines = data['fix'].splitlines()
                leading_white_space = len(lines[0]) - len(lines[0].lstrip())
                cleaned_result[bug_id + extension]["fix"] = "\n".join(
                    [line[leading_white_space:] for line in lines])
            return cleaned_result

        is_python = args.datasets == 'BugsInPy'
        func_save_path = 'single_function_repair.json'
        hunk_save_path = 'single_function_single_hunk_repair.json'
        f_datasets = os.path.join(
            os.path.pardir,
            os.path.pardir,
            args.datasets,
            hunk_save_path if (
                args.suffix or args.single_line) else func_save_path
        )
        with open(f_datasets) as f:
            datasets = json.load(f)
        datasets = (clean_parse_single_hunk if (args.single_line or args.suffix) else clean_parse_single_func)(
            datasets, is_python)
        if args.single_line:
            datasets = filter_single_line(datasets)
        print('Size of datasets:', len(datasets))
        # Copied from Defects4J above
        if args.suffix_end:
            set_prefix = INFILL_BASE_PREFIX if is_python else JAVA_INFILL_BASE_PREFIX
            set_suffix = INFILL_BASE_SUFFIX if is_python else JAVA_INFILL_BASE_SUFFIX
        else:
            if args.vary:
                prompt = VARY_BASE_PROMPT if is_python else JAVA_LONG_VARY_PROMPT
            else:
                prompt = BASE_PROMPT if is_python else JAVA_BASE_PROMPT
        stop = '# Provide a fix for the buggy function' if is_python else '// Provide a fix for the buggy function'
        if args.single_line:
            stop = "\n"
        args.language = 'python' if is_python else 'java'
    else:
        print("Unknown datasets: {}".format(args.datasets))
        return -1

    set_seed(args.seed)
    if args.suffix:
        model = SpanLM(pretrained=args.model_name,
                       weight=args.weight, batch_size=args.batch_size)
        suffix_repair(args, model, datasets, args.folder,
                      args.chances, args.skip_val)
    elif args.suffix_end:
        model = SpanLM(pretrained=args.model_name,
                       weight=args.weight, batch_size=args.batch_size)
        suffix_repair(args, model, datasets, args.folder, args.chances, args.skip_val,
                      set_prefix=set_prefix, set_suffix=set_suffix, only_same=args.datasets.startswith("defects4j"))
    elif args.single_line:
        model = GPT2(batch_size=args.batch_size,
                     pretrained=args.model_name, stop=stop, weight=args.weight)
        single_line_repair(args, model, datasets, args.folder,
                           args.chances, args.skip_val)
    else:
        model = GPT2(batch_size=args.batch_size,
                     pretrained=args.model_name, stop=stop, weight=args.weight)
        repair(args, model, datasets, args.folder, prompt, args.chances,
               args.vary, args.skip_val, testcase=args.testcase, expert=args.expert,
               only_same=args.datasets.startswith("defects4j"))


if __name__ == '__main__':
    main()
