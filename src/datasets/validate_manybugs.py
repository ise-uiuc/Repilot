import bugzoo
import sys
import os
import json
import gc
import glob
from bugzoo import Patch
from difflib import unified_diff
sys.path.append(os.path.dirname(os.path.join(sys.path[0], '../'))) # Hack

from parse_manybugs import parse_manybugs

url = "http://127.0.0.1:8080"


def get_unified_diff(source, mutant, filename):
    output = ""
    for line in unified_diff(source.split('\n'), mutant.split('\n'), lineterm='', tofile=filename, fromfile=filename):
        output += line + "\n"
    return output


def check_passing_test():
    clean_result = {}
    with open("../ManyBugs/pass_tests.json", "r") as f:
        result = json.load(f)
    print(len(result))
    print(len([x in x for x in result.values() if len(x) > 0]))
    for bug, tests in result.items():
        if len(tests) == 0:
            continue
        if "n1" in tests:
            clean_result[bug] = tests
    print(len(clean_result))
    return clean_result

def which_patches_are_good(repair_result):
    tests = check_passing_test()
    with open(repair_result, "r") as f:
        repair_dict = json.load(f)
    count = 0
    total = 0
    first_try = 0
    for bug, bug_dict in repair_dict.items():
        if bug.split(".c")[0] not in tests:
            continue
        if len(bug_dict) > 1:
            total += 1
            for patch in bug_dict:
                if patch['valid']:
                    print(bug)
                    count += 1
                    break
            if bug_dict[0]['valid']:
                first_try += 1
            # for index, patch in enumerate(bug_dict):
            #     if patch['valid']:
            #         print(index)
            #         print(patch['diff'])
    print(count)
    print(first_try)
    print(total)

def validate_all_patches(folder, j_file):
    tests = check_passing_test()

    with open(folder + "/" + j_file, "r") as f:
        repair_dict = json.load(f)
    with open("../ManyBugs/single_function_repair.json", "r") as f:
        bug_dict = json.load(f)

    plausible = 0
    total = 0
    client = bugzoo.Client(url)

    for file in sorted(glob.glob(folder + "/*.c")):

        project = file.split("/")[-1].split("_")[0]
        bug_id = file.split("/")[-1].split("_")[1].split(".c")[0]
        bugname = project + "_" + bug_id
        if len(file.split("/")[-1].split("_")) == 2:
            index = 0
        else:
            index = int(file.split("/")[-1].split("_")[2].split(".c")[0])-1

        if bugname not in tests:
            continue

        if len(repair_dict[project + "_" + bug_id+".c"]) <= index:
            continue

        if repair_dict[project + "_" + bug_id+".c"][index]['finish_reason'] != "stop":
            continue
        if repair_dict[project + "_" + bug_id+".c"][index]['diff'] == "":
            continue

        print(bugname, index)

        with open(file, 'r') as f:
            patch = f.readlines()
        with open("../ManyBugs/buggy_programs/"+bugname+".c", "r") as f:
            source = f.readlines()

        start = bug_dict[bugname]['start']
        end = bug_dict[bugname]['end']
        patch = "".join(source[:start - 1] + patch + source[end:])
        source = "".join(source)
        diff = get_unified_diff(source, patch, bug_dict[bugname]['filename'])
        # print(diff)
        print(repair_dict[project + "_" + bug_id+".c"][index]['diff'])
        bug = client.bugs['manybugs:' + bugname.replace("_", ":")]
        container = client.containers.provision(bug)
        patch = Patch.from_unidiff(diff)
        client.containers.patch(container, patch)
        client.containers.compile(container)
        success = True
        for test in bug.tests:
            if str(test.name) not in tests[bugname] or "p" in test.name:
                continue
            r = client.containers.test(container, test)
            print(r.to_dict())
            if not r.to_dict()['passed']:
                success = False
                break
        if success:
            repair_dict[bugname+".c"][index]['valid'] = True
            plausible += 1

        del client.containers[container.uid]
        gc.collect()
        total += 1

    print(len(sorted(glob.glob(folder + "/*.c"))))
    print("{}/{} are plausible".format(plausible, total))

    with open(folder + "/" + j_file, "w") as f:
        json.dump(repair_dict, f)



def get_all_passing_tests():
    with open("../ManyBugs/pass_tests.json", "r") as f:
        result = json.load(f)
    client = bugzoo.Client(url)
    data = parse_manybugs("../")
    for project, v in data.items():
        print(project)
        if project in result:
            continue
        with open("../ManyBugs/buggy_programs/"+project+".c", 'r') as f:
            buggy = f.read()
        with open("../ManyBugs/correct_programs/"+project+".c", 'r') as f:
            fixed = f.read()
        diff = get_unified_diff(buggy, fixed, v['filename'])
        try:
            bug = client.bugs['manybugs:'+project.replace("_", ":")]
        except:
            continue
        result[project] = []
        try:
            container = client.containers.provision(bug)
        except:
            try:
                client.bugs.build(bug)
                container = client.containers.provision(bug)
            except:
                with open("../ManyBugs/pass_tests.json", 'w') as f:
                    json.dump(result, f)
                continue

        patch = Patch.from_unidiff(diff)
        client.containers.patch(container, patch)
        client.containers.compile(container)
        for test in bug.tests:
            print(test)
            r = client.containers.test(container, test)
            if r.to_dict()['passed']:
                result[project].append(test.name)
            print(r.to_dict())

        del client.containers[container.uid]
        gc.collect()

        with open("../ManyBugs/pass_tests.json", 'w') as f:
            json.dump(result, f)

if __name__ == "__main__":
    # test()
    # get_all_passing_tests()
    # check_passing_test()
    validate_all_patches("../Repair/LM/Results/manybugs-neo-2.7", "lm_repair.json")
    # which_patches_are_good("../Repair/LM/Results/manybugs-neo-2.7/lm_repair.json")
    #which_patches_are_good("../Repair/LM/Results/manybugs-gpt-j-200/lm_repair.json")
