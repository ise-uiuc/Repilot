import glob
import json
import subprocess
from difflib import unified_diff


def get_unified_diff(source, mutant):
    output = ""
    for line in unified_diff(source.split('\n'), mutant.split('\n'), lineterm=''):
        output += line + "\n"
    return output


def get_function_start_end(ctag_output):
    lines = ctag_output.splitlines()[1:]
    function_dict = {}
    for l in lines:
        beginning, end = -1, -1
        function_name = l.split("\t")[0]
        for s in l.split("\t"):
            if "line:" in s:
                beginning = int(s.split(":")[1])
            elif "end:" in s:
                end = int(s.split(":")[1])
        # assert function_name not in function_dict
        function_dict[function_name + str(len(function_dict))] = (beginning, end)

    return function_dict


def run_test(prefix, folder, script):
    test = 1
    tests = []
    while 1:
        current_test = prefix + str(test)
        try:
            output = subprocess.getoutput("cd " + folder + "; ./" + script + " " + current_test)
        except:
            test += 1
            continue
        if output.strip() == "":
            break
        if "syntax error near unexpected token" in output:
            break
        if "Accepted" not in output:
            return False, []
        else:
            tests.append(current_test)
        test += 1
    return True, tests


def run_all_test():
    with open("../Codeflaws/single_function_repair.json") as f:
        result = json.load(f)

    for folder in glob.glob("../../codeflaws/*/"):
        bug = folder.split("/")[-2]
        if bug+".c" not in result:
            continue
        bug_id = bug.split("-bug-")[0]
        buggy_c = bug.split("-bug-")[-1].split("-")[0]
        print(bug)

        with open("../Codeflaws/correct_programs/"+bug+".c", "r") as f:
            source = f.read()
        with open(folder + bug_id + "-" + buggy_c + ".c", "w") as f:
            f.write(source)
        subprocess.getoutput("cd " + folder + "; make clean; make;")
        s, tests = run_test("p", folder, "test-genprog.sh")
        if not s:
            result[bug + ".c"]['tests'] = []
            result[bug + ".c"]['regression_tests'] = []
            continue
        s, temp_tests = run_test("n", folder, "test-genprog.sh")
        if not s:
            result[bug + ".c"]['tests'] = []
            result[bug + ".c"]['regression_tests'] = []
            continue
        tests.extend(temp_tests)
        s, regression_tests = run_test("p", folder, "test-valid.sh")
        if not s:
            result[bug + ".c"]['tests'] = []
            result[bug + ".c"]['regression_tests'] = []
            continue
        s, temp_tests = run_test("n", folder, "test-valid.sh")
        if not s:
            result[bug + ".c"]['tests'] = []
            result[bug + ".c"]['regression_tests'] = []
            continue
        regression_tests.extend(temp_tests)
        result[bug+".c"]['tests'] = tests
        result[bug + ".c"]['regression_tests'] = regression_tests

    with open("../Codeflaws/single_function_repair.json", "w") as f:
        json.dump(result, f)


def single_problem_bugs():
    with open("../Codeflaws/single_function_repair.json") as f:
        data = json.load(f)
    ret = {}
    repeats = {}
    for bug, x in data.items():
        bug_id = bug.split("-bug-")[0]
        if len(x['tests']) > 0 and len(x['regression_tests']) > 0:
            if bug_id not in repeats:
                repeats[bug_id] = 1
                ret[bug] = x

    print(len(ret))
    with open("../Codeflaws/no_unique_single_function_repair.json", "w") as f:
        json.dump(ret, f)


def get_single_hunk_sinlge_function_bugs():
    with open("../Codeflaws/no_unique_single_function_repair.json", "r") as f:
        ret = json.load(f)
    single_hunk_ret = {}
    for bug, repair in ret.items():
        with open("buggy.c", "w") as f:
            f.write(repair['buggy'])
        with open("fixed.c", "w") as f:
            f.write(repair['fix'])
        diff_output = subprocess.getoutput("diff buggy.c fixed.c | grep '^[1-9]'")
        if len(diff_output.splitlines()) > 1:
            continue

        change = diff_output.splitlines()[0]
        if 'a' in change:
            lin_num = int(change.split("a")[0])
            prefix = repair['buggy'].splitlines()[:lin_num]
            suffix = repair['buggy'].splitlines()[lin_num:]
        elif 'c' in change:
            lin_num = change.split("c")[0]
            if "," in lin_num:
                hunk_start = int(lin_num.split(",")[0])
                hunk_end = int(lin_num.split(",")[1])
            else:
                hunk_start = int(lin_num)
                hunk_end = int(lin_num)
            prefix = repair['buggy'].splitlines()[:hunk_start-1]
            suffix = repair['buggy'].splitlines()[hunk_end:]
        elif 'd' in change:
            lin_num = change.split("d")[0]
            if "," in lin_num:
                hunk_start = int(lin_num.split(",")[0])
                hunk_end = int(lin_num.split(",")[1])
            else:
                hunk_start = int(lin_num)
                hunk_end = int(lin_num)
            prefix = repair['buggy'].splitlines()[:hunk_start-1]
            suffix = repair['buggy'].splitlines()[hunk_end:]
        else:
            assert False

        prefix = "\n".join(prefix)
        suffix = "\n".join(suffix)
        single_hunk_ret[bug] = repair
        single_hunk_ret[bug]['prefix'] = prefix
        single_hunk_ret[bug]['suffix'] = suffix

    print(len(single_hunk_ret))
    with open("../Codeflaws/single_hunk_single_function_repair.json", "w") as f:
        json.dump(single_hunk_ret, f)


def main():
    result = {}
    for folder in glob.glob("../../codeflaws/*/"):
        bug = folder.split("/")[-2]
        bug_id = bug.split("-bug-")[0]
        buggy_c = bug.split("-bug-")[-1].split("-")[0]
        fix_c = bug.split("-bug-")[-1].split("-")[1]
        with open(folder+bug_id+"-"+buggy_c+".c", "r") as f:
            buggy = f.read()
        with open(folder + bug_id + "-" + fix_c + ".c", "r") as f:
            fix = f.read()

        output = subprocess.getoutput(
            "ctags --options=NONE -f - -u -DX --excmd=n --kinds-c=f --fields=ne " + folder + bug_id + "-" + buggy_c + ".c")
        buggy_function_dict = get_function_start_end(output)
        output = subprocess.getoutput(
            "ctags --options=NONE -f - -u -DX --excmd=n --kinds-c=f --fields=ne " + folder + bug_id + "-" + fix_c + ".c")
        fixed_function_dict = get_function_start_end(output)
        diff_output = subprocess.getoutput("diff " + folder + bug_id + "-" + buggy_c + ".c " + folder + bug_id + "-" + fix_c + ".c" + " | grep '^[1-9]'")
        buggy_functions = set()
        fixed_functions = set()
        b_found = False
        t_found = False
        buggyflag = False
        fixedflag = False

        for change in diff_output.splitlines():
            if 'a' in change:
                b, f = change.split("a")
            elif 'c' in change:
                b, f = change.split("c")
            elif 'd' in change:
                b, f = change.split("d")
            else:
                assert False
            b_found = False
            t_found = False
            for line in b.split(","):
                for function in buggy_function_dict:
                    if buggy_function_dict[function][0] <= int(line) <= buggy_function_dict[function][1]:
                        buggy_functions.add(function)
                        b_found = True

            for line in f.split(","):
                for function in fixed_function_dict:
                    if fixed_function_dict[function][0] <= int(line) <= fixed_function_dict[function][1]:
                        fixed_functions.add(function)
                        t_found = True
            if not b_found or not t_found:
                break

        if not b_found or not t_found:
            print("Not found")
            continue

        if len(buggy_functions) == 1 and len(fixed_functions) == 1:
            start, end = buggy_function_dict[buggy_functions.pop()]
            buggy_function = "\n".join(buggy.splitlines()[start - 1:end])
            f_start, f_end = fixed_function_dict[fixed_functions.pop()]
            fix_function = "\n".join(fix.splitlines()[f_start - 1:f_end])
            print(get_unified_diff(buggy_function, fix_function))
            result[bug+".c"] = {
                'buggy': buggy_function,
                'start': start,
                'end': end,
                'fix': fix_function
            }
            with open("../Codeflaws/buggy_programs/" + bug + ".c", "w") as f:
                f.write(buggy)
            with open("../Codeflaws/correct_programs/" + bug + ".c", "w") as f:
                f.write(fix)

        with open("../Codeflaws/single_function_repair.json", "w") as f:
            json.dump(result, f)


def clean_parse_codeflaws(folder):
    ret = {}
    with open(folder + "Codeflaws/no_unique_single_function_repair.json") as f:
        data = json.load(f)
    for bug, v in data.items():
        if len(v['tests']) > 0 and len(v['regression_tests']) > 0:
            ret[bug] = v
    print("Number of bugs: {}".format(len(ret)))
    return ret


def clean_parse_codeflaws_single_hunk(folder):
    with open(folder + "Codeflaws/single_hunk_single_function_repair.json") as f:
        data = json.load(f)
    print("Number of bugs: {}".format(len(data)))
    return data


def check_num_data():
    with open("../Codeflaws/single_function_repair.json") as f:
        result = json.load(f)
    print("Number of bugs: {}".format(len(result)))
    with open("../Codeflaws/no_unique_single_function_repair.json") as f:
        result = json.load(f)
    print("Number of bugs: {}".format(len(result)))
    with open("../Codeflaws/single_hunk_single_function_repair.json") as f:
        result = json.load(f)
    print("Number of bugs: {}".format(len(result)))


if __name__ == "__main__":
    # main()
    # run_all_test()
    # single_problem_bugs()
    # get_single_hunk_sinlge_function_bugs()
    check_num_data()