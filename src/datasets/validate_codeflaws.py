import subprocess
import glob
import json


def run_test(folder, test, script):
    try:
        output = subprocess.getoutput("cd " + folder + "; timeout --kill-after=1 15 ./" + script + " " + test)
        print(output)
    except:
        print("Exception: {}".format(test))
        return False
    if "Accepted" not in output:
        return False
    else:
        return True


def validate_all_patches(folder, j_file):

    with open("../Codeflaws/no_unique_single_function_repair.json", "r") as f:
        bug_dict = json.load(f)

    with open(folder + "/" + j_file, "r") as f:
        repair_dict = json.load(f)

    plausible = 0
    total = 0

    for file in sorted(glob.glob(folder + "/*.c")):
        current_file = file.split('/')[-1].split("_")[0]
        if ".c" not in current_file:
            current_file = current_file + ".c"
            index = int(file.split('/')[-1].split("_")[1].split(".c")[0]) - 1
        else:
            index = 0

        if len(repair_dict[current_file]) <= index:
            continue

        if repair_dict[current_file][index]['finish_reason'] != "stop":
            continue
        if repair_dict[current_file][index]['diff'] == "":
            continue
        print(current_file, index)
        start = bug_dict[current_file]['start']
        end = bug_dict[current_file]['end']
        total += 1
        with open("../Codeflaws/buggy_programs/"+current_file, "r") as f:
            source = f.read().splitlines()

        source = "\n".join(source[:start - 1] + repair_dict[current_file][index]['output'].splitlines() + source[end:])
        bug_id = current_file.split("-bug-")[0]
        buggy_c = current_file.split("-bug-")[-1].split("-")[0]

        with open("../../codeflaws/"+current_file.split(".c")[0] + "/" + bug_id+"-"+buggy_c+".c", "w") as f:
            f.write(source)
        compile_output = subprocess.getoutput("cd " + "../../codeflaws/"+current_file.split(".c")[0] + "; make clean; make;")
        # print(compile_output)
        if "error" in compile_output:
            continue

        tests = bug_dict[current_file]['tests']
        success = True
        for test in tests:
            if not run_test("../../codeflaws/"+current_file.split(".c")[0], test, "test-genprog.sh"):
                success = False
                break
        if not success:
            continue

        regression_tests = bug_dict[current_file]['regression_tests']
        for test in regression_tests:
            if not run_test("../../codeflaws/"+current_file.split(".c")[0], test, "test-valid.sh"):
                success = False
                break
        if success:
            plausible += 1
            print("{} has valid patch: {}".format(current_file, file))
            repair_dict[current_file][index]['valid'] = True

    print("{}/{} are plausible".format(plausible, total))
    with open(folder + "/" + j_file, "w") as f:
        json.dump(repair_dict, f)


def check_all_patches(folder, j_file):
    with open(folder + "/" + j_file, "r") as f:
        repair_dict = json.load(f)

    count = 0
    total = 0
    first_try = 0
    for bug, bug_dict in repair_dict.items():
        if len(bug_dict) > 1:
            total += 1
            for patch in bug_dict:
                if patch['valid']:
                    print(bug)
                    count += 1
                    break
            if bug_dict[0]['valid']:
                first_try += 1
            for index, patch in enumerate(bug_dict):
                if patch['valid']:
                    print(index)
                    print(patch['diff'])
    print(count)
    print(first_try)
    print(total)


if __name__ == "__main__":
    # validate_all_patches("../Repair/LM/Results/codeflaws-incoder-6b-suffix", "lm_repair.json")
    check_all_patches("../Repair/LM/Results/codeflaws-incoder-6b-suffix", "lm_repair.json")

