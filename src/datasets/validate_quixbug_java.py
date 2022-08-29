import subprocess
import json
import glob


def validate_all_patches(folder, j_file):
    with open(folder + "/" + j_file, "r") as f:
        repair_dict = json.load(f)

    plausible = 0
    total = 0

    for file in sorted(glob.glob(folder + "/*.java")):
        current_file = "_".join(file.split('/')[-1].split("_")[0:-1])
        if ".java" not in current_file:
            current_file = current_file + ".java"
            try:
                index = int(file.split('/')[-1].split("_")[-1].split(".")[0])
            except:
                current_file = file.split('/')[-1]
                index = 0
        else:
            index = 0
        print(current_file, index)

        if len(repair_dict[current_file]) <= index:
            print("Error: {}".format(file))
            continue

        if repair_dict[current_file][index]["finish_reason"] != "stop":
            continue
        if repair_dict[current_file][index]['diff'] == "":
            continue

        bug = current_file.split(".java")[0]
        exit_code = subprocess.run("cd ../QuixBugs; python java_tester.py --bug {} --file {}/{} --add_pf"
                                   .format(bug.lower(), folder, file.split("/")[-1]), shell=True)
                                   #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if exit_code.returncode == 0:
            plausible += 1
            repair_dict[current_file][index]['valid'] = True
            print("{} has valid patch: {}".format(bug, file))
        else:
            print("{} has invalid patch: {}".format(bug, file))

        total += 1

    print("{}/{} patches are plausible".format(plausible, total))
    with open(folder + "/" + j_file, "w") as f:
        json.dump(repair_dict, f)

def which_patches_are_good(repair_result):
    with open(repair_result, "r") as f:
        repair_dict = json.load(f)

    count = 0
    total = 0
    first_try = 0
    first_10 = 0

    for bug, bug_dict in repair_dict.items():
        if len(bug_dict) >= 1:
            total += 1
            for patch in bug_dict:
                if patch['valid']:
                    # print(bug)
                    count += 1
                    break
            if bug_dict[0]['valid']:
                first_try += 1

            for index, patch in enumerate(bug_dict):
                if patch['valid']:
                    first_10 += 1
                    break
                if index == 9:
                    break

            valid = False
            for index, patch in enumerate(bug_dict):
                if patch['valid']:
                    # print(index)
                    # print(patch['diff'])
                    valid = True

            if valid is False:
                print(bug)

    print(count)
    print(first_10)
    print(first_try)
    print(total)

if "__main__" == __name__:
    validate_all_patches("../Repair/LM/Results/quixbugs-java-gpt-neox-200", "lm_repair.json")
    #which_patches_are_good("../Repair/LM/Results/quixbugs-java-gpt-1.3-200/lm_repair.json")

