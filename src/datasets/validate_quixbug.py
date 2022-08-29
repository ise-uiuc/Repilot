import subprocess
import json
import glob


def validate_all_patches(folder, j_file):
    with open(folder + "/" + j_file, "r") as f:
        repair_dict = json.load(f)

    plausible = 0
    total = 0

    for file in sorted(glob.glob(folder + "/*.py")):
        current_file = "_".join(file.split('/')[-1].split("_")[0:-1])
        if ".py" not in current_file:
            current_file = current_file + ".py"
            try:
                index = int(file.split('/')[-1].split("_")[-1].split(".")[0])
            except:
                current_file = file.split('/')[-1]
                index = 0
        else:
            index = 0

        print(current_file, index)
        print(repair_dict[current_file][index]['diff'])
        if len(repair_dict[current_file]) <= index:
            print("Error: {}".format(file))
            continue

        # if repair_dict[current_file][index]["finish_reason"] != "stop":
        #     continue
        if repair_dict[current_file][index]['diff'] == "":
            continue

        bug = current_file.split(".py")[0]
        exit_code = subprocess.run("cd ../QuixBugs; python python_tester.py --bug {} --file {}/{} --add_pf"
                               .format(bug, folder, file.split("/")[-1]), shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
    valid_entropy = []
    invalid_entropy = []

    for bug, bug_dict in repair_dict.items():
        if len(bug_dict) >= 1:
            total += 1
            for patch in bug_dict:
                if patch['valid']:
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
            for patch in bug_dict:
                if patch['valid']:
                    valid_entropy.append(patch['entropy'][0])
                    valid = True
                # else:
                #     invalid_entropy.append(patch['entropy'][0])

            if valid:
                for patch in bug_dict:
                    if not patch['valid'] and patch['diff'] != "":
                        invalid_entropy.append(patch['entropy'][0])
            num_correct = 0
            total_num = 0
            for patch in bug_dict:
                if patch['valid']:
                    num_correct += patch['num']
                total_num += patch['num']



            # valid = False
            # for index, patch in enumerate(bug_dict):
            #     if patch['valid']:
            #         # print(index)
            #         # print(patch['diff'])
            #         valid = True
            #
            # if valid is False:
            #     print(bug)

    print(count)
    print(first_10)
    print(first_try)
    print(total)
    # import statistics as st
    # plt.figure(figsize=(10, 10))
    # plt.hist(valid_entropy, bins=100, alpha=0.5, label='valid', weights=np.ones_like(valid_entropy) / float(len(valid_entropy)),
    #          range=[0, 2])
    # plt.hist(invalid_entropy, bins=100, alpha=0.5, label='invalid', weights=np.ones_like(invalid_entropy) / float(len(invalid_entropy)),
    #          range=[0, 2])
    # plt.legend(loc='upper right')
    # plt.show()
    # print(st.mean(valid_entropy))
    # print(st.mean(invalid_entropy))

if "__main__" == __name__:
    validate_all_patches("../Repair/LM/Results/quixbugs-gpt-neox-regular-200", "lm_repair.json")
    which_patches_are_good("../Repair/LM/Results/quixbugs-gpt-neox-regular-200/lm_repair.json")
