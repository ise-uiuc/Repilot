import glob
import json
import time
import javalang
import subprocess
import re
import os
import signal


def run_d4j_test(source, testmethods, bug_id):
    bugg = False
    compile_fail = False
    timed_out = False
    entire_bugg = False
    error_string = ""

    try:
        tokens = javalang.tokenizer.tokenize(source)
        parser = javalang.parser.Parser(tokens)
        parser.parse()
    except:
        print("Syntax Error")
        return compile_fail, timed_out, bugg, entire_bugg, True

    for t in testmethods:
        # print(t.strip())
        cmd = 'defects4j test -w %s/ -t %s' % (('/tmp/' + bug_id), t.strip())
        Returncode = ""
        error_file = open("stderr.txt", "wb")
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                 start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()  # child.stdout.read()
                print(b"".join(Returncode).decode('utf-8'))
                error_file.close()
                break
            elif Flag != 0 and Flag is not None:
                compile_fail = True
                error_file.close()
                with open("stderr.txt", "rb") as f:
                    r = f.readlines()
                for line in r:
                    if re.search(':\serror:\s', line.decode('utf-8')):
                        error_string = line.decode('utf-8')
                        break
                print(error_string)
                break
            elif time.time() - while_begin > 15:
                error_file.close()
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                timed_out = True
                break
            else:
                time.sleep(0.01)
        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            continue
        else:
            bugg = True
            break

    # Then we check if it passes all the tests, include the previously okay tests
    if not bugg:
        print('So you pass the basic tests, Check if it passes all the test, include the previously passing tests')
        cmd = 'defects4j test -w %s/' % ('/tmp/' + bug_id)
        Returncode = ""
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1,
                                 start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()  # child.stdout.read()
                break
            elif Flag != 0 and Flag is not None:
                bugg = True
                break
            elif time.time() - while_begin > 180:
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                bugg = True
                break
            else:
                time.sleep(0.01)
        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            print('success')
        else:
            entire_bugg = True

    return compile_fail, timed_out, bugg, entire_bugg, False


def validate_all_patches(folder, j_file, loc_folder):
    with open("../Defects4j" + "/single_function_repair.json", "r") as f:
        bug_dict = json.load(f)

    with open(folder + "/" + j_file, "r") as f:
        codex_dict = json.load(f)

    plausible = 0
    total = 0

    for file in sorted(glob.glob(folder + '/*.java')):
        current_file = file.split('/')[-1].split("_")[0]

        if ".java" not in current_file:
            current_file = current_file + ".java"
            index = int(file.split('/')[-1].split("_")[1].split(".")[0])-1
        else:
            index = 0
        if codex_dict[current_file][index]["finish_reason"] != "stop":
            continue
        if codex_dict[current_file][index]['diff'] == "":
            continue
        bug_id = current_file.split('.')[0]
        project = bug_id.split("-")[0]
        bug = bug_id.split("-")[1]
        start = bug_dict[bug_id]['start']
        end = bug_dict[bug_id]['end']
        tmp_bug_id = "test_" + bug_id

        print(file, bug_id)

        subprocess.run('rm -rf ' + '/tmp/' + tmp_bug_id, shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (project, bug + 'b', ('/tmp/' + tmp_bug_id)), shell=True)
        testmethods = os.popen('defects4j export -w %s -p tests.trigger' % ('/tmp/' + tmp_bug_id)).readlines()
        source_dir = os.popen("defects4j export -p dir.src.classes -w /tmp/" + tmp_bug_id).readlines()[-1].strip()

        with open(loc_folder + "/{}.buggy.lines".format(bug_id), "r") as f:
            locs = f.read()

        loc = set([x.split("#")[0] for x in locs.splitlines()])  # should only be one
        loc = loc.pop()
        try:
            with open(file, 'r') as f:
                patch = f.readlines()
        except:
            continue

        try:
            with open("/tmp/" + tmp_bug_id + "/" + source_dir + "/" + loc, 'r') as f:
                source = f.readlines()
        except:
            with open("/tmp/" + tmp_bug_id + "/" + source_dir + "/" + loc, 'r', encoding='ISO-8859-1') as f:
                source = f.readlines()

        source = "\n".join(source[:start - 1] + patch + source[end:])

        try:
            with open("/tmp/" + tmp_bug_id + "/" + source_dir + "/" + loc, 'w') as f:
                f.write(source)
        except:
            with open("/tmp/" + tmp_bug_id + "/" + source_dir + "/" + loc, 'w', encoding='ISO-8859-1') as f:
                f.write(source)

        compile_fail, timed_out, bugg, entire_bugg, syntax_error = run_d4j_test(source, testmethods, tmp_bug_id)
        if not compile_fail and not timed_out and not bugg and not entire_bugg and not syntax_error:
            plausible += 1
            codex_dict[current_file][index]['valid'] = True
            print("{} has valid patch: {}".format(bug_id, file))
        else:
            print("{} has invalid patch: {}".format(bug_id, file))

        total += 1
        subprocess.run('rm -rf ' + '/tmp/' + tmp_bug_id, shell=True)

    print("{}/{} are plausible".format(plausible, total))

    with open(folder + "/" + j_file, "w") as f:
        json.dump(codex_dict, f)


def which_patches_are_good(repair_result):
    with open(repair_result, "r") as f:
        codex_dict = json.load(f)
    count = 0
    total = 0
    first_try = 0
    for bug, bug_dict in codex_dict.items():
        if 'Time' in bug or 'Closure' in bug or 'Math' in bug or 'Mockito' in bug or 'Chart' in bug or 'Lang' in bug:
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


if "__main__" == __name__:
    # validate_all_patches("../Repair/LM/Results/defects4j", "lm_repair.json", "../Defects4j/location")
    which_patches_are_good("../Repair/LM/Results/defects4j/lm_repair.json")
