# Adopted from https://github.com/LASER-UMASS/AutomatedRepairApplicabilityData/blob/master/AnnotationScripts
# /ManyBugsSpecific/getPatchComplexity.py
import json
import os
import subprocess
import tarfile
import re
import sys
import pandas
from difflib import unified_diff


def get_unified_diff(source, mutant):
    output = ""
    for line in unified_diff(source.split('\n'), mutant.split('\n'), lineterm=''):
        output += line + "\n"
    return output


def remove_comments(text):
    """ remove c-style comments.
        text: blob of text with comments (can include newlines)
        returns: text with comments removed
    """
    pattern = r"""
                            ##  --------- COMMENT ---------
           /\*              ##  Start of /* ... */ comment
           [^*]*\*+         ##  Non-* followed by 1-or-more *'s
           (                ##
             [^/*][^*]*\*+  ##
           )*               ##  0-or-more things which don't start with /
                            ##    but do end with '*'
           /                ##  End of /* ... */ comment
         |                  ##  -OR-  various things which aren't comments:
           (                ## 
                            ##  ------ " ... " STRING ------
             "              ##  Start of " ... " string
             (              ##
               \\.          ##  Escaped char
             |              ##  -OR-
               [^"\\]       ##  Non "\ characters
             )*             ##
             "              ##  End of " ... " string
           |                ##  -OR-
                            ##
                            ##  ------ ' ... ' STRING ------
             '              ##  Start of ' ... ' string
             (              ##
               \\.          ##  Escaped char
             |              ##  -OR-
               [^'\\]       ##  Non '\ characters
             )*             ##
             '              ##  End of ' ... ' string
           |                ##  -OR-
                            ##
                            ##  ------ ANYTHING ELSE -------
             .              ##  Anything other char
             [^/"'\\]*      ##  Chars which doesn't start a comment, string
           )                ##    or escape
    """
    regex = re.compile(pattern, re.VERBOSE | re.MULTILINE | re.DOTALL)
    noncomments = [m.group(2) for m in regex.finditer(text) if m.group(2)]

    return "".join(noncomments)


def remove_spaces(text):
    newtext = ""
    linelist = text.split('\n')
    for line in linelist:
        if len(line.strip()) > 0:
            newtext = newtext + line.strip() + "\n"
    return newtext


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


def main():
    if len(sys.argv) < 2:
        print("ERROR: Please provide path to ManyBugs scenarios")
        sys.exit()

    repoPath = str(sys.argv[1])  # path to ManyBugs scenarios
    scenarios = os.listdir(repoPath)
    filecount = {}
    single_file_bugs = {}
    data = pandas.read_csv('manybugs.csv', header=0)

    for index, value in enumerate(data['FileCount']):
        if value == 1:
            project = data['Project'][index]
            bug_id = data['DefectId'][index]
            single_file_bugs[project + bug_id] = 0

    result = {}

    for scenario in sorted(scenarios):
        scenario_list = scenario.split(".tar.gz")[0].split('-')
        project = scenario_list[0]
        defectid = scenario.split("-bug-")[1].split('.tar.gz')[0]
        if project + defectid not in single_file_bugs:
            continue
        print(project, defectid)

        buggyversion = scenario_list[len(scenario_list) - 2]
        fixedversion = scenario_list[len(scenario_list) - 1]
        scenariotar = tarfile.open(repoPath + scenario)
        buggyflag = False
        fixedflag = False
        changed_file_name = ""
        for filename in sorted(scenariotar.getnames()):
            if "/diffs/" in filename and ".c-" in filename and buggyversion in filename.split(".c-")[
                1]:  # fetch the buggy version of code and minimize it
                buggyflag = True
                tardiff = scenariotar.extractfile(filename)
                changed_file_name = filename.split("/diffs/")[1].split(".c-")[0] + ".c"
                try:
                    b_function_w_comments = tardiff.read().decode('utf-8')
                except:
                    b_function_w_comments = tardiff.read().decode('iso_8859_1')
                fh = open("buggy.c", "w+")
                fh.write(b_function_w_comments)
                fh.close()
                output = subprocess.getoutput(
                    "ctags --options=NONE -f - -u -DX --excmd=n --kinds-c=f --fields=ne buggy.c")
                o_buggy_function_dict = get_function_start_end(output)
                code_wo_comments = remove_comments(b_function_w_comments)
                code_wo_spaces = remove_spaces(code_wo_comments)
                fh = open("buggy.c", "w+")
                fh.write(code_wo_spaces)
                fh.close()
            if "/diffs/" in filename and ".c-" in filename and fixedversion in filename.split(".c-")[
                1]:  # fetch the fixed version of code and minimize it
                fixedflag = True
                tardiff = scenariotar.extractfile(filename)
                try:
                    f_function_w_comments = tardiff.read().decode('utf-8')
                except:
                    f_function_w_comments = tardiff.read().decode('iso_8859_1')
                fh = open("fixed.c", "w+")
                fh.write(f_function_w_comments)
                fh.close()
                output = subprocess.getoutput(
                    "ctags --options=NONE -f - -u -DX --excmd=n --kinds-c=f --fields=ne fixed.c")
                o_fix_function_dict = get_function_start_end(output)
                code_wo_comments = remove_comments(f_function_w_comments)
                code_wo_spaces = remove_spaces(code_wo_comments)
                fh = open("fixed.c", "w+")
                fh.write(code_wo_spaces)
                fh.close()
            if buggyflag is True and fixedflag is True:  # compute results using minimized files and store the results

                output = subprocess.getoutput(
                    "ctags --options=NONE -f - -u -DX --excmd=n --kinds-c=f --fields=ne buggy.c")
                buggy_function_dict = get_function_start_end(output)
                output = subprocess.getoutput(
                    "ctags --options=NONE -f - -u -DX --excmd=n --kinds-c=f --fields=ne fixed.c")
                fixed_function_dict = get_function_start_end(output)
                diff_output = subprocess.getoutput("diff buggy.c fixed.c | grep '^[1-9]'")
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
                if len(buggy_functions) == 1 and len(fixed_functions) == 1:
                    print("Single Function Bug")
                    start, end = o_buggy_function_dict[buggy_functions.pop()]
                    buggy_function = "\n".join(b_function_w_comments.splitlines()[start - 1:end])
                    f_start, f_end = o_fix_function_dict[fixed_functions.pop()]
                    fix_function = "\n".join(f_function_w_comments.splitlines()[f_start - 1:f_end])
                    print(get_unified_diff(buggy_function, fix_function))
                    result[project + "_" + defectid] = {
                        'buggy': buggy_function,
                        'start': start,
                        'end': end,
                        'fix': fix_function,
                        'filename': changed_file_name
                    }
                    with open("../ManyBugs/buggy_programs/" + project + "_" + defectid + ".c", "w") as f:
                        f.write(b_function_w_comments)
                    with open("../ManyBugs/correct_programs/" + project + "_" + defectid + ".c", "w") as f:
                        f.write(f_function_w_comments)

        scenariotar.close()

    with open("../ManyBugs/single_function_repair.json", "w") as f:
        json.dump(result, f)


def parse_manybugs(folder):
    with open(folder + "ManyBugs/single_function_repair.json") as f:
        data = json.load(f)
    return data


def clean_parse_manybugs(folder):
    r = parse_manybugs(folder)
    data = {}
    for key, value in r.items():
        data[key+".c"] = value
    return data


def clean_parse_manybugs_single_hunk(folder):
    r = parse_manybugs(folder)
    data = {}
    for key, value in r.items():
        diff_lines = get_unified_diff(value['buggy'], value['fix']).splitlines()
        remove, gap, add, single_hunk = False, False, False, True
        line_no = 0
        for line in diff_lines[2:]:
            if "@@" in line:
                rline = line.split(" ")[1][1:]
                line_no = int(rline.split(",")[0].split("-")[0])
            elif line.startswith("-"):
                if not remove:
                    start_line_no = line_no
                if gap:
                    single_hunk = False
                    break
                remove = True
                end_line_no = line_no
            elif line.startswith("+"):
                if not remove and not add:
                    start_line_no = line_no
                if not add:
                    end_line_no = line_no
                add = True
                if gap:
                    single_hunk = False
                    break
            else:
                if remove or add:
                    gap = True

            if not single_hunk:
                break
            line_no += 1

        if not single_hunk:
            # print("Not single hunk bug")
            pass
        else:
            data[key + ".c"] = value
            data[key + ".c"]['prefix'] = "\n".join(value['buggy'].splitlines()[:start_line_no - 2])
            data[key + ".c"]['suffix'] = "\n".join(value['buggy'].splitlines()[end_line_no - 2:])
            # print(data[key + ".c"]['suffix'])
    # print(len(data))
    return data

if __name__ == "__main__":
    # main()
    clean_parse_manybugs_single_hunk("../")
