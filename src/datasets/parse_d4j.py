import os
import subprocess
import javalang
import glob
import json

from datasets.check_duplicate import _get_hash

d4j_bug_lists = '''
| Chart           | jfreechart                 |       26       | 1-26                | None                    |
| Cli             | commons-cli                |       39       | 1-5,7-40            | 6                       |
| Closure         | closure-compiler           |      174       | 1-62,64-92,94-176   | 63,93                   |
| Codec           | commons-codec              |       18       | 1-18                | None                    |
| Collections     | commons-collections        |        4       | 25-28               | 1-24                    |
| Compress        | commons-compress           |       47       | 1-47                | None                    |
| Csv             | commons-csv                |       16       | 1-16                | None                    |
| Gson            | gson                       |       18       | 1-18                | None                    |
| JacksonCore     | jackson-core               |       26       | 1-26                | None                    |
| JacksonDatabind | jackson-databind           |      112       | 1-112               | None                    |
| JacksonXml      | jackson-dataformat-xml     |        6       | 1-6                 | None                    |
| Jsoup           | jsoup                      |       93       | 1-93                | None                    |
| JxPath          | commons-jxpath             |       22       | 1-22                | None                    |
| Lang            | commons-lang               |       64       | 1,3-65              | 2                       |
| Math            | commons-math               |      106       | 1-106               | None                    |
| Mockito         | mockito                    |       38       | 1-38                | None                    |
| Time            | joda-time                  |       26       | 1-20,22-27          | 21                      |'''


def parse_source(source):
    method_dict = {}
    tree = javalang.parse.parse(source)
    count = 0  # count use to break functions with the same name
    for path, node in tree:
        if type(node) == javalang.tree.MethodDeclaration or type(node) == javalang.tree.ConstructorDeclaration:
            method_dict[node.name+str(count)] = {'start': node.start_position.line, 'end': node.end_position.line}
            count += 1
    return method_dict


# testing parse source is correct
def test_javalang():
    with open("test.java", "r") as f:
        source = f.read()
    print(parse_source(source))


def get_single_hunk_single_function_bugs(loc_folder, dest_folder):
    single_hunk_ret = {}
    with open("../Defects4j" + "/single_function_repair.json", "r") as f:
        ret = json.load(f)
    count = 0
    for bug, repair in ret.items():
        with open(loc_folder + bug + ".buggy.lines", "r") as f:
            locs = f.read()

        line_numbers = [int(x.split("#")[1]) for x in locs.splitlines()]
        type = [x.split("#")[2] for x in locs.splitlines()]
        type_dict = {num : t for num, t in zip(line_numbers, type)}

        if sorted(line_numbers) == list(range(min(line_numbers), max(line_numbers)+1)):  # Consecutive lines
            lines = repair['buggy'].splitlines()
            print(bug)

            if type.count("FAULT_OF_OMISSION") == len(type):
                print("Fault of omission")
                hunk_start = min(line_numbers) - repair['start']
                prefix = "\n".join(lines[:hunk_start])
                suffix = "\n".join(lines[hunk_start:])
            else:
                start = None
                end = None
                for l in sorted(line_numbers):
                    if type_dict[l] != "FAULT_OF_OMISSION":
                        start = l
                        break
                for l in sorted(line_numbers, reverse=True):
                    if type_dict[l] != "FAULT_OF_OMISSION":
                        end = l
                        break
                hunk_start = start - repair['start']
                hunk_end = end - repair['end']
                prefix = "\n".join(lines[:hunk_start])
                suffix = "\n".join(lines[hunk_end:])
            single_hunk_ret[bug] = {
                'prefix': prefix,
                'suffix': suffix,
                'start': repair['start'],
                'end': repair['end'],
                'buggy': repair['buggy'],
                'fix': repair['fix']
            }
    print(count)
    with open(dest_folder + "/single_function_single_hunk_repair.json", "w") as f:  # write to file
        json.dump(single_hunk_ret, f)



def generate_d4j_patch_location(dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for l in d4j_bug_lists.splitlines():
        if not l.startswith('|'):
            continue
        project = l.split("|")[1].strip()
        bugs = l.split("|")[4].strip()
        for bug in bugs.split(','):
            if '-' in bug:
                start, end = bug.split("-")
                for i in range(int(start), int(end) + 1):
                    subprocess.run(
                        "bash scripts/get_fixed_lines.sh {} {} ~/llm_repair/{}".format(project, str(i), dest_folder),
                        shell=True)
            else:
                subprocess.run(
                    "bash scripts/get_fixed_lines.sh {} {} ~/llm_repair/{}".format(project, bug, dest_folder),
                    shell=True)


def generate_d4j_buggy_location(dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for l in d4j_bug_lists.splitlines():
        if not l.startswith('|'):
            continue
        project = l.split("|")[1].strip()
        bugs = l.split("|")[4].strip()
        for bug in bugs.split(','):
            if '-' in bug:
                start, end = bug.split("-")
                for i in range(int(start), int(end) + 1):
                    subprocess.run(
                        "bash scripts/get_buggy_lines.sh {} {} ~/llm_repair/{}".format(project, str(i), dest_folder),
                        shell=True)
            else:
                subprocess.run(
                    "bash scripts/get_buggy_lines.sh {} {} ~/llm_repair/{}".format(project, bug, dest_folder),
                    shell=True)


def generate_d4j_fix_function(loc_folder, dest_folder):

    new_ret = {}

    with open("Defects4j" + "/single_function_repair.json", "r") as f:  # write to file
        ret = json.load(f)

    for bug_id, value in ret.items():

        project = bug_id.split("-")[0]
        bug = bug_id.split("-")[1]
        with open(loc_folder + bug_id + ".fixed.lines", "r") as f:
            locs = f.read()

        loc = set([x.split("#")[0] for x in locs.splitlines()])
        if len(loc) > 1:
            print("{} has more than one buggy files".format(bug_id))
            assert False

        loc = loc.pop()
        line_numbers = [int(x.split("#")[1]) for x in locs.splitlines()]

        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
            project, bug + 'f', ('/tmp/' + bug_id)), shell=True)
        source_dir = os.popen("defects4j export -p dir.src.classes -w /tmp/" + bug_id).readlines()[-1].strip() + "/"

        try:
            with open("/tmp/" + bug_id + "/" + source_dir + "/" + loc, 'r') as f:
                source = f.read()
        except:
            with open("/tmp/" + bug_id + "/" + source_dir + "/" + loc, 'r', encoding='ISO-8859-1') as f:
                source = f.read()

        method_dict = parse_source(source)
        function_names = []
        for line_number in line_numbers:
            found = False
            for name, s in method_dict.items():
                if s['start'] <= line_number <= s['end']:
                    found = True
                    function_names.append(name)
            if found == False:
                print("{} is not in any function".format(line_number))
                break

        if not found:
            continue

        if len(set(function_names)) > 1:
            print(set(function_names))
            print("has more than one buggy functions")
            start = 1000000
            function_name = ""
            for f in set(function_names):
                if method_dict[f]['start'] < start:
                    start = method_dict[f]['start']
                    function_name = f
            function_names = [function_name]
        elif len(function_names) == 0:
            print("Does not belong in any function")
            assert False

        start = method_dict[function_names[0]]['start']
        end = method_dict[function_names[0]]['end']

        print("\n".join(source.splitlines()[start - 1:end]))
        new_ret[bug_id] = ret[bug_id]
        new_ret[bug_id]['fix'] = "\n".join(source.splitlines()[start-1:end])

        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)

    with open(dest_folder + "/single_function_repair.json", "w") as f:  # write to file
        json.dump(new_ret, f)


def generate_d4j_buggy_function(loc_folder, dest_folder):
    ret = {}
    for file in sorted(glob.glob(loc_folder + "/*.lines")):
        bug_id = file.split("/")[-1].split(".")[0]
        print(bug_id)
        project = bug_id.split("-")[0]
        bug = bug_id.split("-")[1]
        try:
            with open(file, "r") as f:
                locs = f.read()
        except:
            continue

        loc = set([x.split("#")[0] for x in locs.splitlines()])
        if len(loc) > 1:
            print("{} has more than one buggy files".format(bug_id))
            continue
        loc = loc.pop()
        line_numbers = [int(x.split("#")[1]) for x in locs.splitlines()]
        # Verifying that all locations are in the same function
        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
           project, bug + 'b', ('/tmp/' + bug_id)), shell=True)
        source_dir = os.popen("defects4j export -p dir.src.classes -w /tmp/" + bug_id).readlines()[-1].strip() + "/"

        try:
            with open("/tmp/" + bug_id + "/" + source_dir + "/" + loc, 'r') as f:
                source = f.read()
        except:
            with open("/tmp/" + bug_id + "/" + source_dir + "/" + loc, 'r', encoding='ISO-8859-1') as f:
                source = f.read()

        method_dict = parse_source(source)
        function_names = []
        for line_number in line_numbers:
            found = False
            for name, s in method_dict.items():
                if s['start'] <= line_number <= s['end']:
                    found = True
                    function_names.append(name)
            if found == False:
                print("{} is not in any function".format(line_number))
                break

        if not found:
            continue

        if len(set(function_names)) > 1:
            print("has more than one buggy functions")
            continue
        elif len(function_names) == 0:
            print("Does not belong in any function")
            continue

        start = method_dict[function_names[0]]['start']
        end = method_dict[function_names[0]]['end']

        print("\n".join(source.splitlines()[start-1:end]))
        ret[bug_id] = {'buggy': "\n".join(source.splitlines()[start-1:end]), "start": start, "end": end}
        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)

    with open(dest_folder + "/single_function_repair.json", "w") as f:  # write to file
        json.dump(ret, f)


def get_buggy_patch_file_hash(loc_folder, dest_folder):
    ret = {}
    for file in sorted(glob.glob(loc_folder + "/*.lines")):
        bug_id = file.split("/")[-1].split(".")[0]
        version = file.split("/")[-1].split(".")[1]
        print("Bug :{} {}".format(bug_id, version))
        project = bug_id.split("-")[0]
        bug = bug_id.split("-")[1]
        try:
            with open(file, "r") as f:
                locs = f.read()
        except:
            continue
        if bug_id not in ret:
            # Format for each bug we get hash of all files that were changed / fixed
            ret[bug_id] = {'fixed': [], 'buggy': []}

        loc = set([x.split("#")[0] for x in locs.splitlines()])

        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)
        subprocess.run("defects4j checkout -p %s -v %s -w %s" % (
            project, bug + version[0], ('/tmp/' + bug_id)), shell=True)
        source_dir = os.popen("defects4j export -p dir.src.classes -w /tmp/" + bug_id).readlines()[-1].strip() + "/"

        for file in loc:
            try:
                hash_v, _ = _get_hash("/tmp/" + bug_id + "/" + source_dir + file)
                print({'path':  source_dir + file, 'hash': hash_v})
                ret[bug_id][version].append({'path':  source_dir + file, 'hash': hash_v})
            except:
                pass

        subprocess.run('rm -rf ' + '/tmp/' + bug_id, shell=True)

    with open(dest_folder + "/file_hash.json", "w") as f:
        json.dump(ret, f)


def check_num_of_single_function_repairs():
    with open("Defects4j" + "/single_function_repair.json", "r") as f:
        ret = json.load(f)

    count = 0
    for bug, repair in ret.items():
        if 'Time' in bug or 'Closure' in bug or 'Math' in bug or 'Mockito' in bug or 'Chart' in bug or 'Lang' in bug:
            count += 1

    print(len(ret), count)


def clean_parse_d4j_single_hunk(folder):
    with open(folder + "Defects4j/single_function_single_hunk_repair.json", "r") as f:
        result = json.load(f)
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
        lines = v["prefix"].splitlines()
        cleaned_result[k + ".java"]["prefix"] = "\n".join([line[leading_white_space:] for line in lines])
        lines = v["suffix"].splitlines()
        cleaned_result[k + ".java"]["suffix"] = "\n".join([line[leading_white_space:] for line in lines])
        lines = v['fix'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"]["fix"] = "\n".join([line[leading_white_space:] for line in lines])
    return cleaned_result


def clean_parse_d4j(folder):
    with open("Defects4j/single_function_repair.json", "r") as f:
        result = json.load(f)
    cleaned_result = {}
    for k, v in result.items():
        lines = v['buggy'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"] = {"buggy": "\n".join([line[leading_white_space:] for line in lines])}
        lines = v['fix'].splitlines()
        leading_white_space = len(lines[0]) - len(lines[0].lstrip())
        cleaned_result[k + ".java"]["fix"] = "\n".join([line[leading_white_space:] for line in lines])
    return cleaned_result


if '__main__' == __name__:
    # generate_d4j_buggy_location('../Defects4j/location/')
    # generate_d4j_patch_location('../Defects4j/location/')
    # generate_d4j_buggy_function('Defects4j/location/', "../Defects4j")
    # generate_d4j_fix_function('Defects4j/location/', "../Defects4j")
    # test_javalang()
    # check_num_of_single_function_repairs()
    # get_buggy_patch_file_hash("../Defects4j/location/", "../Defects4j")
    get_single_hunk_single_function_bugs("../Defects4j/location/", "../Defects4j")