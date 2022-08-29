import subprocess
import random
import torch
import numpy as np
import json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# run validation, TODO fix for quixbugs
def _run_validation(bug, patch_file, folder, patch, skip_val=True):
    try:
        with open(folder + "/" + patch_file, 'w') as f:
            f.write(patch)
    except:
        with open(folder + "/" + patch_file, 'w') as f:
            f.write("write error ... ")
        return False

    if skip_val:
        print("Skipping validation ... ")
        return False

    print("Validating patch ... ")
    exit_code = subprocess.run("cd QuixBugs; python python_tester.py --bug {} --file ../{}/{} --add_pf"
                               .format(bug, folder, patch_file), shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if exit_code.returncode == 0:
        print("Patch is valid")
        return True
    else:
        print("Patch is invalid")
        return False


def _get_relevant_bugs(bugs, current_bug, only_same):
    potential_pairs = []
    project = current_bug.split("-")[0]
    for file_name, bug in bugs.items():
        if file_name == current_bug:
            continue
        if file_name.startswith(project + "-") and only_same:
            potential_pairs.append((len(bug['buggy']) + len(bug['fix']), file_name))
        elif not only_same:
            potential_pairs.append((len(bug['buggy']) + len(bug['fix']), file_name))
    # sort from smallest to largest
    potential_pairs.sort(key=lambda x: x[0])
    return potential_pairs


# picking an example fix pairs from a project
def pick_smallest_example_fix(bugs, current_bug, only_same=False):
    potential_pairs = _get_relevant_bugs(bugs, current_bug, only_same)
    return bugs[potential_pairs[0][1]]['buggy'], bugs[potential_pairs[0][1]]['fix']


def pick_smallest_example_fix_name(bugs, current_bug, only_same=False):
    potential_pairs = _get_relevant_bugs(bugs, current_bug, only_same)
    return bugs[potential_pairs[0][1]]['buggy'], bugs[potential_pairs[0][1]]['fix'], potential_pairs[0][1]


graph_based = {"breadth_first_search": '''# nodef =  Node("F")
# nodee =  Node("E")
# noded =  Node("D")
# nodec =  Node("C", None, [nodef])
# nodeb =  Node("B", None, [nodee])
# nodea =  Node("A", None, [nodeb, nodec, noded])
# input: nodea, nodee output: True
# input: nodef, nodee output: False
# input nodef, nodef output: True''',
               "depth_first_search": '''# nodef =  Node("F")
# nodee =  Node("E")
# noded =  Node("D")
# nodec =  Node("C", None, [nodef])
# nodeb =  Node("B", None, [nodee])
# nodea =  Node("A", None, [nodeb, nodec, noded])
# input: nodea, nodee output: True
# input: nodef, nodee output: False
# input nodef, nodef output: True''',
               "detect_cycle": '''# node1 = Node(1)
# node2 = Node(2, node1)
# node3 = Node(3, node2)
# node4 = Node(4, node3)
# node5 = Node(5, node4)
# input: node5 output: False''',
               "minimum_spanning_tree": '''# input: {(1, 2): 10, (2, 3): 15, (3, 4): 10,(1, 4): 10} output: (1, 2) (3, 4) (1, 4)
# input: {(1, 2): 6, (1, 3): 1, (1, 4): 5, (2, 3): 5, (2, 5): 3, (3, 4): 5, (3, 5): 6, (3, 6): 4, (4, 6): 2, (5, 6): 6} output: (2, 5) (1, 3) (2, 3) (4, 6) (3, 6)
# input: {(1, 2): 6, (1, 3): 1, (2, 4): 2} output: (1, 2) (1, 3) (2, 4)''',
               "reverse_linked_list": '''# node1 = Node(1)
# node2 = Node(2, node1)
# node3 = Node(3, node2)
# node4 = Node(4, node3)
# node5 = Node(5, node4)
# input: node5 output: node1''',
               "shortest_path_length": '''# node1 = Node("1")
# node5 = Node("5")
# node4 = Node("4", None, [node5])
# node3 = Node("3", None, [node4])
# node2 = Node("2", None, [node1, node3, node4])
# node0 = Node("0", None, [node2, node5])
# length_by_edge = {(node0, node2): 3, (node0, node5): 10, (node2, node1): 1, (node2, node3): 2, (node2, node4): 4, (node3, node4): 1, (node4, node5): 1}
# input: length_by_edge, node0, node1 output: 4
# input: length_by_edge, node0, node5 output: 7
# input: length_by_edge, node2, node2 output: 0
# input: length_by_edge, node1, node5 output: INT_MAX''',
               "shortest_path_lengths":'''# graph = {(0, 2): 3, (0, 5): 5, (2, 1): -2, (2, 3): 7, (2, 4): 4, (3, 4): -5, (4, 5): -1}
# input: 6, graph output: (0, 0) 0 (1, 1) 0 (2, 2) 0 (3, 3) 0 (4, 4) 0 (5, 5) 0 (0, 2) 3 (0, 5) 4 (2, 1) -2 (2, 3) 7 (2, 4) 2 (3, 4) -5 (4, 5) -1 (0, 1) 1 (0, 3) 10 (0, 4) 5 (1, 0) inf (1, 2) inf (1, 3) inf (1, 4) inf (1, 5) inf (2, 0) inf (2, 5) 1 (3, 0) inf (3, 1) inf (3, 2) inf (3, 5) -6 (4, 0) inf (4, 1) inf (4, 2) inf (4, 3) inf (5, 0) inf (5, 1) inf (5, 2) inf (5, 3) inf (5, 4) inf''',
               "shortest_paths": '''# graph = {('A', 'B'): 3, ('A', 'C'): 3, ('A', 'F'): 5, ('C', 'B'): -2, ('C', 'D'): 7, ('C', 'E'): 4, ('D', 'E'): -5, ('E', 'F'): -1}
# input: 'A', 6 output: {'A': 0, 'C': 3, 'B': 1, 'E': 5, 'D': 10, 'F': 4}''',
               "topological_ordering": '''# five = Node(5)
# seven = Node(7)
# three = Node(3)
# eleven = Node(11)
# eight = Node(8)
# two = Node(2)
# nine = Node(9)
# ten = Node(10)
# five.outgoing_nodes = [eleven]
# seven.outgoing_nodes = [eleven, eight]
# three.outgoing_nodes = [eight, ten]
# eleven.incoming_nodes = [five, seven]
# eleven.outgoing_nodes = [two, nine, ten]
# eight.incoming_nodes = [seven, three]
# eight.outgoing_nodes = [nine]
# two.incoming_nodes = [eleven]
# nine.incoming_nodes = [eleven, eight]
# ten.incoming_nodes = [eleven, three]
# input: [five, seven, three, eleven, eight, two, nine, ten] output: 5 7 3 11 8 10 2 9'''
               }


def get_testcase(bug, folder):
    if bug not in graph_based:
        with open("{}/{}.json".format(folder, bug), "r") as f:
            cases = f.readlines()
        i_o_template = "# input: {} output: {}"
        str_builder = ""
        for case in cases:
            input, output = json.loads(case)
            if not isinstance(input, list):
                input = [input]
            output = [output]
            input_str = ", ".join([str(x) for x in input])
            output_str = ",".join([str(x) for x in output])
            if len(str_builder + i_o_template.format(input_str, output_str) + "\n") > 500:
                break
            str_builder += i_o_template.format(input_str, output_str) + "\n"
        return str_builder.strip()
    else:
        return graph_based[bug]



if __name__ == "__main__":
    print(get_testcase("gcd","../QuixBugs/json_testcases"))



PYTHON_PROMPT = """# Provide a fix for the buggy function

# Buggy Function
{example_bug}

# Fixed Function
{example_fix}
"""

JAVA_PROMPT = """// Provide a fix for the buggy function

// Buggy Function
{example_bug}

// Fixed Function
{example_fix}
"""


# use multiple examples to build prompt
def build_example_fixes(bugs, current_bug, model, only_same=False, language="python"):
    potential_pairs = _get_relevant_bugs(bugs, current_bug, only_same)
    if language == 'python':
        add_on = "\n# Provide a fix for the buggy function\n\n# Buggy Function {}\n\n# Fixed Function".format(bugs[current_bug]['buggy'])
    else:
        add_on = "\n// Provide a fix for the buggy function\n\n// Buggy Function {}\n\n// Fixed Function".format(bugs[current_bug]['buggy'])

    if language == 'python':
        prompt = PYTHON_PROMPT.format(example_bug = bugs[potential_pairs[0][1]]['buggy'], example_fix = bugs[potential_pairs[0][1]]['fix'])
    else:
        prompt = JAVA_PROMPT.format(example_bug = bugs[potential_pairs[0][1]]['buggy'], example_fix = bugs[potential_pairs[0][1]]['fix'])
    for i in range(1, len(potential_pairs)):
        if language == 'python':
            new_prompt = prompt + "\n" + PYTHON_PROMPT.format(example_bug = bugs[potential_pairs[i][1]]['buggy'], example_fix = bugs[potential_pairs[i][1]]['fix'])
        else:
            new_prompt = prompt + "\n" + JAVA_PROMPT.format(example_bug = bugs[potential_pairs[i][1]]['buggy'], example_fix = bugs[potential_pairs[i][1]]['fix'])

        if model.check_input(new_prompt+add_on, bugs[current_bug]['buggy']):
            prompt = new_prompt
        else:
            break
    return prompt + add_on
