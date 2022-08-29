# Used for completion prompts
BASE_PROMPT = """# Provide a fix for the buggy function

# Buggy Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) - fibonacci(n-2)

# Fixed Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Provide a fix for the buggy function

# Buggy Function
{}

# Fixed Function
"""

VARY_BASE_PROMPT = """# Provide a fix for the buggy function

# Buggy Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) - fibonacci(n-2)

# Fixed Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Provide a fix for the buggy function

# Buggy Function
{example_bug}

# Fixed Function
{example_fix}

# Provide a fix for the buggy function

# Buggy Function
{bug}

# Fixed Function
"""

TESTCASE_BASE_PROMPT = """# Provide a fix for the buggy function

# Test Cases:
# input: 1 output: 1
# input: 4 output: 3
# input: 7 output: 13

# Buggy Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) - fibonacci(n-2)

# Fixed Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Provide a fix for the buggy function

# Test Cases:
{example_testcases}

# Buggy Function
{example_bug}

# Fixed Function
{example_fix}

# Provide a fix for the buggy function

# Test Cases:
{testcases}

# Buggy Function
{bug}

# Fixed Function
"""

ZERO_SHOT_PROMPT = """# Provide a fix for the buggy function

# Buggy Function
{bug}

# Fixed Function
"""

EXPERT_PROMPT = """# Provide a fix for the buggy function

# A buggy function:
{bug}

# A programmer patches the buggy function into:
"""

LONG_BASE_PROMPT = """# Provide a fix for the buggy function

# Buggy Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) - fibonacci(n-2)

# Fixed Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
        
# Provide a fix for the buggy function

# Buggy Function
def bubbleSort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] < arr[j + 1] :
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# Fixed Function
def bubbleSort(arr):
    n = len(arr)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j + 1] :
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                
# Provide a fix for the buggy function

# Buggy Function
def checkHasUpper(s):
    res = False
    for ele in s:
        if ele.islower():
            res = True
            break
    return res

# Fixed Function
def checkHasUpper(s):
    res = False
    for ele in s:
        if ele.isUpper():
            res = True
            break
    return res

# Provide a fix for the buggy function

# Buggy Function
{}

# Fixed Function
"""

LOCATION_PROMPT = """# Provide the location of the bug in the buggy function

# Buggy Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) - fibonacci(n-2)

# Location
Line: return fibonacci(n-1) - fibonacci(n-2)

# Provide the location of the bug in the buggy function

# Buggy Function
{}

# Location
Line: """


JAVA_BASE_PROMPT = """// Provide a fix for the buggy function

// Buggy Function
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r + l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Fixed Function
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Provide a fix for the buggy function

// Buggy Function
{}

// Fixed Function
"""


JAVA_VARY_PROMPT = """// Provide a fix for the buggy function

// Buggy Function
{example_bug}

// Fixed Function
{example_fix}

// Provide a fix for the buggy function

// Buggy Function
{bug}

// Fixed Function
"""

JAVA_LONG_VARY_PROMPT = """// Provide a fix for the buggy function

// Buggy Function
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r + l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Fixed Function
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Provide a fix for the buggy function

// Buggy Function
{example_bug}

// Fixed Function
{example_fix}

// Provide a fix for the buggy function

// Buggy Function
{bug}

// Fixed Function
"""


C_BASE_PROMPT = """/* Provide a fix for the buggy function */

/* Buggy Function */
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r + l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

/* Fixed Function */
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

/* Provide a fix for the buggy function */

/* Buggy Function */
{}

/* Fixed Function */
"""

C_VARY_PROMPT = """/* Provide a fix for the buggy function */

/* Buggy Function */
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r + l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

/* Fixed Function */
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

/* Provide a fix for the buggy function */

/* Buggy Function */
{example_bug}

/* Fixed Function */
{example_fix}

/* Provide a fix for the buggy function */

/* Buggy Function */
{bug}

/* Fixed Function */
"""


# Use for infilling prompts
INFILL_BASE_PREFIX = """# Provide a fix for the buggy function

# Buggy Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) - fibonacci(n-2)

# Fixed Function
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Provide a fix for the buggy function

# Buggy Function
{}

# Fixed Function
"""

INFILL_BASE_SUFFIX = """
# Provide a fix for the buggy function

# Buggy Function
{example_bug}

# Fixed Function
{example_fix}
"""


JAVA_INFILL_BASE_PREFIX = """// Provide a fix for the buggy function

// Buggy Function
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r + l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Fixed Function
int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Provide a fix for the buggy function

// Buggy Function
{}

// Fixed Function
"""

JAVA_INFILL_BASE_SUFFIX = """
// Provide a fix for the buggy function

// Buggy Function
{example_bug}

// Fixed Function
{example_fix}
"""


