import sys

if len(sys.argv) != 3:
    print("Usage: raw_code.cu transformed_code.cu")
    exit(0)

source_code_lines = open(sys.argv[1], "r").readlines()
f = open(sys.argv[2], "w")
index = 0
insert_code_template = """
  if (*preempted) return;
"""

def add_param(line):
    if (line.find("void") != -1) and (line.find("__global__") != -1):
        if line.find(";") != -1:
            line = line.replace(");", ", int* preempted);")
        else:
            line = line.replace(") {", ", int* preempted) {")
    return line

for line in source_code_lines:
    f.write(add_param(line))
    if line.find("void") != -1 and line.find(";") == -1:
        insert_code = insert_code_template.format(index=index)
        f.write(insert_code)
        index += 1
