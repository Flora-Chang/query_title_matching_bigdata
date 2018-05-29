import os
import sys

input_file_path = sys.argv[1]
out_file_path = sys.argv[2]
index = 0
label = 2
with open(input_file_path, "r") as in_f, open(out_file_path, "w") as out_f:
    for line in in_f:
        line = line.strip().split("\t")
        query = line[3]
        title = line[4]
        query = query.strip("q[").strip("]").split()
        title = title.strip("d[").strip("]").split()
        out_f.write(str(index) + "\t" + " ".join(query) + "\t" + " ".join(title) + "\t" + str(label) + "\n")
        index += 1


