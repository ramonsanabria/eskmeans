
import os
from collections import defaultdict

experiment="hubert_base_ls960_average"
lan="mandarin"

path="./results/"
path_src = os.path.join(path, experiment, lan)
agregate_result = os.path.join(path, experiment, lan, "all.tdev")

dict_all=defaultdict(list)

for element in os.listdir(path_src):
    if(element == "all.tdev"):
        continue
    classname=""
    with open(os.path.join(path_src, element)) as speaker_file:
        for line in  speaker_file.readlines():
            if(not line.strip()):
                continue
            if("Class" in line):
                continue
            else:
                dict_all[line.strip().split()[0]].append(line.strip().split())
            
        
with open(agregate_result, "w") as outputfile:
    outputfile.write("Class 1\n")
    for key in dict_all.keys():
        for segment in dict_all[key]:
            outputfile.write(" ".join(segment)+"\n")
    outputfile.write("\n")
    
