import re
from collections import Counter

f = open("Chinese_dep.txt", encoding="utf-8").readlines()

counts = Counter()
for line in f:
    if line == "\n":
        continue
    line = line.rstrip().split("\t")
    pos = line[-1].split(" ")
    tok = line[2].split(" ")
    # M = re.search("M\[(.)-.\]", pos)
    classifier = ""
    classifier_index = 0
    for n in range(len(pos)):
        if "M" in pos[n]:
            classifier = tok[n]
            classifier_index = n
    # M_list = re.findall("M\[(..?)-..?\]", pos)
    # M_index = M_list[-1]
    # classifier = tok[int(M_index) - 1]
    # pos = pos.split(" ")
    # NN = re.search("\[(.*)-.*\]", pos[-1])
    # if len(tok) < int(NN.group(1)):
    #     continue
    # headNP = tok[int(NN.group(1)) - 1]
    headNP_index = len(tok) - 1
    headNP = tok[-1]

    if headNP not in counts.keys():
        counts[headNP] = Counter()
    counts[headNP][classifier] += 1

out = open("output_ChineseDep.txt", "w", encoding="utf-8")
for key in counts.keys():
    out.write("%s\t%s\n" %(key, counts[key]))
out.close()

