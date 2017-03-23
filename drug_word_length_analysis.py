
with open("drug_list.txt") as f:
    lines = f.readlines()

lengths = {}
for line in lines:
    line = line[:-1]  # remove newline-symbol
    drug = line.split("\t")[0]
    length = len(drug)
    if length not in lengths:
        lengths[length] = 1
    else:
        lengths[length] = lengths[length] + 1

total = 0
for length in sorted(list(lengths)):
    freq = lengths[length]
    print("%i\t%i" % (length, freq))
    total = total + freq
print("total\t%i" % total)
