import os

result = []
for filename in os.listdir("./"):
    if (filename.startswith("full-report")):
        with open(filename) as f:
            lines = f.readlines()
            if len(result) == 0:
                result.append(lines[0])
            result.append(lines[1])

with open('report-merged.csv', 'w') as f:
    for line in result:
        f.write(line)
