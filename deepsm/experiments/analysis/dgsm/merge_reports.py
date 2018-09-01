import os

result = []
for filename in sorted(os.listdir("./")):
    if (filename.startswith("full-report")):
        with open(filename) as f:
            lines = f.readlines()
            if len(result) == 0:
                if not lines[0].startswith("trial_name"):
                    lines.insert(0, "trial_name")
                result.append(lines[0])
            result.append(lines[1])

with open('report-merged.csv', 'w') as f:
    for line in result:
        f.write(line)
