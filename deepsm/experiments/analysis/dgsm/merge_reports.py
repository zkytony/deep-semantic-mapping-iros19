import os

result = []
inserted = False
for filename in sorted(os.listdir("./")):
    if (filename.startswith("full-report")):
        with open(filename) as f:
            lines = f.readlines()
            
            if len(result) == 0:
                if not lines[0].startswith("trial_name"):
                    lines[0] = "trial_name," + lines[0]
                    inserted = True
                result.append(lines[0])

            # Get trial_name
            if inserted:
                trial_name = os.path.splitext(filename)[0][len("full-report-"):]
                result.append("%s," % trial_name + lines[1])
            else:
                result.append(lines[1])

with open('report-merged.csv', 'w') as f:
    for line in result:
        f.write(line)
