import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# original label no.
org_value = '0';
# new label no.
modi_value = '1';

# read file in every file in the directory
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.txt")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    data = []

    # read each file content and put into data array
    file = open(title + '.txt', 'r')
    context = file.read()
    lines = context.split('\n')
    for line in lines:
        values = line.split(' ')
        if len(values)==5:
            # modified the selected label values
            if values[0] == org_value:
                values[0] = modi_value
                data.append(values)
    file.close()

    # write the file back with new label values
    file = open(title + '.txt', 'w')
    for l_idx in range(len(data)):
	file.write(data[l_idx][0]+" "+data[l_idx][1]+" "+data[l_idx][2]+" "+data[l_idx][3]+" "+data[l_idx][4])
    file.close()
