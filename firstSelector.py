import sys
import operator
import re
import csv

### Argv loading
if len(sys.argv)!=6:
    print 'python '+sys.argv[0]+' <csv_filename> + <col_select> + <min_number> + <out_csv_filename> + <hist_name_csv>'
    sys.exit()

fn = sys.argv[1]
col_index = int(sys.argv[2])
min_number = int(sys.argv[3])
outfile_name = sys.argv[4]
hist_name = sys.argv[5]



### Load CSV

f = open(fn,'r')
lines = f.readlines()
f.close()

header = lines[0][:-1].split(',')
body = []
for line in range(1,len(lines)):
    tmpstr = lines[line].replace(",,",",_,")
    tmpstr = lines[line].replace(", ,",",_,")
    tmpstr = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', lines[line])
    tmpstr = [item.translate(None,'"').strip() for item in tmpstr]
    body.append(tmpstr)



### Sort by col_select and filter by

body = sorted(body, key=operator.itemgetter(col_index), reverse=False)

name_Temp = 'null'
cont = 0
new_body = []
comp_name = []
num_works = []

for line in range(len(body)):
    if body[line][col_index] != '':
        if body[line][col_index] == name_Temp:
            cont += 1
        else :
            if cont != 0 and cont >= min_number:
                for rewrite in range(cont):
                    new_body.append(body[line-1-rewrite])
                comp_name.append(name_Temp)
                num_works.append(cont)
            name_Temp = body[line][col_index]
            cont = 1

histog = zip(comp_name,num_works)
histog = sorted(histog, key=operator.itemgetter(1), reverse=True)


outfile = open(hist_name,'wb')
wr = csv.writer(outfile)
wr.writerows(histog)
outfile.close()

outfile = open(outfile_name,'wb')
wr = csv.writer(outfile)
wr.writerow(header)
wr.writerows(new_body)
outfile.close()


