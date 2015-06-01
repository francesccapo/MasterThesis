import numpy as np


PATH = '/Users/Xesc/Dropbox/Master/Projecte/git/100_works/'
CLASSIFIERS = ['dum', 'mnb', 'rf', 'svm']
FILENAME = 'Resultats_100_works.csv'

RANGE = range(5,18,3)
RANGE.insert(0,4)
RANGE.insert(0,3)
#RANGE.append(200)
#RANGE.append(240)
#RANGE.append(490)

RES = []

for clas in CLASSIFIERS:
	line = []
	for num in RANGE:
		tmpMat = np.loadtxt(PATH + str(num) + '_composers_100_works_' + clas + '.txt')
		line.append(np.around(np.mean(tmpMat), decimals=10))
	RES.append(line)
	line = []
	for num in RANGE:
		tmpMat = np.loadtxt(PATH + str(num) + '_composers_100_works_' + clas + '.txt')
		line.append(np.around(np.std(tmpMat), decimals=10))
	RES.append(line)

header = ['Classifier/Num_Composers']

for num in RANGE:
	header.append(str(num))

first_col = []

for clas in CLASSIFIERS:
	first_col.append(clas + '_mean')
	first_col.append(clas + '_std')

for row in RES:
	row.insert(0,first_col.pop(0))

f = open(FILENAME,'w')

for col in range(len(header)-1):
	f.write(header[col] + ',')
f.write(header[-1] + '\n')

for line in RES:
	for col in range(len(line)-1):
		f.write(str(line[col]) + ',')
	f.write(str(line[-1]) + '\n')

f.close()

