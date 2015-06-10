import numpy as np


PATH = '/Users/Xesc/Dropbox/Master/Projecte/git/8Juny/'
CLASSIFIERS = ['dum', 'rf', 'svm']
FEATURE_IMP = ['rf_feat_imp', 'svm_coef']
FILENAME = '8Juny/Resultats_8Juny_Features_20works.csv'

##### Classifier accuracies

"""
RANGE = []
RANGE.insert(0,4)
RANGE.insert(0,3)
#RANGE.append(200)
#RANGE.append(240)
#RANGE.append(490)
"""
"""
RANGE = []
RANGE.append(50)

RES = []

for clas in CLASSIFIERS:
	line = []
	for num in RANGE:
		tmpMat = np.loadtxt(PATH + 'Results_more_' + str(num) + '_' + clas + '.txt')
		line.append(np.around(np.mean(tmpMat), decimals=10))
	RES.append(line)
	line = []
	for num in RANGE:
		tmpMat = np.loadtxt(PATH + 'Results_more_' + str(num) + '_' + clas + '.txt')
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
"""

###### Feature importances

RANGE = []
RANGE.append(20)

RES = []

for clas in FEATURE_IMP:
	line = []
	for num in RANGE:
		tmpMat = np.loadtxt(PATH + 'Results_more_' + str(num) + '_' + clas + '.txt')
		tmpMat = np.around(np.sum(tmpMat, axis = 0), decimals=10)
		for val in tmpMat:
			line.append(val)
	RES.append(line)

header = ['Classifier/Feature_imp']

header = np.hstack((header, np.loadtxt(PATH +'Feature_names_' + str(RANGE[0]) + '.txt', dtype='str', delimiter=',')))


first_col = []

for clas in FEATURE_IMP:
	first_col.append(clas)

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


