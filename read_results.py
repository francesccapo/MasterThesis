import numpy as np
import pdb


PATH = '/Users/Xesc/Dropbox/Master/Projecte/git/Results_3Juliol_groups/'
CLASSIFIERS = ['dum', 'rf', 'svm']
FEATURE_IMP = ['rf_feat_imp', 'svm_coef']
FEATURE_GROUPS = 'featureGroups.txt'
CONFMAT = ['rf', 'svm']



###### Classifier accuracies
def extractAccuracy(ran, path, classifiers, filename):
	res = []

	for clas in classifiers:
		line = []
		for num in ran:
			tmpMat = np.loadtxt(path + 'Results_more_' + str(num) + '_' + clas + '.txt')
			line.append(np.around(np.mean(tmpMat), decimals=10))
		res.append(line)
		line = []
		for num in ran:
			tmpMat = np.loadtxt(path + 'Results_more_' + str(num) + '_' + clas + '.txt')
			line.append(np.around(np.std(tmpMat), decimals=10))
		res.append(line)

	header = ['Classifier/Num_Composers']

	for num in ran:
		header.append(str(num))

	first_col = []

	for clas in classifiers:
		first_col.append(clas + '_mean')
		first_col.append(clas + '_std')

	for row in res:
		row.insert(0,first_col.pop(0))

	f = open(path + filename,'w')

	for col in range(len(header)-1):
		f.write(header[col] + ',')
	f.write(header[-1] + '\n')

	for line in res:
		for col in range(len(line)-1):
			f.write(str(line[col]) + ',')
		f.write(str(line[-1]) + '\n')

	f.close()


###### Feature importances
def extractFeatures(number, path, feature_imp, filename):
	firstheader = np.array(([' ','Classifier']))
	secondheader = np.array((['Feature_name', 'Feature_group']))


	for clas in feature_imp:
		firstheader = np.append(firstheader,clas)
		secondheader = np.append(secondheader,' ')

	mat = np.loadtxt(path +'Feature_names_' + str(number) + '.txt', dtype='str', delimiter=',')
	tmpMat = np.loadtxt(path +'Feature_groups_' + str(number) + '.txt', dtype='str', delimiter=',')
	mat = np.vstack((mat,tmpMat))



	for clas in feature_imp:
		tmpMat = np.loadtxt(path + 'Results_more_' + str(number) + '_' + clas + '.txt')
		tmpMat = np.around(np.sum(tmpMat, axis = 0), decimals=10)
		mat = np.vstack((mat,tmpMat))


	res = np.hstack((firstheader.reshape(len(firstheader),1),secondheader.reshape(len(secondheader),1), mat))

	res = np.transpose(res)

	f = open(path + filename,'w')

	for line in res:
		for col in range(len(line)-1):
			f.write(str(line[col]) + ',')
		f.write(str(line[-1]) + '\n')

	f.close()

	header = np.array(('Feature_imp/Classifier'))

	for clas in feature_imp:
		header = np.append(header, clas + '_group')

	mat = np.array(())
	groupsCounter = np.array((),dtype=int)

	tmpGroups = np.loadtxt(path +'Feature_groups_' + str(number) + '.txt', dtype='str', delimiter=',')
	previous = 'null'

	for it in range(len(tmpGroups)):
		if tmpGroups[it] != previous:
			groupsCounter = np.append(groupsCounter,it)
			mat = np.append(mat,tmpGroups[it])
			previous = tmpGroups[it]
	groupsCounter = np.append(groupsCounter,it+1)


	for clas in feature_imp:
		tmpMat = np.loadtxt(path + 'Results_more_' + str(number) + '_' + clas + '.txt')
		tmpMat = np.around(np.sum(tmpMat, axis = 0), decimals=10)
		tmpVec = np.array(())
		for group in range(len(groupsCounter)-1):
			tmpVec = np.append(tmpVec,np.sum(tmpMat[groupsCounter[group]:groupsCounter[group+1]]))
		mat = np.vstack((mat,tmpVec))

	res = np.hstack((header.reshape(len(header),1), mat))

	res = np.transpose(res)

	f = open(path + 'Groups_' + filename,'w')

	for line in res:
		for col in range(len(line)-1):
			f.write(str(line[col]) + ',')
		f.write(str(line[-1]) + '\n')

	f.close()



###### Confusion matrix creation
def extractConfMat(number, path, confmat, filename, runs='indep'):

	for clas in confmat:
		mat = np.loadtxt(path + 'Confusion_matrix_' + str(number) + '_' + clas + '.txt')
		numRuns = int(mat[0][0])
		mat = np.delete(mat,0,0)
		names = np.loadtxt(path + 'Confusion_matrix_names_' + str(number) + '.txt', dtype='str', delimiter=';')
		numNames = len(names[0])
		totMat = np.empty([0,numNames+1])
		if runs=='indep':
			for runn in range(numRuns):
				h_names = names[runn].reshape(1,numNames)
				h_names = np.append('Run_' + str(runn+1),h_names)
				v_names = names[runn].reshape(numNames,1)
				runMat = np.hstack((v_names,mat[runn*numNames:(runn+1)*numNames,:]))
				totMat = np.vstack((totMat,runMat,h_names))
		else:
			h_names = names[0].reshape(1,numNames)
			h_names = np.append('All_runs', h_names)
			v_names = names[0].reshape(numNames,1)
			runMat = np.zeros([numNames,numNames])
			for runn in range(numRuns):
				runMat += mat[runn*numNames:(runn+1)*numNames,:]
			runMat = runMat/numRuns
			runMat = np.hstack((v_names,runMat))
			totMat = np.vstack((totMat,runMat,h_names))		
		f = open(path + 'Class_' + clas + '_' + filename,'w')
		for line in totMat:
			for col in range(len(line)-1):
				f.write(str(line[col]) + ';')
			f.write(str(line[-1]) + '\n')
		f.close()


###### Classifier accuracies
def extractAccurGropus(ran, path, classifiers, groupsfile, filename):

	featureList = np.loadtxt(groupsfile, dtype=str)

	for works in ran:
		res = []
		for clas in classifiers:
			line = []
			for group in featureList:
				tmpMat = np.loadtxt(path  + group + '/' + 'Results_more_' + str(works) + '_' + clas + '.txt')
				line.append(np.around(np.mean(tmpMat), decimals=10))
			res.append(line)
			line = []
			for group in featureList:
				tmpMat = np.loadtxt(path  + group + '/' + 'Results_more_' + str(works) + '_' + clas + '.txt')
				line.append(np.around(np.std(tmpMat), decimals=10))
			res.append(line)

		header = ['Classifier/Num_Composers (' + str(works) + ' works)']

		for group in featureList:
			header.append(group)

		first_col = []

		for clas in classifiers:
			first_col.append(clas + '_mean')
			first_col.append(clas + '_std')

		for row in res:
			row.insert(0,first_col.pop(0))

		f = open(path + str(works) + '_works_' + filename,'w')

		for col in range(len(header)-1):
			f.write(header[col] + ',')
		f.write(header[-1] + '\n')

		for line in res:
			for col in range(len(line)-1):
				f.write(str(line[col]) + ',')
			f.write(str(line[-1]) + '\n')

		f.close()






extractAccurGropus([20,50,100], PATH, CLASSIFIERS, FEATURE_GROUPS, 'Accuracies.csv')


#extractAccuracy([20,50,100], PATH, CLASSIFIERS, 'Accuracies.csv')

#extractFeatures(20, PATH, FEATURE_IMP, 'Features_20_works.csv')
#extractConfMat(20, PATH, CONFMAT, 'ConfMat_20_works.csv', runs='dep')
#extractFeatures(50, PATH, FEATURE_IMP, 'Features_50_works.csv')
#extractConfMat(50, PATH, CONFMAT, 'ConfMat_50_works.csv', runs='dep')
#extractFeatures(100, PATH, FEATURE_IMP, 'Features_100_works.csv')
#extractConfMat(100, PATH, CONFMAT, 'ConfMat_100_works.csv', runs='dep')

