import CTprocessing
import csv
import numpy as np
import operator
import copy
import pdb
from sklearn import svm, cross_validation
from sklearn.feature_selection import VarianceThreshold 

FINALCSV = 'Information.csv'
COMP_INFO = 'Works_per_composer.csv'
COMPOSER_COL = 1
BINARY_VEC_COL = 14
HISTOGRAM_COL = 15

class Composer():
    def __init__(self):
        self.name = []
        self.numWorks = []
        self.numTarget = []
        self.data = []
        self.pointer = 0

def loadcomposers(mincomp,comppath):
    comp = []
    with open(comppath,'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if int(row[1]) < mincomp:
                break
            comp.append(row)
    return comp


def composerdivision(mincomp, description, numRuns):

    info = dict()

    csvheader, body = CTprocessing.loadCSVinfo(FINALCSV)

    infocomps = loadcomposers(mincomp, COMP_INFO)

    info['target_names'] = np.array(())

    for i in infocomps:
        info['target_names'] = np.append(info['target_names'],i[0])

    info['DESCR'] = description
    info['feature_names'] = np.array(())

    ### Binary features
    for num in range(12):
        info['feature_names'] = np.append(info['feature_names'],'Binary_number_' + str(num+1))
    ### Histogram features
    for num in range(4096):
        info['feature_names'] = np.append(info['feature_names'],'Histogram_bin_' + str(num+1))

    composers = []

    cont = -1
    body = sorted(body, key=operator.itemgetter(COMPOSER_COL), reverse=False)
    previouscomp = 'null'
    files = 0

    for row in body:
        try:
            np.where(info['target_names'] == row[COMPOSER_COL])[0][0]
        except:
            continue
        if row[COMPOSER_COL] != previouscomp:
            cont += 1
            composers.append(Composer())
            previouscomp = row[COMPOSER_COL]
            composers[cont].name = row[COMPOSER_COL]
            composers[cont].numWorks = 1
            composers[cont].numTarget = cont
            composers[cont].data = np.empty([0,len(info['feature_names'])])
        else:
            composers[cont].numWorks += 1

        tempvec = np.array(())
        ### Binary features
        tempvec = np.append(tempvec,np.loadtxt(row[BINARY_VEC_COL]))
        ### Histogram features
        tempvec = np.append(tempvec,np.loadtxt(row[HISTOGRAM_COL]))

        composers[cont].data = np.vstack((composers[cont].data,tempvec))

        files += 1
        print "Processed files " + str(files)

    #### Create subspreading

    realminimum = int(infocomps[-1][1]) #Number of works of the last composer of the list


    run = []

    #### Seed random!
    for seed in range(numRuns):
        np.random.seed(seed)

        info['data'] = np.empty([0,len(info['feature_names'])])
        info['target'] = np.array((),dtype=int)

        for comp in composers:
            np.random.shuffle(comp.data)
            info['target'] = np.append(info['target'],np.repeat(comp.numTarget,realminimum))
            if comp.pointer + realminimum <= comp.numWorks:
                info['data'] = np.vstack((info['data'],comp.data[comp.pointer:comp.pointer+realminimum]))
                if comp.pointer + realminimum == comp.numWorks:
                    comp.pointer = 0
                    np.random.shuffle(comp.data)
                else:
                    comp.pointer += realminimum
            else:
                info['data'] = np.vstack((info['data'],comp.data[comp.pointer:comp.numWorks]))
                np.random.shuffle(comp.data)
                info['data'] = np.vstack((info['data'],comp.data[:realminimum-(comp.numWorks-comp.pointer)]))
                comp.pointer = realminimum - (comp.numWorks-comp.pointer)

        run.append(copy.deepcopy(info))
        print 'Run done'

    return run

def featureSelection(info):
 
    for run in info:
        run['selector'] = VarianceThreshold()
        run['data'] = run['selector'].fit_transform(run['data'])
        run['feature_names'] = np.array(run['feature_names'])[run['selector'].get_support()]

    print 'Features Selected'
    return info

def classify(info, numFolds):

    results = np.empty([0,numFolds])

    #Feature Selection
    info = featureSelection(copy.deepcopy(info))

    #CROSS VALIDATION
    for rin in info:
        results_run = np.array(())
        ### Creation of cross validation folds
        rin['folds'] = []
        skf = cross_validation.StratifiedKFold(rin['target'], n_folds=numFolds)
        for train_index, test_index in skf:
            tmpdict = dict()
            tmpdict['data_fold_train'], tmpdict['data_fold_test'] = rin['data'][train_index], rin['data'][test_index]
            tmpdict['target_fold_train'], tmpdict['target_fold_test'] = rin['target'][train_index], rin['target'][test_index]
            rin['folds'].append(tmpdict)

        ### Classify folds
        for fold in rin['folds']:
            fold['classifier'] = svm.LinearSVC(penalty='l1', C=1, dual=False)
            fold['classifier'].fit(fold['data_fold_train'],fold['target_fold_train'])
            fold['score'] = fold['classifier'].score(fold['data_fold_test'], fold['target_fold_test'])
            results_run = np.append(results_run,fold['score'])
            print 'Classification done'
        results = np.vstack((results,results_run))

    return info, results


more_100 = composerdivision(100, 'more_100', 10)
inf_100, res_100 = classify(more_100, 10)

more_50 = composerdivision(50, 'more_50', 10)
inf_50, res_50 = classify(more_50, 10)

more_20 = composerdivision(20, 'more_20', 10)
inf_20, res_20 = classify(more_20, 10)

np.savetxt('Results_more_100.txt',res_100)
np.savetxt('Results_more_50.txt',res_50)
np.savetxt('Results_more_20.txt',res_20)



