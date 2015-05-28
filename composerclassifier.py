import CTprocessing
import csv
import numpy as np
import operator
import copy
import pdb
import traceback
from sklearn import svm, cross_validation, dummy, naive_bayes, ensemble
from sklearn.feature_selection import VarianceThreshold
from enviar_email import *

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
    realminimum = int(infocomps[-1][1]) #Number of works of the last composer of the list

    infocomps = sorted(infocomps, key=operator.itemgetter(0), reverse=False)

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

    run = []
    np.random.seed(0)


    #### Seed random!
    for runn in range(numRuns):

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
 
    for runn in info:
        runn['selector'] = VarianceThreshold()
        runn['data'] = runn['selector'].fit_transform(runn['data'])
        runn['feature_names'] = np.array(runn['feature_names'])[runn['selector'].get_support()]

    print 'Features Selected'
    return info

def classify(info, numFolds):

    results_dum = np.empty([0,numFolds])
    results_skv = np.empty([0,numFolds])
    results_mnb = np.empty([0,numFolds])
    results_rf = np.empty([0,numFolds])


    #Feature Selection
    info = featureSelection(copy.deepcopy(info))

    #CROSS VALIDATION
    for runn in info:
        results_run_dum = np.array(())
        results_run_skv = np.array(())
        results_run_mnb = np.array(())
        results_run_rf = np.array(())
        ### Creation of cross validation folds & classification
        skf = cross_validation.StratifiedKFold(runn['target'], n_folds=numFolds)
        for train_index, test_index in skf:
            dum = []
            dum = dummy.DummyClassifier(strategy='most_frequent')
            dum.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_dum = np.append(results_run_dum,dum.score(runn['data'][test_index], runn['target'][test_index]))
            skv = []
            skv = svm.LinearSVC(penalty='l1', C=1, dual=False)
            skv.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_skv = np.append(results_run_skv,skv.score(runn['data'][test_index], runn['target'][test_index]))            
            mnb = []
            mnb = naive_bayes.MultinomialNB()
            mnb.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_mnb = np.append(results_run_mnb,mnb.score(runn['data'][test_index], runn['target'][test_index]))
            rf = []
            rf = ensemble.RandomForestClassifier()
            rf.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_rf = np.append(results_run_rf,rf.score(runn['data'][test_index], runn['target'][test_index]))
            print 'Classification done'
        results_dum = np.vstack((results_dum,results_run_dum))
        results_skv = np.vstack((results_skv,results_run_skv))        
        results_mnb = np.vstack((results_mnb,results_run_mnb))        
        results_rf = np.vstack((results_rf,results_run_rf))        


    return results_dum, results_skv, results_mnb, results_rf

"""
res_100_dum, res_100_skv, res_100_mnb, res_100_rf = classify(composerdivision(100, 'more_100', 10), 10)
res_50_dum, res_50_skv, res_50_mnb, res_50_rf = classify(composerdivision(50, 'more_50', 10), 10)
res_20_dum, res_20_skv, res_20_mnb, res_20_rf = classify(composerdivision(20, 'more_20', 10), 10)

np.savetxt('Results_more_100_dum.txt',res_100_dum)
np.savetxt('Results_more_100_skv.txt',res_100_skv)
np.savetxt('Results_more_100_mnb.txt',res_100_mnb)
np.savetxt('Results_more_100_rf.txt',res_100_rf)

np.savetxt('Results_more_50_dum.txt',res_50_dum)
np.savetxt('Results_more_50_skv.txt',res_50_skv)
np.savetxt('Results_more_50_mnb.txt',res_50_mnb)
np.savetxt('Results_more_50_rf.txt',res_50_rf)

np.savetxt('Results_more_20_dum.txt',res_20_dum)
np.savetxt('Results_more_20_skv.txt',res_20_skv)
np.savetxt('Results_more_20_mnb.txt',res_20_mnb)
np.savetxt('Results_more_20_rf.txt',res_20_rf)
"""


try:
    for i in range(20,120,5):
        str1 = 'more_' + str(i)
        strdum = 'Results_more_' + str(i) + '_dum.txt'
        strskv = 'Results_more_' + str(i) + '_skv.txt'
        strmnb = 'Results_more_' + str(i) + '_mnb.txt'
        strrf = 'Results_more_' + str(i) + '_rf.txt'
        res_dum, res_skv, res_mnb, res_rf = classify(composerdivision(i,str1,10), 10)
        np.savetxt(strdum,res_dum)
        np.savetxt(strskv,res_skv)
        np.savetxt(strmnb,res_mnb)
        np.savetxt(strrf,res_rf)

    for i in (120,200,240,490):
        str1 = 'more_' + str(i)
        strdum = 'Results_more_' + str(i) + '_dum.txt'
        strskv = 'Results_more_' + str(i) + '_skv.txt'
        strmnb = 'Results_more_' + str(i) + '_mnb.txt'
        strrf = 'Results_more_' + str(i) + '_rf.txt'
        res_dum, res_skv, res_mnb, res_rf = classify(composerdivision(i,str1,10), 10)
        np.savetxt(strdum,res_dum)
        np.savetxt(strskv,res_skv)
        np.savetxt(strmnb,res_mnb)
        np.savetxt(strrf,res_rf)

    send_email('Classificacions correctes','MEEEEEEEEEEEEEEL')
except:
    err = traceback.format_exc()
    print err
    send_email('Error des collons','Qualque cosa ha anat una puta merda: ' + err)
