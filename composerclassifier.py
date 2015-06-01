import CTprocessing
import csv
import numpy as np
import operator
import copy
import pdb
import traceback
import sys
import random
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

def featureSelection(composers, info):
 
    all_data = np.empty([0,len(info['feature_names'])])
    for com in composers:
        all_data = np.vstack((all_data,com.data))
    
    info['selector'] = VarianceThreshold()
    all_data = info['selector'].fit_transform(all_data)
    info['feature_names'] = np.array(info['feature_names'])[info['selector'].get_support()]

    for com in composers:
        com.data = np.array(com.data)[:,info['selector'].get_support()]

    print 'Features Selected'
    return composers, info

def composerdivision(description, numRuns, mincomp='', numComposers='',fixedNumWorks=''):

    if not mincomp and not (numComposers and fixedNumWorks):
        sys.exit('ERROR: It needs as parameters mincomp, or numComposers and fixedNumWorks')

    info = dict()

    csvheader, body = CTprocessing.loadCSVinfo(FINALCSV)

    random.shuffle(body)

    if fixedNumWorks:
        mincomp = fixedNumWorks

    if numComposers:
        infocomps = loadcomposers(mincomp, COMP_INFO)
        random.shuffle(infocomps)
        infocomps = infocomps[:numComposers]
        if numComposers > len(infocomps):
            sys.exit('ERROR: There are only %d composers with %d works.' % (len(infocomps), fixedNumWorks))
    else:
        infocomps = loadcomposers(mincomp, COMP_INFO)


    if fixedNumWorks:
        realminimum = fixedNumWorks
    else:
        realminimum = min(val for val in infocomps) #Number of works of the last composer of the list

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
        elif composers[cont].numWorks < numRuns * realminimum:
            composers[cont].numWorks += 1
        else:
            continue

        tempvec = np.array(())
        ### Binary features
        tempvec = np.append(tempvec,np.loadtxt(row[BINARY_VEC_COL]))
        ### Histogram features
        tempvec = np.append(tempvec,np.loadtxt(row[HISTOGRAM_COL]))

        composers[cont].data = np.vstack((composers[cont].data,tempvec))

        files += 1
        print "Processed files " + str(files)

    #### Feature selection: Variance Threshold

    composers, info = featureSelection(composers, info)

    #### Create subspreading
    all_run = []

    #### Seed random!
    for runn in range(numRuns):

        info['data'] = np.empty([0,len(info['feature_names'])]) ######################
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

        all_run.append(copy.deepcopy(info))
        print 'Run done'
    pdb.set_trace()
    return all_run


def classify(info, numFolds):

    results_dum = np.empty([0,numFolds])
    results_svm = np.empty([0,numFolds])
    #results_mnb = np.empty([0,numFolds])
    results_rf = np.empty([0,numFolds])
    rf_feature_importances = np.empty([0,len(info['feature_names'])])

    dum = dummy.DummyClassifier(strategy='most_frequent')
    svm = svm.LinearSVC(penalty='l1', C=1, dual=False)
    #mnb = naive_bayes.MultinomialNB()
    rf = ensemble.RandomForestClassifier()
    
    #CROSS VALIDATION
    for runn in info:
        results_run_dum = np.array(())
        results_run_svm = np.array(())
        #results_run_mnb = np.array(())
        results_run_rf = np.array(())
        ### Creation of cross validation folds & classification
        skf = cross_validation.StratifiedKFold(runn['target'], n_folds=numFolds)
        for train_index, test_index in skf:
            dum.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_dum = np.append(results_run_dum,dum.score(runn['data'][test_index], runn['target'][test_index]))
            svm.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_svm = np.append(results_run_svm,svm.score(runn['data'][test_index], runn['target'][test_index]))            
            #mnb.fit(runn['data'][train_index],runn['target'][train_index])
            #results_run_mnb = np.append(results_run_mnb,mnb.score(runn['data'][test_index], runn['target'][test_index]))
            rf.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_rf = np.append(results_run_rf,rf.score(runn['data'][test_index], runn['target'][test_index]))
            rf_feature_importances = np.vstack((rf_feature_importances,rf.feature_importances_ / np.sum(rf.feature_importances_))) 
            print 'Classification done'
        results_dum = np.vstack((results_dum,results_run_dum))
        results_svm = np.vstack((results_svm,results_run_svm))        
        #results_mnb = np.vstack((results_mnb,results_run_mnb))        
        results_rf = np.vstack((results_rf,results_run_rf))        


    #return results_dum, results_svm, results_mnb, results_rf
    return results_dum, results_svm, results_rf, rf_feature_importances


"""
try:
    for i in range(106,1,-5):
        str1 = str(i) + '_composers_20_works'
        strdum = str(i) + '_composers_20_works_dum.txt'
        strsvm = str(i) + '_composers_20_works_svm.txt'
        strmnb = str(i) + '_composers_20_works_mnb.txt'
        strrf = str(i) + '_composers_20_works_rf.txt'
        res_dum, res_svm, res_mnb, res_rf = classify(composerdivision(str1,10,numComposers=i,fixedNumWorks=20), 10)
        np.savetxt('20_works/' + strdum,res_dum)
        np.savetxt('20_works/' + strsvm,res_svm)
        np.savetxt('20_works/' + strmnb,res_mnb)
        np.savetxt('20_works/' + strrf,res_rf)

    send_email('Classificacions correctes','20 Works acabat')
except:
    err = traceback.format_exc()
    print err
    send_email('Error des collons','Qualque cosa ha anat una puta merda: ' + err)


try:
    itr = range(47,3,-5)
    itr.append(4)
    for i in itr:
        str1 = str(i) + '_composers_50_works'
        strdum = str(i) + '_composers_50_works_dum.txt'
        strsvm = str(i) + '_composers_50_works_svm.txt'
        strmnb = str(i) + '_composers_50_works_mnb.txt'
        strrf = str(i) + '_composers_50_works_rf.txt'
        res_dum, res_svm, res_mnb, res_rf = classify(composerdivision(str1,10,numComposers=i,fixedNumWorks=50), 10)
        np.savetxt('50_works/' + strdum,res_dum)
        np.savetxt('50_works/' + strsvm,res_svm)
        np.savetxt('50_works/' + strmnb,res_mnb)
        np.savetxt('50_works/' + strrf,res_rf)

    send_email('Classificacions correctes','50 Works acabat')
except:
    err = traceback.format_exc()
    print err
    send_email('Error des collons','Qualque cosa ha anat una puta merda: ' + err)


try:
    itr = range(17,3,-3)
    itr.append(4)
    itr.append(3)
    for i in itr:
        str1 = str(i) + '_composers_100_works'
        strdum = str(i) + '_composers_100_works_dum.txt'
        strsvm = str(i) + '_composers_100_works_svm.txt'
        strmnb = str(i) + '_composers_100_works_mnb.txt'
        strrf = str(i) + '_composers_100_works_rf.txt'
        res_dum, res_svm, res_mnb, res_rf = classify(composerdivision(str1,10,numComposers=i,fixedNumWorks=100), 10)
        np.savetxt('100_works/' + strdum,res_dum)
        np.savetxt('100_works/' + strsvm,res_svm)
        np.savetxt('100_works/' + strmnb,res_mnb)
        np.savetxt('100_works/' + strrf,res_rf)

    send_email('Classificacions correctes','100 Works acabat')
except:
    err = traceback.format_exc()
    print err
    send_email('Error des collons','Qualque cosa ha anat una puta merda: ' + err)
"""
"""
try:
    for i in range(20,120,5):
        str1 = 'more_' + str(i)
        strdum = 'Results_more_' + str(i) + '_dum.txt'
        strsvm = 'Results_more_' + str(i) + '_svm.txt'
        strmnb = 'Results_more_' + str(i) + '_mnb.txt'
        strrf = 'Results_more_' + str(i) + '_rf.txt'
        res_dum, res_svm, res_mnb, res_rf = classify(composerdivision(i,str1,10), 10)
        np.savetxt(strdum,res_dum)
        np.savetxt(strsvm,res_svm)
        np.savetxt(strmnb,res_mnb)
        np.savetxt(strrf,res_rf)

    for i in (120,200,240,490):
        str1 = 'more_' + str(i)
        strdum = 'Results_more_' + str(i) + '_dum.txt'
        strsvm = 'Results_more_' + str(i) + '_svm.txt'
        strmnb = 'Results_more_' + str(i) + '_mnb.txt'
        strrf = 'Results_more_' + str(i) + '_rf.txt'
        res_dum, res_svm, res_mnb, res_rf = classify(composerdivision(i,str1,10), 10)
        np.savetxt(strdum,res_dum)
        np.savetxt(strsvm,res_svm)
        np.savetxt(strmnb,res_mnb)
        np.savetxt(strrf,res_rf)

    send_email('Classificacions correctes','MEEEEEEEEEEEEEEL')
except:
    err = traceback.format_exc()
    print err
    send_email('Error des collons','Qualque cosa ha anat una puta merda: ' + err)
"""

res_dum, res_svm, res_rf = classify(composerdivision('str1',10,numComposers=5,fixedNumWorks=20), 10)

