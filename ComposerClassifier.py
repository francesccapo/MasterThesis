import CTprocessing
import csvprocessing
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
from sklearn.metrics import confusion_matrix
from enviar_email import *

FINALCSV = 'Information.csv'
COMP_INFO = 'Works_per_composer.csv'
RESULTS = 'Results/'
DATASET = '../kunstderfugue/'
FEATURES = 'featureGroups.txt'
COMPOSER_COL = 1
PATH_FILENAME_COL = 9

#features = ['binary_vec','histogram_vec']


class Composer():
    def __init__(self):
        self.name = []
        self.numWorks = 0
        self.maxWorks = []
        self.neededWorks = 0
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
            comp.append([row[0],int(row[1])])
    return comp

def featureSelection(composers, all_run):
 
    all_data = np.empty([0,np.shape(composers[0].data)[1]])
    for com in composers:
        all_data = np.vstack((all_data,com.data)) 


    selector = VarianceThreshold()
    all_data = selector.fit_transform(all_data)
    for runn in all_run:
        runn['feature_names'] = np.array(runn['feature_names'])[selector.get_support()]
        runn['feature_groups'] = np.array(runn['feature_groups'])[selector.get_support()]

    for com in composers:
        com.data = np.array(com.data)[:,selector.get_support()]

    print 'Features Selected'
    return composers, all_run

def composerdivision(featureFile, numRuns, mincomp='', numComposers='',fixedNumWorks=''):

    if not mincomp and not (numComposers and fixedNumWorks):
        sys.exit('ERROR: It needs as parameters mincomp, or numComposers and fixedNumWorks')

    info = dict()

    info['feature_names'] = np.array(())
    info['feature_groups'] = np.array(())

    featureList = np.loadtxt(featureFile, dtype=str)

    for feat in featureList:
        info['feature_names'] = np.hstack((info['feature_names'], np.loadtxt(DATASET + feat + '/' + 'feature_names.txt', dtype='str', delimiter=',')))
        info['feature_groups'] = np.hstack((info['feature_groups'],np.repeat(feat,info['feature_names'].shape[0]-info['feature_groups'].shape[0])))

    info['target_names'] = np.array(())
    info['target'] = np.array((),dtype=int)

    pdb.set_trace()

    csvheader, body = CTprocessing.loadCSVinfo(FINALCSV)

    random.shuffle(body)

    if fixedNumWorks:
        mincomp = fixedNumWorks

    infocomps = loadcomposers(mincomp, COMP_INFO)

    if numComposers:
        realminimum = fixedNumWorks
        if numComposers > len(infocomps):
            sys.exit('ERROR: There are only %d composers with %d works.' % (len(infocomps), fixedNumWorks))
    else:
        realminimum = min(val[1] for val in infocomps)

    composers = []
    namecomposers = []
    all_run = []
    
    for runn in range(numRuns):
        all_run.append(copy.deepcopy(info))
        if numComposers:
            random.shuffle(infocomps)
            tmpcomposers = infocomps[:numComposers]
        else:
            tmpcomposers = infocomps
        for comp in tmpcomposers:
            if not comp[0] in namecomposers:
                namecomposers.append(comp[0])
                composers.append(Composer())
                composers[-1].name = comp[0]
                composers[-1].maxWorks = comp[1]
                composers[-1].numTarget = len(composers)-1
                composers[-1].data = np.empty([0,len(info['feature_names'])])
                composers[-1].neededWorks += realminimum
                all_run[runn]['target_names'] = np.append(all_run[runn]['target_names'],composers[-1].name)
                all_run[runn]['target'] = np.append(all_run[runn]['target'],np.repeat(composers[-1].numTarget,realminimum))
            else:
                ind = namecomposers.index(comp[0])
                composers[ind].neededWorks += realminimum
                all_run[runn]['target_names'] = np.append(all_run[runn]['target_names'],composers[ind].name)
                all_run[runn]['target'] = np.append(all_run[runn]['target'],np.repeat(composers[ind].numTarget,realminimum))
        del(tmpcomposers)

    body = sorted(body, key=operator.itemgetter(COMPOSER_COL), reverse=False)
    previouscomp = 'null'
    files = 0

    for row in body:
        try:
            ind = namecomposers.index(row[COMPOSER_COL])
        except:
            continue
        if row[COMPOSER_COL] != previouscomp:
            previouscomp = row[COMPOSER_COL]
            composers[ind].numWorks = 1
        elif composers[ind].numWorks < composers[ind].neededWorks:
            composers[ind].numWorks += 1
        else:
            continue

        ### Loading features
        tempvec = np.array(())
        for feat in featureList:
            tempvec = np.append(tempvec, np.loadtxt(csvprocessing.newtxtpath(row[PATH_FILENAME_COL],feat)))
   
        composers[ind].data = np.vstack((composers[ind].data,tempvec))

        files += 1
        print "Processed files " + str(files)

    #### Feature selection: Variance Threshold
    composers, all_run = featureSelection(composers, all_run)

    #### Create subspreading

    #### Seed random!
    for runn in all_run:
        runn['data'] = np.empty([0,len(runn['feature_names'])])
        for comp in range(0,np.shape(runn['target'])[0],realminimum):
            if composers[runn['target'][comp]].pointer + realminimum <= composers[runn['target'][comp]].numWorks:
                runn['data'] = np.vstack((runn['data'],composers[runn['target'][comp]].data[composers[runn['target'][comp]].pointer:composers[runn['target'][comp]].pointer+realminimum]))
                if composers[runn['target'][comp]].pointer + realminimum == composers[runn['target'][comp]].numWorks:
                    composers[runn['target'][comp]].pointer = 0
                    np.random.shuffle(composers[runn['target'][comp]].data)
                else:
                    composers[runn['target'][comp]].pointer += realminimum
            else:
                runn['data'] = np.vstack((runn['data'],composers[runn['target'][comp]].data[composers[runn['target'][comp]].pointer:composers[runn['target'][comp]].numWorks]))
                np.random.shuffle(composers[runn['target'][comp]].data)
                runn['data'] = np.vstack((runn['data'],composers[runn['target'][comp]].data[:realminimum-(composers[runn['target'][comp]].numWorks-composers[runn['target'][comp]].pointer)]))
                composers[runn['target'][comp]].pointer = realminimum - (composers[runn['target'][comp]].numWorks-composers[runn['target'][comp]].pointer)

        print 'Run done'
    return all_run


def classify(info, numFolds):
    
    results_dum = np.empty([0,numFolds])
    results_svm = np.empty([0,numFolds])
    svm_coef = np.empty([0,len(info[0]['feature_names'])])
    svm_confmat = np.zeros((numFolds,len(info[0]['target_names']),len(info[0]['target_names'])))
    #results_mnb = np.empty([0,numFolds])
    results_rf = np.empty([0,numFolds])
    rf_feature_importances = np.empty([0,len(info[0]['feature_names'])])
    rf_confmat = np.zeros((numFolds, len(info[0]['target_names']),len(info[0]['target_names'])))
    compNames = np.empty([0,len(info[0]['target_names'])])


    dum_cla = dummy.DummyClassifier(strategy='most_frequent')
    svm_cla = svm.LinearSVC(penalty='l1', C=1, dual=False)
    #mnb_cla = naive_bayes.MultinomialNB()
    rf_cla = ensemble.RandomForestClassifier(n_estimators=20)

    runCounter = 0 
    
    #CROSS VALIDATION
    for runn in info:
        results_run_dum = np.array(())
        results_run_svm = np.array(())
        #results_run_mnb = np.array(())
        results_run_rf = np.array(())

        ### Creation of cross validation folds & classification
        skf = cross_validation.StratifiedKFold(runn['target'], n_folds=numFolds)
        foldCounter = 0

        for train_index, test_index in skf:
            dum_cla.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_dum = np.append(results_run_dum,dum_cla.score(runn['data'][test_index], runn['target'][test_index]))
            svm_cla.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_svm = np.append(results_run_svm,svm_cla.score(runn['data'][test_index], runn['target'][test_index]))
            svm_coef = np.vstack((svm_coef,np.mean([np.abs(val)/np.sum(np.abs(val)) for val in svm_cla.coef_],axis=0)))
            svm_confmat[runCounter] += confusion_matrix(runn['target'][test_index],svm_cla.predict(runn['data'][test_index]))
            #mnb_cla.fit(runn['data'][train_index],runn['target'][train_index])
            #results_run_mnb = np.append(results_run_mnb,mnb_cla.score(runn['data'][test_index], runn['target'][test_index]))
            rf_cla.fit(runn['data'][train_index],runn['target'][train_index])
            results_run_rf = np.append(results_run_rf,rf_cla.score(runn['data'][test_index], runn['target'][test_index]))
            rf_feature_importances = np.vstack((rf_feature_importances,rf_cla.feature_importances_ / np.sum(rf_cla.feature_importances_)))
            rf_confmat[runCounter] += confusion_matrix(runn['target'][test_index],rf_cla.predict(runn['data'][test_index]))


            foldCounter += 1

            print 'Fold number ' + str(foldCounter) + ' done'
            
        results_dum = np.vstack((results_dum,results_run_dum))
        results_svm = np.vstack((results_svm,results_run_svm))        
        #results_mnb = np.vstack((results_mnb,results_run_mnb))        
        results_rf = np.vstack((results_rf,results_run_rf))
        compNames = np.vstack((compNames, runn['target_names']))
        
        runCounter += 1 

        print 'Run number ' + str(runCounter) + ' done'

    svm_confmat = svm_confmat.astype(float)/(len(info[0]['target'])/len(info[0]['target_names'])) #Number of works normalization
    rf_confmat = rf_confmat.astype(float)/(len(info[0]['target'])/len(info[0]['target_names'])) #Number of works normalization

    svm_confmat = svm_confmat.reshape(svm_confmat.shape[0] * svm_confmat.shape[1], svm_confmat.shape[2])
    svm_confmat = np.vstack((np.repeat(runCounter, len(info[0]['target_names'])),svm_confmat))
    rf_confmat = rf_confmat.reshape(rf_confmat.shape[0] * rf_confmat.shape[1], rf_confmat.shape[2])
    rf_confmat = np.vstack((np.repeat(runCounter, len(info[0]['target_names'])),rf_confmat))


    #return results_dum, results_svm, results_mnb, results_rf
    return results_dum, results_svm, svm_coef, results_rf, rf_feature_importances, info[0]['feature_names'], info[0]['feature_groups'], svm_confmat, rf_confmat, compNames


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

try:
    for i in (20,50,100):
        strdum = 'Results_more_' + str(i) + '_dum.txt'
        strsvm = 'Results_more_' + str(i) + '_svm.txt'
        strsvm_coef = 'Results_more_' + str(i) + '_svm_coef.txt'
        strrf = 'Results_more_' + str(i) + '_rf.txt'
        strrf_feat_imp = 'Results_more_' + str(i) + '_rf_feat_imp.txt'
        str_feat_names = 'Feature_names_' + str(i) + '.txt'
        str_feat_groups = 'Feature_groups_' + str(i) + '.txt'
        str_svm_conmat = 'Confusion_matrix_' + str(i) + '_svm.txt'
        str_rf_conmat = 'Confusion_matrix_' + str(i) + '_rf.txt'
        str_conmatnames = 'Confusion_matrix_names_' + str(i) + '.txt'
        res_dum, res_svm, svm_coef, res_rf, rf_feat_imp, feat_names, feat_groups, svm_conmat, rf_conmat, conmatnames   = classify(composerdivision(FEATURES,10, mincomp=i), 10)
        np.savetxt(RESULTS + strdum,res_dum)
        np.savetxt(RESULTS + strsvm,res_svm)
        np.savetxt(RESULTS + strsvm_coef,svm_coef)
        np.savetxt(RESULTS + strrf,res_rf)
        np.savetxt(RESULTS + strrf_feat_imp,rf_feat_imp)
        np.savetxt(RESULTS + str_feat_names, feat_names, delimiter=',', fmt='%s')
        np.savetxt(RESULTS + str_feat_groups, feat_groups, delimiter=',', fmt='%s')
        np.savetxt(RESULTS + str_svm_conmat, svm_conmat, header= str(svm_conmat[0][0]) + ' Runs. The first row is related to the number of runs.')
        np.savetxt(RESULTS + str_rf_conmat, rf_conmat, header= str(rf_conmat[0][0]) + ' Runs. The first row is related to the number of runs.')
        np.savetxt(RESULTS + str_conmatnames, conmatnames, delimiter=';', fmt='%s')
        
        send_email('Classificacions correctes','MEEEEEEEEEEEEEEL: ' + str(i) + ' classificats!')
except:
    err = traceback.format_exc()
    print err
    send_email('Error des collons','Qualque cosa ha anat una puta merda: \n' + err)

print 'OK'

