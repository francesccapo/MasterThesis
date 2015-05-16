import CTprocessing
import csv
import numpy as np
import arff
import pdb


FINALCSV = 'Information.csv'
COMP_INFO = 'Works_per_composer.csv'
COMPOSER_COL = 1
BINARY_VEC_COL = 14
HISTOGRAM_COL = 15

def loadcomposers(mincomp,comppath):
    comp = []
    with open(comppath,'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if int(row[1]) < mincomp:
                break
            comp.append(row)
    return comp

csvheader, body = CTprocessing.loadCSVinfo(FINALCSV)



def createarff(mincomp, relation, filename, body):

    compmat = loadcomposers(mincomp, COMP_INFO)

    arffinfo = dict()
    arffinfo['relation'] = relation
    arffinfo['attributes'] = []

    ### Binary features
    for num in range(12):
        arffinfo['attributes'].append(('Binary_number_' + str(num+1), 'REAL'))
    ### Histogram features
    for num in range(4096):
        arffinfo['attributes'].append(('Histogram_bin_' + str(num+1), 'REAL'))

    arffinfo['data'] = []
    Cont = 0
    for row in body:
        if any(row[COMPOSER_COL] in s for s in compmat):
            tempvec = []
            ### Binary features
            tempvec.extend(np.loadtxt(row[BINARY_VEC_COL]))
            ### Histogram features
            tempvec.extend(np.loadtxt(row[HISTOGRAM_COL]))
            ### Composer name
            tempvec.append(row[COMPOSER_COL])
            arffinfo['data'].append(tempvec)
            Cont += 1
            print  Cont

    strcomp = []
    for com in compmat:
        strcomp.append(com[0])

    arffinfo['attributes'].append(('Class', strcomp))

    totext = arff.dumps(arffinfo)

    outfile = open(filename, 'wb')
    outfile.write(totext)
    outfile.close()

createarff(100,'Composers_more_100_works', 'more_100_works.arff',body)
createarff(50,'Composers_more_50_works', 'more_50_works.arff',body)
createarff(10,'Composers_more_20_works', 'more_20_works.arff',body)