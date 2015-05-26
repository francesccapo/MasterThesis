import numpy as np
import csv
import csvprocessing
import scipy.stats
import pdb

PATH_FILENAME_COL = 9
PATH_CHROMATABLE_COL = 10
PITCH_ESTIMATED = 11
MODE_ESTIMADED = 12
PITCH_INDEX = 13

ChromaNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

#Takes Temperley except when they estimate the same key but different mode.
KEYPROFILES = [[np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]), 'Major'], #Krumhansl
                [np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]), 'Minor'], #Krumhansl
                [np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]), 'Major'], #Temperley
                [np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]), 'Minor']  #Temperley
                ]

def loadCSVinfo(csvpath):
    csvarray = []
    header = []
    with open(csvpath,'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if not header:
                header = row
            else:
                csvarray.append(row)
    return header, csvarray

def keyestimation(csvarray):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])
        temparray = np.flipud(temparray)
        tempmean = np.mean(temparray, axis=1)
        tempval_1 = []
        tempind_1 = []
        tempprof_1 = []
        tempval_2 = []
        tempind_2 = []
        tempprof_2 = []
        for prof in range(2):
            for chromepitch in range(12):
                tempcorr = scipy.stats.pearsonr(np.roll(tempmean,-chromepitch),KEYPROFILES[prof][0])
                if tempcorr[0] > tempval_1 or not tempval_1:
                    tempind_1 = chromepitch
                    tempval_1 = tempcorr[0]
                    tempprof_1 = KEYPROFILES[prof][1]
        for prof in range(2):
            for chromepitch in range(12):
                tempcorr = scipy.stats.pearsonr(np.roll(tempmean,-chromepitch),KEYPROFILES[prof+2][0])
                if tempcorr[0] > tempval_2 or not tempval_2:
                    tempind_2 = chromepitch
                    tempval_2 = tempcorr[0]
                    tempprof_2 = KEYPROFILES[prof][1]
        if tempind_1 == tempind_2 and tempprof_1 != tempind_2:
            csvarray[file].append(ChromaNames[tempind_1])
            csvarray[file].append(tempprof_1)
            csvarray[file].append(tempind_1)
        else:
            csvarray[file].append(ChromaNames[tempind_2])
            csvarray[file].append(tempprof_2)
            csvarray[file].append(tempind_2)
        print 'File number ' + str(file) + ' processed'
    return csvarray


def binarize(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])
        temparray = np.flipud(temparray)
        temparray = np.around(temparray)
        temparray = np.sum(temparray, axis=1)/temparray.shape[1]
        if csvarray[file][MODE_ESTIMADED].find('Major') != -1:
            temparray = np.roll(temparray, - int(csvarray[file][PITCH_INDEX]))
        else :
            temparray = np.roll(temparray, -3 - int(csvarray[file][PITCH_INDEX]))

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath)
        csvarray[file].append(pathtmp)
        np.savetxt(pathtmp,temparray)


        print 'File number ' + str(file) + ' binarized'

    return csvarray



def histogram(csvarray,resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])
        temparray = np.flipud(temparray)
        temparray = np.around(temparray)
        tempdec = []
        for vec in range(temparray.shape[1]):
            tempvector = temparray[:,vec].astype(int)
            if csvarray[file][MODE_ESTIMADED].find('Major') != -1:
                tempvector = np.roll(tempvector, - int(csvarray[file][PITCH_INDEX]))
            else :
                tempvector = np.roll(tempvector, -3 - int(csvarray[file][PITCH_INDEX]))
            tempdec.append(int(''.join(tempvector.astype(str)),2))
        histog = np.histogram(tempdec,bins=np.arange(4097))

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath)
        csvarray[file].append(pathtmp)
        np.savetxt(pathtmp,histog[0].astype(float)/temparray.shape[1])
        print 'File number ' + str(file) + ' processed'

    return csvarray
