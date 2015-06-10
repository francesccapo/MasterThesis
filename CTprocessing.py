import numpy as np
import csv
import csvprocessing
import scipy.stats


PATH_CHROMATABLE_COL = 10


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



