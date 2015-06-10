import CTprocessing
import csvprocessing
import numpy as np
import scipy

FINALCSV = 'Information.csv'
NUMBER_OF_0_PATH = ['number_of_0', '../kunstderfugue/number_of_0/']
NUMBER_OF_SILENCES_PATH = ['number_of_silences', '../kunstderfugue/number_of_silences/']
PROFILE_TEMPLATE_CORR_PATH = ['correlation_template', '../kunstderfugue/correlation_template/']
INTERNAL_CORR_PATH = ['internal_correlation', '../kunstderfugue/internal_correlation/']
DIFF_NO_BINARY_PATH = ['differential_no_binary', '../kunstderfugue/differential_no_binary/']
DIFF_BINARY_PATH = ['differential_binary', '../kunstderfugue/differential_binary/']
SUMATORY_COLS_PATH = ['sumatory_columns', '../kunstderfugue/sumatory_columns/']
HIST_NON_ZERO_PATH = ['hist_non_zero_col', '../kunstderfugue/hist_non_zero_col/']
HIST_STRONGEST_VAL_PATH = ['hist_strongest_val', '../kunstderfugue/hist_strongest_val/']

PATH_FILENAME_COL = 9
PATH_CHROMATABLE_COL = 10
PITCH_ESTIMATED = 11
MODE_ESTIMADED = 12
PITCH_INDEX = 13

#Takes Temperley except when they estimate the same key but different mode.
KEYPROFILES = [[np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]), 'Major'], #Krumhansl
                [np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]), 'Minor'], #Krumhansl
                [np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]), 'Major'], #Temperley
                [np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]), 'Minor']  #Temperley
                ]


csvheader, body = CTprocessing.loadCSVinfo(FINALCSV)


def calculateStatistics(vector, usage):

    if usage == 'values':
        newVector = np.array(())

        newVector = np.append(newVector, np.mean(vector))
        newVector = np.append(newVector, np.std(vector))
        newVector = np.append(newVector, scipy.stats.kurtosis(vector))
        newVector = np.append(newVector, scipy.stats.skew(vector))
        newVector = np.append(newVector, np.percentile(vector, 95))
        newVector = np.append(newVector, np.percentile(vector, 5))

        return newVector

    else:
        newVectorString = np.array(())

        newVectorString = np.append(newVectorString, vector + '_mean')
        newVectorString = np.append(newVectorString, vector + '_std')
        newVectorString = np.append(newVectorString, vector + '_kurtosis')
        newVectorString = np.append(newVectorString, vector + '_skewness')
        newVectorString = np.append(newVectorString, vector + '_percentile_95')
        newVectorString = np.append(newVectorString, vector + '_percentile_5')

        return newVectorString


def numberOf0(csvarray,resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])
        temparray = np.flipud(temparray)

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        temparray = np.around(temparray)
        tempdec = []
        for vec in range(start_beat, stop_beat, 1):
            tempvector = temparray[:,vec].astype(int)
            if csvarray[file][MODE_ESTIMADED].find('Major') != -1:
                tempvector = np.roll(tempvector, - int(csvarray[file][PITCH_INDEX]))
            else :
                tempvector = np.roll(tempvector, -3 - int(csvarray[file][PITCH_INDEX]))
            tempdec.append(int(''.join(tempvector.astype(str)),2))
        histog = np.histogram(tempdec,bins=np.arange(4097))

        feature = np.array((),dtype=int)
        number = 0
        for bin in histog[0]:
            if bin == 0:
                number += 1
        feature = np.append(feature,number)
        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp,feature)
        print 'File number ' + str(file) + ' number of 0 file created'

    feats = np.array(())
    feats = np.append(feats,'Number_of_0')
    np.savetxt(resultpath[1] + 'feature_names.txt', feats, delimiter=',', fmt='%s')


def numberOfSilences(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])
        
        silence_count = 0
        
        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        for col in range(start_beat, stop_beat, 1):
            if temparray[:,col].any() == 0:
                silence_count += 1

        feature_value = float(silence_count) / (stop_beat - start_beat)

        feature = np.array(())
        feature = np.append(feature,feature_value)
        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp,feature)
        print 'File number ' + str(file) + ' number of silences file created'

    feats = np.array(())
    feats = np.append(feats,'Number_of_silences')
    np.savetxt(resultpath[1] + 'feature_names.txt', feats, delimiter=',', fmt='%s')


def correlationTemplate(csvarray, resultpath):
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

        feature = np.array(())

        if tempind_1 == tempind_2 and tempprof_1 != tempind_2:
            feature = np.append(feature,tempval_1)
        else:
            feature = np.append(feature,tempval_2)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp,feature)
        print 'File number ' + str(file) + ' profile template correlation file created'


    feats = np.array(())
    feats = np.append(feats,'Profile_template_correlation')
    np.savetxt(resultpath[1] + 'feature_names.txt', feats, delimiter=',', fmt='%s')

def internalCorrelation(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])
        temparray = np.flipud(temparray)

        for col in range(temparray.shape[1]-1,-1,-1):
            if temparray[:,col].any() == 0:
                temparray = np.delete(temparray, col, 1)
        
        features = np.array(())

        for col in range(temparray.shape[1]-1):
            tempcorr = scipy.stats.pearsonr(temparray[:,col], temparray[:,col+1])
            features = np.append(features,tempcorr[0])

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, calculateStatistics(features, 'values'))

        print 'File number ' + str(file) + ' internal correlation file created'

    
    np.savetxt(resultpath[1] + 'feature_names.txt', calculateStatistics('Internal_correlation', 'string') , delimiter=',', fmt='%s')


def differentialNoBinary(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        features = np.mean(np.absolute(np.diff(temparray[:,start_beat:stop_beat])), axis= 0)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, calculateStatistics(features, 'values'))

        print 'File number ' + str(file) + ' differential no binary file created'

    np.savetxt(resultpath[1] + 'feature_names.txt', calculateStatistics('Differential_no_binary', 'string') , delimiter=',', fmt='%s')


def differentialBinary(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        temparray = np.around(temparray)

        features = np.mean(np.absolute(np.diff(temparray[:,start_beat:stop_beat])), axis= 0)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, calculateStatistics(features, 'values'))

        print 'File number ' + str(file) + ' differential binary file created'

    
    np.savetxt(resultpath[1] + 'feature_names.txt', calculateStatistics('Differential_binary', 'string') , delimiter=',', fmt='%s')


def sumatoryColumns(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        features = np.sum(temparray[:,start_beat:stop_beat], axis= 0)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, calculateStatistics(features, 'values'))

        print 'File number ' + str(file) + ' sumatory columns file created'
    
    np.savetxt(resultpath[1] + 'feature_names.txt', calculateStatistics('Sumatory_columns', 'string') , delimiter=',', fmt='%s')


def histNonZeroPerCol(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        features = np.zeros(13) #[0]: number of columns with all elements at 0, [1]: number of columns with 1 non zero element....

        for col in range(start_beat, stop_beat, 1):
            ind = np.count_nonzero(temparray[:,col])
            features[ind] += 1

        features = features / (stop_beat - start_beat)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' histogram of non zero per column file created'

    feats = np.array(())
    for bin in range(13):
        feats = np.append(feats, str(bin) + '_non_zero_elements')
    np.savetxt(resultpath[1] + 'feature_names.txt', feats, delimiter=',', fmt='%s') 

def strongestValueHist(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        temparray = np.trunc(temparray)

        features = np.zeros(13) #[0]: number of columns with all elements at 0, [1]: number of columns with 1 non zero element....

        for col in range(start_beat, stop_beat, 1):
            ind = np.count_nonzero(temparray[:,col])
            features[ind] += 1

        features = features / (stop_beat - start_beat)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' strongest value histogram file created'

    feats = np.array(())
    for bin in range(13):
        feats = np.append(feats, str(bin) + '_strongest_values')
    np.savetxt(resultpath[1] + 'feature_names.txt', feats, delimiter=',', fmt='%s') 


def combinationHist(csvarray, resultpath, fileCombinations):
    commatrix = np.loadtxt(file, dtype=[('mask', ('i1', 12)),('name', '|S32')])




##### Number of 0 presents in the histogram
#numberOf0(body, NUMBER_OF_0_PATH)

##### Number of silences in the chroma table
#numberOfSilences(body, NUMBER_OF_SILENCES_PATH)

##### Profile correlation value
#correlationTemplate(body, PROFILE_TEMPLATE_CORR_PATH)

##### Internel correlation beat-beat value
#internalCorrelation(body, INTERNAL_CORR_PATH)

##### Differential between no binary beat-beat values
#differentialNoBinary(body, DIFF_NO_BINARY_PATH)

##### Differential between binary beat-beat values
#differentialBinary(body, DIFF_BINARY_PATH)

##### Sumatory of each column of the Chroma Table
#sumatoryColumns(body, SUMATORY_COLS_PATH)

##### Histogram of non zero values per column of th Chroma Table
#histNonZeroPerCol(body, HIST_NON_ZERO_PATH)

##### Histogram of strongest values per column of th Chroma Table
#strongestValueHist(body, HIST_STRONGEST_VAL_PATH)




