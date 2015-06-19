import CTprocessing
import csvprocessing
import numpy as np
import scipy
import pdb

FINALCSV = 'Information.csv'
NOTES_COMBINATIONS_NUMPY_TXT = 'notesDictionary.txt'
NUMBER_OF_0_PATH = ['number_of_0', '../kunstderfugue/number_of_0/']
PROFILE_TEMPLATE_CORR_PATH = ['correlation_template', '../kunstderfugue/correlation_template/']
INTERNAL_CORR_PATH = ['internal_correlation', '../kunstderfugue/internal_correlation/']
DIFF_NO_BINARY_PATH = ['differential_no_binary', '../kunstderfugue/differential_no_binary/']
DIFF_BINARY_PATH = ['differential_binary', '../kunstderfugue/differential_binary/']
SUMATORY_COLS_PATH = ['sumatory_columns', '../kunstderfugue/sumatory_columns/']
PRES_CHROMA_BIN_PATH = ['beat_pres_chroma_bin', '../kunstderfugue/beat_pres_chroma_bin/']
PRES_CHROMA_STRONG_PATH = ['beat_pres_chroma_strongest', '../kunstderfugue/beat_pres_chroma_strongest/']
COMBINATION_STRONGEST_VAL_PATH = ['comb_strongest_val', '../kunstderfugue/comb_strongest_val/']
COMBINATION_BINARIZED_VAL_PATH = ['comb_binarized_val', '../kunstderfugue/comb_binarized_val/']
INTERVALS_UNISON_VAL_PATH = ['interval_unison_val', '../kunstderfugue/interval_unison_val/']
REL_INTERVAL_VAL_PATH = ['rel_interval_val','../kunstderfugue/rel_interval_val/']

PATH_FILENAME_COL = 9
PATH_CHROMATABLE_COL = 10
PITCH_ESTIMATED = 11
MODE_ESTIMADED = 12
PITCH_INDEX = 13

CHROMA_PITCH_NAMES = ['B', 'A#', 'A', 'G#', 'G', 'F#', 'F', 'E', 'D#', 'D', 'C#', 'C']

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
    feats = np.append(feats,'Empty_hist_bins')
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

def internalCorrelation(csvarray, resultpath, max_beat_lag):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])
        temparray = np.flipud(temparray)

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1
        
        features = np.array(())

        for lag in range(max_beat_lag):
            tempfeat = np.array(())
            for col in range(start_beat, stop_beat-1-lag, 1):
                tempcorr = scipy.stats.pearsonr(temparray[:,col], temparray[:,col+1+lag])
                if np.isnan(tempcorr[0]):
                    tempfeat = np.append(tempfeat,0)
                    continue
                tempfeat = np.append(tempfeat,tempcorr[0])
            features = np.append(features, calculateStatistics(tempfeat, 'values'))


        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' internal correlation file created'

    featNames = np.array(())
    for lag in range(max_beat_lag):
        featNames = np.append(featNames, calculateStatistics('Internal_correlation_lag_' + str(lag+1), 'string'))
    np.savetxt(resultpath[1] + 'feature_names.txt', featNames , delimiter=',', fmt='%s')


def differentialNoBinary(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        features = np.array(())
        features = np.append(features,calculateStatistics(np.sum(np.absolute(np.diff(temparray[:,start_beat:stop_beat])), axis= 0), 'values'))
        for pitch in range(12):
            features = np.append(features,calculateStatistics(np.absolute(np.diff(temparray[pitch,start_beat:stop_beat])), 'values'))

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' differential no binary file created'


    featNames = np.array(())
    featNames = np.append(featNames, calculateStatistics('Diff_no_binary_ALL', 'string'))
    for pitch in CHROMA_PITCH_NAMES:
        featNames = np.append(featNames, calculateStatistics('Diff_no_binary_' + pitch, 'string'))

    np.savetxt(resultpath[1] + 'feature_names.txt', featNames , delimiter=',', fmt='%s')


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

        features = np.array(())
        features = np.append(features,calculateStatistics(np.sum(np.absolute(np.diff(temparray[:,start_beat:stop_beat])), axis= 0), 'values'))
        for pitch in range(12):
            features = np.append(features,calculateStatistics(np.absolute(np.diff(temparray[pitch,start_beat:stop_beat])), 'values'))

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' differential binary file created'

    featNames = np.array(())
    featNames = np.append(featNames, calculateStatistics('Diff_binary_ALL', 'string'))
    for pitch in CHROMA_PITCH_NAMES:
        featNames = np.append(featNames, calculateStatistics('Diff_binary_' + pitch, 'string'))

    np.savetxt(resultpath[1] + 'feature_names.txt', featNames , delimiter=',', fmt='%s')


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


def beatPresentChromaBin(csvarray, resultpath):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        temparray = np.around(temparray)

        features = np.zeros(13) #[0]: number of columns with all elements at 0, [1]: number of columns with 1 non zero element....

        for col in range(start_beat, stop_beat, 1):
            ind = np.count_nonzero(temparray[:,col])
            features[ind] += 1

        features = features / (stop_beat - start_beat)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' histogram of present pitch binarized per beat file created'

    feats = np.array(())
    for bin in range(13):
        feats = np.append(feats, str(bin) + '_binarized_chroma_classes')
    np.savetxt(resultpath[1] + 'feature_names.txt', feats, delimiter=',', fmt='%s') 

def beatPresentChromaStrongest(csvarray, resultpath):
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

        print 'File number ' + str(file) + ' histogram of present strongest pitch per beat file created'

    feats = np.array(())
    for bin in range(13):
        feats = np.append(feats, str(bin) + '_strongest_chroma_classes')
    np.savetxt(resultpath[1] + 'feature_names.txt', feats, delimiter=',', fmt='%s') 


def combinationHistStrongest(csvarray, resultpath, fileCombinations):
    commatrix = np.loadtxt(fileCombinations, dtype=[('mask', ('i1', 12)),('name', '|S32')], comments='%')

    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        temparray = np.trunc(temparray)

        features = np.zeros(len(commatrix)+1)

        for col in range(start_beat, stop_beat, 1):
            detected = False
            for chord in range(len(commatrix)):
                for pos in range(12):
                    if np.array_equal(temparray[:,col],np.roll(commatrix[chord]['mask'],pos)):
                        detected = True
                        features[chord] += 1
                        break
                if detected:
                    break
            if not detected:
                features[-1] += 1

        features = features / (stop_beat - start_beat)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' histogram combination of strongest values file created'

    feats = commatrix[:]['name']
    feats = np.append(feats,'No_chord')
    feats = np.array(([it + '_of_strongest_classes_hist' for it in feats]))
    np.savetxt(resultpath[1] + 'feature_names.txt', feats, delimiter=',', fmt='%s')


def combinationHistBinary(csvarray, resultpath, fileCombinations):
    commatrix = np.loadtxt(fileCombinations, dtype=[('mask', ('i1', 12)),('name', '|S32')], comments='%')

    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        temparray = np.around(temparray)

        features = np.zeros(len(commatrix)+1)

        for col in range(start_beat, stop_beat, 1):
            detected = False
            for chord in range(len(commatrix)):
                for pos in range(12):
                    if np.array_equal(temparray[:,col],np.roll(commatrix[chord]['mask'],pos)):
                        detected = True
                        features[chord] += 1
                        break
                if detected:
                    break
            if not detected:
                features[-1] += 1

        features = features / (stop_beat - start_beat)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' histogram combination of binarized values file created'

    feats = commatrix[:]['name']
    feats = np.append(feats,'No_chord')
    feats = np.array(([it + '_of_binary_classes_hist' for it in feats]))
    np.savetxt(resultpath[1] + 'feature_names.txt', feats, delimiter=',', fmt='%s')


def intervalsBetweenStrongestUnisons(csvarray, resultpath, max_beat_lag):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        temparray = np.trunc(temparray)

        features = np.array(()) #Description above, at feats definition

        for lag in range(max_beat_lag):
            tempfeat = np.zeros(7)
            for col in range(start_beat, stop_beat - 1 - lag , 1):
                if np.count_nonzero(temparray[:,col]) == 1 and np.count_nonzero(temparray[:,col+1+lag]) == 1:
                    if np.mod(temparray[:,col].nonzero()[0][0]-temparray[:,col+1+lag].nonzero()[0][0],12) == 0:
                        tempfeat[0] += 1
                    elif np.mod(temparray[:,col].nonzero()[0][0]-temparray[:,col+1+lag].nonzero()[0][0],12) in (1, 11):
                        tempfeat[1] += 1                
                    elif np.mod(temparray[:,col].nonzero()[0][0]-temparray[:,col+1+lag].nonzero()[0][0],12) in (2, 10):
                        tempfeat[2] += 1
                    elif np.mod(temparray[:,col].nonzero()[0][0]-temparray[:,col+1+lag].nonzero()[0][0],12) in (3, 9):
                        tempfeat[3] += 1
                    elif np.mod(temparray[:,col].nonzero()[0][0]-temparray[:,col+1+lag].nonzero()[0][0],12) in (4, 8):
                        tempfeat[4] += 1
                    elif np.mod(temparray[:,col].nonzero()[0][0]-temparray[:,col+1+lag].nonzero()[0][0],12) in (5, 7):
                        tempfeat[5] += 1
                    else:
                        tempfeat[6] += 1
            features = np.append(features, tempfeat)

        features = features / (stop_beat - start_beat)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' intervals between strongest unison values file created'

    feats = np.array(())
    feats = np.append(feats,'Unison_dist_between_beats')
    feats = np.append(feats,'2nd_m_or_7th_M_dist_between_beats')
    feats = np.append(feats,'2nd_M_or_7th_m_dist_between_beats')
    feats = np.append(feats,'3rd_m_or_6th_M_dist_between_beats')
    feats = np.append(feats,'3rd_M_or_6th_m_dist_between_beats')
    feats = np.append(feats,'4th_or_5th_dist_between_beats')
    feats = np.append(feats,'Tritone_dist_between_beats')

    newfeats = np.array(())
    for i in range(max_beat_lag):
        newfeats = np.append(newfeats,[j + '_lag_'+ str(i+1) for j in feats])

    np.savetxt(resultpath[1] + 'feature_names.txt', newfeats, delimiter=',', fmt='%s')


def relBetweenStrongestVal(csvarray, resultpath, max_beat_lag):
    for file in range(len(csvarray)):
        temparray = np.loadtxt(csvarray[file][PATH_CHROMATABLE_COL])

        start_beat = 0
        stop_beat = temparray.shape[1]
        
        while temparray[:,start_beat].any() == 0:
            start_beat += 1
        while temparray[:,stop_beat-1].any() == 0:
            stop_beat -= 1

        temparray = np.trunc(temparray)

        """
        +-------------------------------------+--------+-------+-------+-------+-------+-----+---------+
        |           Feature number:           | Unison | 2m_7M | 2M_7m | 3m_6M | 3M_6m | 4_5 | Tritone |
        | Origin (row) / Destination (column) |        |       |       |       |       |     |         |
        +-------------------------------------+--------+-------+-------+-------+-------+-----+---------+
        |                Unison               |    0   |   1   |   2   |   3   |   4   |  5  |    6    |
        +-------------------------------------+--------+-------+-------+-------+-------+-----+---------+
        |                2m_7M                |    7   |   8   |   9   |   10  |   11  |  12 |    13   |
        +-------------------------------------+--------+-------+-------+-------+-------+-----+---------+
        |                2M_7m                |   14   |   15  |   16  |   17  |   18  |  19 |    20   |
        +-------------------------------------+--------+-------+-------+-------+-------+-----+---------+
        |                3m_6M                |   21   |   22  |   23  |   24  |   25  |  26 |    27   |
        +-------------------------------------+--------+-------+-------+-------+-------+-----+---------+
        |                3M_6m                |   28   |   29  |   30  |   31  |   32  |  33 |    34   |
        +-------------------------------------+--------+-------+-------+-------+-------+-----+---------+
        |                4P_5P                |   35   |   36  |   37  |   38  |   39  |  40 |    41   |
        +-------------------------------------+--------+-------+-------+-------+-------+-----+---------+
        |               Tritone               |   42   |   43  |   44  |   45  |   46  |  47 |    48   |
        +-------------------------------------+--------+-------+-------+-------+-------+-----+---------+
        """

        features = np.array(())

        for lag in range(max_beat_lag):
            tempfeat = np.zeros(49)
            for col in range(start_beat, stop_beat-1-lag, 1):
                if np.count_nonzero(temparray[:,col]) == 1 and np.count_nonzero(temparray[:,col+1+lag]) == 1:
                    tempfeat[0] += 1
                elif np.count_nonzero(temparray[:,col]) == 1 and np.count_nonzero(temparray[:,col+1+lag]) == 2:
                    tmpint = np.diff(np.nonzero(temparray[:,col+1+lag])[0])[0]
                    if tmpint > 6:
                        tmpint = 12 - tmpint
                    tempfeat[tmpint] += 1
                elif np.count_nonzero(temparray[:,col]) == 2 and np.count_nonzero(temparray[:,col+1+lag]) == 1:
                    tmpint = np.diff(np.nonzero(temparray[:,col])[0])[0]
                    if tmpint > 6:
                        tmpint = 12 - tmpint
                    tempfeat[tmpint*7] += 1
                elif np.count_nonzero(temparray[:,col]) == 2 and np.count_nonzero(temparray[:,col+1+lag]) == 2:
                    tmpint1 = np.diff(np.nonzero(temparray[:,col])[0])[0]
                    if tmpint1 > 6:
                        tmpint1 = 12 - tmpint1
                    tmpint2 = np.diff(np.nonzero(temparray[:,col+1+lag])[0])[0]
                    if tmpint2 > 6:
                        tmpint2 = 12 - tmpint2
                    tempfeat[tmpint1*7 + tmpint2] += 1
            features = np.append(features, tempfeat)

        features = features / (stop_beat - start_beat)

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp, features)

        print 'File number ' + str(file) + ' relation between strongest values file created'

    featNames = ['Unison', '2m_7M', '2M_7m', '3m_6M', '3M_6m', '4P_5P', 'Tritone']

    feats = np.array(())

    for row in range(7):
        for col in range(7):
            feats = np.append(feats,featNames[row] + '_to_' + featNames[col])

    newfeats = np.array(())
    for i in range(max_beat_lag):
        newfeats = np.append(newfeats,[j + '_lag_'+ str(i+1) for j in feats])

    np.savetxt(resultpath[1] + 'feature_names.txt', newfeats, delimiter=',', fmt='%s')


##### Number of 0 presents in the histogram
numberOf0(body, NUMBER_OF_0_PATH)

##### Profile correlation value
correlationTemplate(body, PROFILE_TEMPLATE_CORR_PATH)

##### Internel correlation beat-beat value
internalCorrelation(body, INTERNAL_CORR_PATH, 4)

##### Differential between no binary beat-beat values
differentialNoBinary(body, DIFF_NO_BINARY_PATH)

##### Differential between binary beat-beat values
differentialBinary(body, DIFF_BINARY_PATH)

##### Sumatory of each column of the Chroma Table
sumatoryColumns(body, SUMATORY_COLS_PATH)

##### Histogram of non zero values per column of the Chroma Table
beatPresentChromaBin(body, PRES_CHROMA_BIN_PATH)

##### Histogram of strongest values per column of the Chroma Table
beatPresentChromaStrongest(body, PRES_CHROMA_STRONG_PATH)

##### Histogram of notes combination of strongest values per column of the Chroma Table
combinationHistStrongest(body, COMBINATION_STRONGEST_VAL_PATH, NOTES_COMBINATIONS_NUMPY_TXT)

##### Histogram of notes combination of binarized values per column of the Chroma Table
combinationHistBinary(body, COMBINATION_BINARIZED_VAL_PATH, NOTES_COMBINATIONS_NUMPY_TXT)

##### Interval between strongest unison values per column of the Chroma Table
intervalsBetweenStrongestUnisons(body, INTERVALS_UNISON_VAL_PATH, 4)

##### Relation between strongest interval values per column of the Chroma Table
relBetweenStrongestVal(body, REL_INTERVAL_VAL_PATH, 4)
