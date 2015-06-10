import CTprocessing
import csvprocessing
import numpy as np

FINALCSV = 'Information.csv'
BINARY_VEC_PATH = ['binary_vec', '../kunstderfugue/binary_vec/']
HISTOGRAM_VEC = ['histogram_vec', '../kunstderfugue/histogram_vec/']
PATH_FILENAME_COL = 9
PATH_CHROMATABLE_COL = 10
PITCH_ESTIMATED = 11
MODE_ESTIMADED = 12
PITCH_INDEX = 13


csvheader, body = CTprocessing.loadCSVinfo(FINALCSV)


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

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp,temparray)
        print 'File number ' + str(file) + ' binarized'

    feats = np.array(())
    for num in range(12):
        feats = np.append(feats, 'Binary_number_' + str(num + 1) )
    np.savetxt(resultpath[1] + 'feature_names_beleb.txt', feats, delimiter=',', fmt='%s')

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

        pathtmp = csvprocessing.newtxtpath(csvarray[file][PATH_FILENAME_COL],resultpath[0])
        np.savetxt(pathtmp,histog[0].astype(float)/temparray.shape[1])
        print 'File number ' + str(file) + ' histogram created'

    feats = np.array(())
    for num in range(4096):
        feats = np.append(feats, 'Histogram_bin_' + str(num + 1) )
    np.savetxt(resultpath[1] + 'feature_names_beleb.txt', feats, delimiter=',', fmt='%s')



###### Binary vector processing: csv information modification and results export
binarize(body, BINARY_VEC_PATH)


##### Histogram creation: csv information modification and results export
histogram(body, HISTOGRAM_VEC)




