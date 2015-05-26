import CTprocessing
import csv
import pdb
import csvprocessing
import numpy as np

FINALCSV = 'Information.csv'
PATH_FILENAME_COL = 9
BINARY_VEC_PATH = 'binary_vec'
HISTOGRAM_VEC = 'histogram_vec'



csvheader, body = CTprocessing.loadCSVinfo(FINALCSV)


"""
###### Pitch estimation: csv information modification
body = CTprocessing.keyestimation(body)
csvheader.append('Pitch estimated')
csvheader.append('Mode estimated')
csvheader.append('Pitch index') #['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
"""



###### Binary vector processing: csv information modification and results export
body = CTprocessing.binarize(body, BINARY_VEC_PATH)
csvheader.append('Binary Vector') #['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']



##### Histogram creation: csv information modification and results export
body = CTprocessing.histogram(body, HISTOGRAM_VEC)
csvheader.append('Histogram binary vector (4096 bins)')




#### CSV information writting
outfile = open(FINALCSV, 'wb')
wr = csv.writer(outfile)
wr.writerow(csvheader)
wr.writerows(body)
outfile.close()

