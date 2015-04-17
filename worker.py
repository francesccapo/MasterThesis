import CTprocessing
import csv
import csvprocessing
import numpy as np

FINALCSV = 'Information.csv'
EXTRACTED_INFO = 'Extractions.csv'
PATH_FILENAME_COL = 9
BINARY_VEC_PATH = 'binary_vec'
HISTOGRAM_VEC = 'histogram_vec'



csvheader, body = CTprocessing.loadCSVinfo(FINALCSV)


"""
body = CTprocessing.keyestimation(body)
csvheader.append('Pitch estimated')
csvheader.append('Mode estimated')
csvheader.append('Pitch index') #['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
"""
"""
body = CTprocessing.binarize(body)
csvheader.append('Processed Vector') #['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
"""

"""
outfile = open(FINALCSV, 'wb')
wr = csv.writer(outfile)
wr.writerow(csvheader)
wr.writerows(body)
outfile.close()
"""

csvheader.append('Binary vector')
csvheader.append('Histogram binary vector (4096 bins)')

newbody = []
for i in range(len(body)):
    newbody.append([])
    newbody[i].append(body[i][0])

newbody = CTprocessing.histogram(body, newbody)

for file in range(len(body)):
    pathtmp = csvprocessing.newtxtpath(body[file][PATH_FILENAME_COL],BINARY_VEC_PATH)
    body[file].append(pathtmp)
    np.savetxt(pathtmp,newbody[file][1])

for file in range(len(body)):
    pathtmp = csvprocessing.newtxtpath(body[file][PATH_FILENAME_COL],HISTOGRAM_VEC)
    body[file].append(pathtmp)
    np.savetxt(pathtmp,newbody[file][2])



outfile = open(FINALCSV, 'wb')
wr = csv.writer(outfile)
wr.writerow(csvheader)
wr.writerows(body)
outfile.close()


