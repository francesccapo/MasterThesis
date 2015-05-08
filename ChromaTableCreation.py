import midiprocessing
import csvprocessing
import sys
import pdb
import numpy as np
import csv


PATHMIDI = '/Users/Xesc/Dropbox/Master/Projecte/kunstderfugue/midi/'
CSV_FILE = '/Users/Xesc/Dropbox/Master/Projecte/kunstderfugue/metadata/metadata.tsv'
MIN_WORKS = 20
MAX_TRACKS = 50
MIN_NOTES = 40
PATH_FILENAME_COL = 9
TRACKLIST_COL = 10
PATH_CHROMATABLE_COL = 11
CHROMATABLE_FOLDERNAME = 'chromatables'
WORKS_COMPOSER = 'Works_per_composer.csv'
FINALCSV = 'Information.csv'


pathinfo = csvprocessing.filterroot(PATHMIDI)
print 'Path loaded.'
csvheader,csvinfo = csvprocessing.loadcsv(CSV_FILE)
csvheader.append('Purified Path Midi File')
csvheader.append('Chroma Table Text File')
print 'Csv loaded.'
csvcomplete, csvnonenter = csvprocessing.associate(csvinfo,pathinfo)
print 'Path & Csv associated.'
selection_1, csvfiltered_1 = csvprocessing.filterminworks(csvcomplete,MIN_WORKS)
print 'Data base filtered with minimum of ' + str(MIN_WORKS) + ' works.'

csvfiltered_midi = []
csvfiltered_midi_MAXTRACKS = []
csvfiltered_midi_MINNOTES = []
errors_tot = []

for midfile in range (len(csvfiltered_1)):
    print 'Arxiu ' + str(midfile)
    temptracklist = []
    templine = csvfiltered_1[midfile][:]
    errors = []
    temptracklist,errors = midiprocessing.midiload(csvfiltered_1[midfile][PATH_FILENAME_COL])

    if errors:
        errors_tot.append((csvfiltered_1[midfile][PATH_FILENAME_COL],errors))
        errors = []
        continue
    templine.append(temptracklist)
    midifilt = midiprocessing.midifilter(temptracklist,MAX_TRACKS,MIN_NOTES)
    if midifilt == 0:
        csvfiltered_midi.append(templine)
    elif midifilt == 1:
        csvfiltered_midi_MAXTRACKS.append(templine)
    elif midifilt == 2:
        csvfiltered_midi_MINNOTES.append(templine)
    else:
        print 'ERROR with ' + csvfiltered_1[midfile][PATH_FILENAME_COL]
        sys.exit(-1)

print 'Filtered by MAXTRACKS and MINNOTES'


selection_2, csvfiltered_2 = csvprocessing.filterminworks(csvfiltered_midi,MIN_WORKS)


print 'Data base filtered with minimum of ' + str(MIN_WORKS) + ' works: 2nd round'


for file in range(len(csvfiltered_2)):
    pathtmp = csvprocessing.newtxtpath(csvfiltered_2[file][PATH_FILENAME_COL],CHROMATABLE_FOLDERNAME)
    csvfiltered_2[file].append(pathtmp)
    np.savetxt(pathtmp,midiprocessing.chromatable(csvfiltered_2[file][TRACKLIST_COL]))
    del csvfiltered_2[file][TRACKLIST_COL]
    print 'File num ' + str(file) + ' processed'

pdb.set_trace()

outfile = open(WORKS_COMPOSER,'wb')
wr = csv.writer(outfile)
wr.writerows(selection_2)
outfile.close()


outfile = open(FINALCSV, 'wb')
wr = csv.writer(outfile)
wr.writerow(csvheader)
wr.writerows(csvfiltered_2)
outfile.close()

pdb.set_trace()

print 'Finished'

