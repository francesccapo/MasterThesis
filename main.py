import midiprocessing
import csvprocessing
import sys
import pdb

PATHMIDI = '/Users/Xesc/Dropbox/Master/Projecte/kunstderfugue/midi/'
CSV_FILE = '/Users/Xesc/Dropbox/Master/Projecte/kunstderfugue/metadata/metadata.tsv'
MIN_WORKS = 20
MAX_TRACKS = 50
MIN_NOTES = 40
PATH_FILENAME_COL = 9
TRACKLIST_COL = 10


pathinfo = csvprocessing.filterroot(PATHMIDI)
print 'Path loaded.'
csvheader,csvinfo = csvprocessing.loadcsv(CSV_FILE)
print 'Csv loaded.'
csvcomplete, csvnonenter = csvprocessing.associate(csvinfo,pathinfo)
print 'Path & Csv associated.'
selection_1, csvfiltered_1 = csvprocessing.filterminworks(csvcomplete,MIN_WORKS)
print 'Data base filtered with minimum of ' + str(MIN_WORKS) + ' works.'

csvfiltered_midi = []
csvfiltered_midi_MAXTRACKS = []
csvfiltered_midi_MINNOTES = []
errors_tot = []


for file in range (len(csvfiltered_1)):
    temptracklist = []
    temptracklist,errors = midiprocessing.midiload(csvfiltered_1[file][PATH_FILENAME_COL])
    if errors:
        errors_tot.append((csvfiltered_1[file][PATH_FILENAME_COL],errors))
    midifilt = midiprocessing.midifilter(temptracklist,MAX_TRACKS,MIN_NOTES)
    if midifilt == 0:
        csvfiltered_midi.append(csvfiltered_1[file])
        csvfiltered_midi[-1].append(temptracklist)
    elif midifilt == 1:
        csvfiltered_midi_MAXTRACKS.append(csvfiltered_1[file])
        csvfiltered_midi_MAXTRACKS[-1].append(temptracklist)
    elif midifilt == 2:
        csvfiltered_midi_MINNOTES.append(csvfiltered_1[file])
        csvfiltered_midi_MINNOTES[-1].append(temptracklist)
    else:
        print 'ERROR with ' + csvfiltered_1[file][PATH_FILENAME_COL]
        sys.exit(-1)
    print 'File num ' + str(file)



selection_2, csvfiltered_2 = csvprocessing.filterminworks(csvfiltered_midi,MIN_WORKS)

print 'Finished'

