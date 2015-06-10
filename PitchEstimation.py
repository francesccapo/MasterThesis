import CTprocessing
import csv

FINALCSV = 'Information.csv'


csvheader, body = CTprocessing.loadCSVinfo(FINALCSV)


###### Pitch estimation: csv information modification
body = CTprocessing.keyestimation(body)
csvheader.append('Pitch estimated')
csvheader.append('Mode estimated')
csvheader.append('Pitch index') #['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


#### CSV information re-writting
outfile = open(FINALCSV, 'wb')
wr = csv.writer(outfile)
wr.writerow(csvheader)
wr.writerows(body)
outfile.close()

print 'Finished'
