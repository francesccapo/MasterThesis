import numpy as np
import csv
import pdb

ChromaNames = {'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'}

MajorKey_orig = {6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88}
MinorKey_orig = {6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17}

MajorKey_rev = {5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0}
MinorKey_rev = {5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0}

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

def loadChromaTables(csvarray):
    for i

def binarize(chromatable):
    for row in range()
"""