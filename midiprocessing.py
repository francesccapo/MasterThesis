import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pdb


SMF2TXT_BIN = './smf2txt'



class Track():
    def __init__(self):
        self.trackID = []
        self.notesNum = []
        self.name = []
        self.info = np.empty([0, 4])
        self.resolution = []


def midiload(midifile):
    txttemp = 'temp.txt'
    fo = open(txttemp,'w')
    fo.close()
    order = SMF2TXT_BIN + ' ' + midifile + ' > ' + txttemp
    try:
        os.system(order)
    except:
        print 'ERROR on ' + order
        sys.exit(1)

    f = open(txttemp, 'r')
    midi_lines = f.readlines()
    f.close()

    tracklist = []
    cont = -1
    notes = 0
    resolution = 0
    errors = []
    for midi_line in range(len(midi_lines)):
        midi_lines[midi_line] = midi_lines[midi_line][:-1]
        if midi_lines[midi_line] == '':
            continue
        if midi_lines[midi_line].find('#') != -1 or midi_lines[midi_line].find('@') != -1:
            if midi_lines[midi_line].find('resolution') != -1:
                tp = midi_lines[midi_line].split(' ')
                resolution = int(tp[1])
            if midi_lines[midi_line].find('track') != -1:
                cont += 1
                tracklist.append(Track())
                tp = midi_lines[midi_line].split(' ')
                tracklist[cont].trackID = tp[1]
                tracklist[cont].resolution = resolution
                if tp[2] == '':
                    tracklist[cont].name = 'Empty'
                else:
                    tracklist[cont].name = ' '.join(tp[2:])
                if notes != 0:
                    tracklist[cont - 1].notesNum = notes
                    notes = 0
        else:
            tp = np.array(midi_lines[midi_line].split(' '))
            try:
                tracklist[cont].info = np.vstack([tracklist[cont].info, tp.astype(np.float)])
            except:
                errors.append(tp)
                continue
            notes += 1
    if cont >= 0 :
        tracklist[cont].notesNum = notes

    rmlist = []

    for i in range(len(tracklist)):
        if tracklist[i].notesNum == [] or tracklist[i].notesNum == 0:
            rmlist.append(i)
    for i in range(len(rmlist) - 1, -1, -1):
        tracklist.__delitem__(rmlist[i])
    #print 'Midi "' + midifile + '" loaded'

    os.remove(txttemp)

    return tracklist, errors


def durationmidi(tracklist):
    maxtime = 0
    for i in range(len(tracklist)):
        if tracklist[i].notesNum >= 20:
            for j in range(tracklist[i].notesNum - 1, tracklist[i].notesNum - 21, -1):
                if maxtime < (tracklist[i].info[j][1] + tracklist[i].info[j][2]):
                    maxtime = tracklist[i].info[j][1] + tracklist[i].info[j][2]
        else:
            for j in range(tracklist[i].notesNum - 1, -1, -1):
                if maxtime < (tracklist[i].info[j][1] + tracklist[i].info[j][2]):
                    maxtime = tracklist[i].info[j][1] + tracklist[i].info[j][2]
    #print 'Midi duration computed'
    maxtime = int(maxtime / tracklist[0].resolution)
    return maxtime


def midifilter(tracklist, maxtracks, minnotes):
    result = 0 #Result = 0 -> OK, Result = 1 -> More than maxtracks, Result = 2 -> Less than min tracks
    notes = 0
    if len(tracklist) <= maxtracks:
        for track in range(len(tracklist)):
            notes += tracklist[track].notesNum
        if notes < minnotes:
            result = 2
    else:
        result = 1

    return result



def cutmidi(tracklist, maxbeat):
    for i in range(len(tracklist)):
        for j in range(tracklist[i].notesNum - 1):
            if tracklist[i].info[j][1] > maxbeat * tracklist[i].resolution:
                tracklist[i].info = tracklist[i].info[:j]
                tracklist[i].notesNum = j
                break

    print 'Midi cut'
    return tracklist


def midiconversor(line, resolution):
    (integ, note) = divmod(line[0], 12)  # Note: (integer, residual)
    (pos, res) = divmod(line[1], resolution)
    duration = line[2]
    current = line[1]
    matrix = []
    while duration > 0.0:
        tmp = np.zeros(12)
        (pos_2, res_2) = divmod(current, resolution)
        if duration < resolution - res_2:
            tmp[note] = duration
        else:
            tmp[note] = resolution - res_2
        current += tmp[note]
        duration -= tmp[note]
        matrix.append(tmp/resolution)
    return int(pos), matrix


def normalize(v):
    norm=np.linalg.norm(v,np.inf)
    if norm==0:
       return v
    return v/norm


def chromatable(tracklist):
    table = np.zeros((12,durationmidi(tracklist)))
    for i in range(len(tracklist)):
        for note in range(tracklist[i].notesNum):
            (pos, mat) = midiconversor(tracklist[i].info[note], tracklist[i].resolution)
            for col in range(len(mat)):
                table[:, pos] = table[:, pos] + mat[col]
    for col in range(np.size(table,1)):
        table[:, col] = normalize(table[:,col])
    print 'Chroma table created'
    return table


###############
def printtable(table):

    plt.imshow(table, interpolation='nearest', cmap=cm.coolwarm)
    plt.title('Chroma table')

    plt.show()
################



mid,er = midiload('verdirequiem.mid')

tab = chromatable(mid)

printtable(tab)