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


def midiload(midifile):
    txttemp = 'temp.txt'
    order = SMF2TXT_BIN + ' -q 1024 ' + midifile + ' > ' + txttemp
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
    for midi_line in range(len(midi_lines)):
        midi_lines[midi_line] = midi_lines[midi_line][:-1]
        if midi_lines[midi_line].find('#') != -1 or midi_lines[midi_line].find('@') != -1:
            if midi_lines[midi_line].find('track') != -1:
                cont += 1
                tracklist.append(Track())
                tp = midi_lines[midi_line].split(' ')
                tracklist[cont].trackID = tp[1]
                tracklist[cont].name = ' '.join(tp[2:])
                if notes != 0:
                    tracklist[cont - 1].notesNum = notes
                    notes = 0
        else:
            tp = np.array(midi_lines[midi_line].split(' '))
            tracklist[cont].info = np.vstack([tracklist[cont].info, tp.astype(np.float)])
            notes += 1

    tracklist[cont].notesNum = notes

    rmlist = []

    for i in range(len(tracklist)):
        if tracklist[i].notesNum == [] or tracklist[i].notesNum == 0:
            rmlist.append(i)
    for i in range(len(rmlist) - 1, -1, -1):
        tracklist.__delitem__(rmlist[i])
    print 'Midi "' + midifile + '" loaded'
    return tracklist


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
    print 'Midi duration computed'
    maxtime = int(maxtime / 1024)
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
            if tracklist[i].info[j][1] > maxbeat * 1024:
                tracklist[i].info = tracklist[i].info[:j]
                tracklist[i].notesNum = j
                break

    print 'Midi cut'
    return tracklist


def midiconversor(line):
    (integ, note) = divmod(line[0], 12)  # Note: (integer, residual)
    (pos, res) = divmod(line[1], 1024)
    duration = line[2]
    current = line[1]
    matrix = []
    while duration > 0.0:
        tmp = np.zeros(12)
        (pos_2, res_2) = divmod(current, 1024)
        if duration < 1024 - res_2:
            tmp[note] = duration
        else:
            tmp[note] = 1024 - res_2
        current += tmp[note]
        duration -= tmp[note]
        matrix.append(tmp/1024)
    return int(pos), matrix


def chromatable(tracklist):
    table = np.zeros((12,durationmidi(tracklist)))
    for i in range(len(tracklist)):
        for note in range(tracklist[i].notesNum):
            (pos, mat) = midiconversor(tracklist[i].info[note])
            for col in range(len(mat)):
                table[:, pos] = table[:, pos] + mat[col]
    print 'Chroma table created'
    return table

def printtable(table):
    fig, ax = plt.subplots()

    cax = ax.imshow(table, interpolation='nearest', cmap=cm.coolwarm)
    ax.set_title('Chroma table')
    maxval = np.max(table)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    #cbar = fig.colorbar(cax, ticks=[ 0, maxval])
    #cbar.ax.set_yticklabels(['0', (str(maxval))])# vertically oriented colorbar

    plt.show()