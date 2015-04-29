import os
import operator
import pdb


SKIP_CHARACTERS = ['(', ')', '!', '[', ']', '\'', '&', ' ']
COMPOSER_COL = 1
FILE_COL = 0

def filterroot(mainroot):
    res = []
    for root, dirs, files in os.walk(mainroot):
        for name in files:
            if os.path.join(root, name).find('.mid') != -1:
                filename = os.path.join(root, name)
                fntmp = ''
                fntmp_2 = ''
                for i in range(len(filename) - 4):
                    if filename[i] in SKIP_CHARACTERS:
                        fntmp += str('_')
                        fntmp_2 += str('\\')+str(filename[i])
                        continue
                    fntmp += str(filename[i])
                    fntmp_2 += str(filename[i])
                res.append((fntmp_2+'.mid', name, fntmp[len(mainroot):]))
    return res


def loadcsv(csv_file):
    f = open(csv_file, 'r')
    lines = f.readlines()
    f.close()

    header = lines[0][:-1].split('\t')
    body = []
    for line in range(1, len(lines)):
        tmpstr = lines[line][:-1].split('\t')
        body.append(tmpstr)

    return header, body


def associate(csv,pathinfo):
    noenter = []
    newcsv = []
    for csvfile in range(len(csv)):
        for pathfile in range(len(pathinfo)):
            if csv[csvfile][0] == pathinfo[pathfile][2]:
                newcsv.append((csv[csvfile]))
                newcsv[-1].append(pathinfo[pathfile][0])
                del pathinfo[pathfile]
                break
            if pathfile == len(pathinfo)-1:
                noenter.append(csv[csvfile])

    return newcsv, noenter


def filterminworks(matrixinfo, minworks):
    matrixinfo = sorted(matrixinfo, key=operator.itemgetter(COMPOSER_COL), reverse=False)
    name_Temp = 'null'
    number = 0
    new_body = []
    selection = []

    for line in range(len(matrixinfo)):
        if matrixinfo[line][COMPOSER_COL] != '':
            if matrixinfo[line][COMPOSER_COL] == name_Temp:
                number += 1
            else:
                if number != 0 and number >= minworks:
                    for rewrite in range(number):
                        new_body.append(matrixinfo[line-1-rewrite])
                    selection.append((name_Temp,number))
                name_Temp = matrixinfo[line][COMPOSER_COL]
                number = 1
    if number != 0 and number >= minworks:
        for rewrite in range(number):
            new_body.append(matrixinfo[line-rewrite])
        selection.append((name_Temp,number))


    new_body = sorted(new_body, key=operator.itemgetter(FILE_COL), reverse=False)
    new_body = sorted(new_body, key=operator.itemgetter(COMPOSER_COL), reverse=False)


    selection = sorted(selection, key=operator.itemgetter(1), reverse=True)

    return selection,new_body


"""
CHROMATABLEPAHT EXAMPLE:
    midiname = '~/kunstderfugue/midi/midifile.mid'
    subfoldername = 'chromatables'
    result = '~/kunstderfugue/chromatables/midifile.txt'
"""


def newtxtpath(midiname, subfoldername):
    tmpname = midiname.replace('\\', '')
    tmpname = tmpname.replace('.mid','.txt')
    tmpname = tmpname.replace('/midi/','/'+subfoldername+'/')
    finpos = tmpname.rfind('/')+1
    tmppath = tmpname[:finpos]
    if not os.path.exists(tmppath):
        os.makedirs(tmppath)
    return tmpname
