import os
import re
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
    cont = 0
    new_body = []
    selection = []

    for line in range(len(matrixinfo)):
        if matrixinfo[line][COMPOSER_COL] != '_':
            if matrixinfo[line][COMPOSER_COL] == name_Temp:
                cont += 1
            else :
                if cont != 0 and cont >= minworks:
                    for rewrite in range(cont):
                        new_body.append(matrixinfo[line-1-rewrite])
                    selection.append((name_Temp,cont))
                name_Temp = matrixinfo[line][COMPOSER_COL]
                cont = 1

    selection = sorted(selection, key=operator.itemgetter(1), reverse=True)

    return selection,new_body