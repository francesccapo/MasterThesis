

def write(filename, header, body):
    f=open(filename,'w')
    for i in range(header.shape[1]):
        f.write(str(header[i])+',')
    f.seek(-1,2)
    f.write('\n')
    for i in range(body.shape[0]):
        for j in range(body.shape[1]):
            if j != body.shape[1]-1:
                f.write(str(body[i,j])+',')
            else:
                f.write(str(body[i,j])+'\n')
    f.close()

"""
ORIGINAL JOAN
X
Y
f=open('filename.csv','w')
for j in range(X.shape[1]):
    f.write('x'+str(i)+',')
f.write('composer\n')
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        f.write(str(X[i,j])+',')
    f.write(Y[i]+'\n')
f.close()
"""