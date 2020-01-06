"""

@author: Braden D. Kelly

This is a little song I wrote, you might want to sing it note by note... be happy...

This code converts GROMACS energy files (*.xvg) into "IPolQ" style energy files.
i.e., it averages the pairs of windows so that for example...
U_0->1 = -5.5 kJ/mol
U_1->0 = 5.2 kJ/mol

afterwards we will have
U_0->1 = -5.35 kJ/mol
U_1->0 = 5.35 kJ/mol
"""

import numpy as np
import shutil

nFile = 29
nRows = 20001
nCoul = 10
n=55
skipColumn = 3
fileNameIn = "NVTPro"
fileNameOut = "nvtpro"
#                file   row  column
print('** Thanks for using this code, here are some assumptions: **')
print(' --------  29 windows total')
print(' --------  10 Coulomb windows')
print(' --------  20001 rows of data per file')
print(' --------  55 rows of header space in the GROMACS xvg file')
print(' --------  the first 3 columns of data are skipped in the GROMACS xvg file i.e., they contain time, total Coul, total vdW, which we do not need')
print(' --------  file name to read from starts with {}'.format(fileNameIn) )
print(' --------  file name to output to starts with {}'.format(fileNameOut) )
print("** All of these presets can be changed at the top of the IPolQ.py file.**")
print("** This currently works for TI, FEP(IEXP,DEXP), BAR and MBAR. Hold onto your hat when you use MBAR though.**")

data = np.zeros((nFile,nRows,nFile))
data_cat = np.zeros((nFile,nRows,nFile+skipColumn))
columns = np.zeros((nFile,nRows,skipColumn))
data_bar = np.zeros((nFile,nRows,nFile))

header = []

print("---Starting to read data from GROMACS *.xvg files.")

for i in range(nFile):
    data[i,:,:] = np.genfromtxt(fileNameIn+"_"+str(i)+".xvg",skip_header=n,usecols=(np.arange(skipColumn,nFile+skipColumn)) )
    columns[i,:,:] = np.genfromtxt(fileNameIn+"_"+str(i)+".xvg",skip_header=n,usecols=(0,1,2) )

print("---Fetching headers.")
for i in range(nFile):
    header_temp = []
    f = open(fileNameIn+"_"+str(i)+".xvg",'r')    
    for ind,line in enumerate(f):
        if ind < n:
            header_temp.append(line )
        else:
            break
    f.close()
    header.append(header_temp)
    

print("---Finished reading in data.")
data_bar = data

for i in range(nCoul-1):         # Note, make sure this is only going up to # Coulomb windows,
    c=i                          # we don't want to average LJ windows with each other!
    for j in range(i+1,nCoul):   
        holder =  np.add(abs(data[i,:,j]),abs(data[j,:,j +i - c -1]) ) / 2.0
        data_bar[i,:,j] = holder * np.sign(data[i,:,j])
        data_bar[j,:,j +i -c -1] = holder * np.sign(data[j,:,j +i - c -1])
        c += 1   

print("---Concatenating first 3 columns with window data matrix.")               
for i in range(nFile):
    data_cat[i,:,:] = np.concatenate((columns[i,:,:],data_bar[i,:,:]),axis=1 )   
             
# make new xvg files
print("---Using np.savetxt to put data in a temporary file.")

for i in range(nFile):
    np.savetxt("test_"+str(i)+".xvg",data_cat[i,:,:], fmt='%1.8f')  # np.array2string(header[i])

print("---Making temporary file with just the header.")   
for i in range(nFile):
    fo = open("test2_"+str(i)+".xvg",'w')
    for line in header[i]:
        fo.write(line)
    fo.close()

print("---Concatenating temporary header and data files into final *.xvg files now.")
for i in range(nFile):
    with open(fileNameOut+"_"+str(i)+'.xvg','wb' ) as wfd :
        for f in ["test2_"+str(i)+".xvg","test_"+str(i)+".xvg"]:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)

print("---Finished, that will be $100.")          

     
