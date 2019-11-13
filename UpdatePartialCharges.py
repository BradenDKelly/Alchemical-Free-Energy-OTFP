###################################################################################
#
#    Written June 10th, 2018 by Braden Kelly
#
#        This is an interface for QM/MM (of sorts)
#
#    The partial charges on a solute molecule are updated using
#    Quantum Mechanical Software.
#    This requires finding the atomic coordinates of the solute molecule
#    Sending them as input to Guassian (or other QM package)
#    and then updating the charges in the Gromacs topology file
#
#    It is recommended that editor preferences be set to the Hello Kitty style.
#
#    Updated April 2019 by Braden Kelly to scale background charges
#
###################################################################################
#
#     This assumes the ghost particle is denoted as MOL in gromacs files i.e., g96/gro
#     This assumes the solvent is denoted SOL in gromacs files i.e., g96/gro
#     The call to gaussian assumes
#         - 6 cpu
#         - 1000 mb memory
#          This can be changed at line ~533/534
#     HORTON (MBIS charges) is called from a virtual environment - you need to manually specify its location for your system 
#     - go to line ~660
#     This assumes Gaussian is loaded or can be loaded via module load gaussian, and that it is G16
#     This assumes the use of AMBER/18 - make changes on line ~673
#
#     While ESP, Mulliken, Hirshfeld, CM5, DDEC6 charges can be called using this, I cannot guarantee they will work as I have not
#         used them in a long time and they are not part of this publication. I have left the code in for the curious...
#         - DDEC6 requires the program CHARGEMOL, and its location is hardcoded here on line ~654, update as necessary to fit your location
#         - Update DDEC function call on line ~186 to modify f.write( '/home/bkelly08/Programs/chargemol_09_26_2017/atomic_densities/ \n' ) to your system
#    It should not be assumed this will work for atoms other than C/H/N/O, although it should work for C/H/N/O/P/S and it should be as easy 
#    to modify as adding additional atoms into the mass dictionary on ~ line 113. Also, modify atomDict on line 293
###################################################################################
# 
#                          INSTRUCTIONS FOR USE
#     I have made a function "Make_itp_Template_4_newCharges():" which makes a dummy itp template
#     make sure only one itp file is present in the folder, and that it is the solutes
#     You can modify Make_itp_Template_4_newCharges(): to weed out other itp folders if you want
#         - it assumes your itp is of the form NAME_STUFF.itp where the solute's name is NAME, and
#         the name is followed by an underscore
#
#      A pdb file of the solute must be present in this directory if doing AM1BCC with OTFP
#
#      If you have questions, you can contact me at bkelly08@uoguelph.ca
#      I do not deal with system specific dependencies, that is on you. There should be very few, 
#      other than perhaps installing Gaussian/HORTON/Antechamber/Chargmol etc, but contact those folks for those issues.
#      It isn't that I am lazy, I am simply useless when it comes to those types of issues.
#      This is written in Python 3.x 
#
###################################################################################
import argparse
import sys
import numpy as np
from shutil import copyfile
import os
import re
import subprocess as sub
import fileinput

parser = argparse.ArgumentParser()
parser.add_argument('-sm',   type = str,    help='name of special molecule')
parser.add_argument('-t',    type = str,    help='QM theory name, no default')
parser.add_argument('-b',    type = str,    help='QM basis name, no default')
parser.add_argument('-p',    type = str,    default = "esp",  help='abbreviation of population analysis call, default is esp')
parser.add_argument('-r',    type = float,  help='radius of cutoff, default is half the box or specified, whichever is smaller')
parser.add_argument('-bg',   type = bool,   default = True,   help='use background charges?, default is True')
parser.add_argument('-l',    type = int,    default = 20,     help='lambda number, default is 20')
parser.add_argument('-ov',   type = float,  default = 2.0,    help='QMMM mol COM overlap cutoff, default is 2.0')
parser.add_argument('-bov',  type = float,  default = 5.0,    help='QMMM atom COM overlap cutoff, default is 5.0')
parser.add_argument('-ff',   type = str,    default = "gaff", help='force field, default is gaff')
parser.add_argument('-mult', type = int,    default = 1,      help='QM multiplicity, default is 1')
parser.add_argument('-gauss',type = str,    default = "gaussInput",  help='name gaussian input file, default is gaussInput')
parser.add_argument('-log',  type = str,    default = "gjf",         help='postfix of gaussian input file, default is gjf')
parser.add_argument('-o',    type = str,    default = "aqueous.g96", help='name of gromacs file holding charges, default is aqueous.g96')
parser.add_argument('-cm',   type = str,    help = 'charge method, i.e., resp, bcc, HI, CI, mbis etc...')
parser.add_argument('-g',    type = str,    default = "g16",       help = 'name of QM program i.e., g09 or g16. default is g16')
parser.add_argument('-dir',  type = str,    help = 'name of current directory, no default allowed')
parser.add_argument('-qs',   type = float,  help = 'coulomb window, used to scale MM electric field sent to gaussian, must be between [0,1]')
parser.add_argument('-qscale', type = float,default = 1.0,         help = 'Scale MM charges to account for overpolarization from QM theory, default = 1.0')
parser.add_argument('-h2o',  type = str,    default = "spce",      help = 'water model, default is spce')

# for info on parsing see https://docs.python.org/3/library/argparse.html#dest

input = parser.parse_args()

molname      = input.sm    # solute molecule
pop          = input.p     # charge method i.e., ESP or MBK (mulliken)
Lambda       = input.l     # which window is being decoupled
overlap      = input.ov    # minimum center-center distance between solute and solvent COM's if they are to be allowed as background charges
QMtheory     = input.t     # "HF"  #HF/6-31G*  # B3LYP # HF
QMbasis      = input.b     # "AUG-cc-pVTZ", "6-31G*"
QMprogram    = input.g     # g09/g16
mult         = input.mult  # multiplicity of molecule
background   = input.bg    # True or False
ffType       = input.ff    # gaff
goutput      = input.o
inputname    = input.gauss
end          = input.log
chargeMethod = input.cm    # Needed to distinguish Iterative Hirshfeld from Iterative CM5, mbis, ddec6, resp, bcc, etc .... bcc is used to reference AM1-BCC 
cutoff       = input.r
COMoverlap   = input.bov   # center - center overlap
mmScale      = float( input.qs )    # scale factor for MM charges sent to gaussian  
qqScale      = input.qscale   # this should not be used, but is left here if one wants to furthur scale the background charges.

# this code assumes a three point water model is the solvent
if input.h2o.lower() == "spce":
    q_oxygen = -0.8476
    q_hydrogen = 0.4238
elif input.h2o.lower() =="tip3p":
    q_oxygen = -0.8340
    q_hydrogen = 0.4170   
elif "opc3" in input.h2o.lower(): # I have not technically tried OPC3 yet, but it should work fine.
    q_oxygen = -0.8952
    q_hydrogen = 0.4476 
else:
    print("Either no water method was selected or you are the ultimate Post-Modernist")   

MOL          = 'MOL'  # yeah... it seems trivial... it is. Just leave it be.

def findMass( str ): # for a given atom type, return its mass. Used in COM calculation.
    massDict = {"H":1.0079, "O":15.998, "C":12.0107, "N": 14.0067, "P":30.973762, "S":32.0600}
    
    mass = massDict.get( str[0].upper() )

    return mass

def COM(xyz, masses, num): # calculate center of mass of a molecule
    mx = 0.0
    my = 0.0
    mz = 0.0
    mm = sum( masses )
    for i in range( num ):
        mx += xyz[i,0] * masses[i]
        my += xyz[i,1] * masses[i]
        mz += xyz[i,2] * masses[i]
    COM = np.array([mx/mm, my/mm, mz/mm])
    return COM

# distance between two points using mirror image separation
def DIST(solv, sol, box):
    rij = np.array([solv[0] - sol[0],solv[1] - sol[1],solv[2] - sol[2]  ])
    rij  = rij - box * np.rint ( rij/box )
    rij_mag = np.sqrt( np.sum( rij**2 ) )
    return rij_mag

# this is only used if partial charges in windows where Coulomb is already turned off.
# If using a soft-core LJ potential, there will be overlap between solute/solvent in end windows
# QM/MM DOES NOT LIKE OVERLAP. Given that the MM charges will be scaled to 0 here anyways, this should
# be redundant code, but I leave it in.
def AtomCOMCheck(soluteCoords, solventCoords, boxSize, overlap):
    accept = True
    for coordi in soluteCoords:
        for coordj in solventCoords:
            rij = DIST( coordi, coordj, boxSize)
            if rij < overlap:
                accept = False
    return accept

def FindLength( file ): # get the box length
    f = open(file, "r" )
    for line in f:
        if "BOX" in line:
            break
    for line in f:
        print("post", line)
        if len( line.split() ) == 3:
            length = float( line.split()[0] )
            break
        else:
            print("couldn't find box length")

    return length

def GenerateHortonFile(name): # Generate template for calling Horton and making MBIS charges

    f = open(name + ".py",'w')

    f.write( '#!/usr/bin/env python \n\n' )
    f.write( 'import numpy as np \n\n' )
    f.write( 'from horton import * \n\n' )
    f.write( "fn_fchk = 'molecule.fchk' \n" )
    f.write( 'mol = IOData.from_file(fn_fchk) \n\n')
    f.write( '# Partition the density with the Becke scheme \n')
    f.write( "grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, mode='only', agspec='ultrafine') \n" )
    f.write( 'moldens = mol.obasis.compute_grid_density_dm(mol.get_dm_full(), grid.points) \n' )
    f.write( 'wpart = MBISWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens) \n')
    f.write( 'wpart.do_charges() \n')
    f.write( '# Write the result to a file \n')
    f.write( "np.savetxt('hortonCharges.txt', wpart['charges']) \n")

    f.close()

def GenerateDDEC6File(): # Generate the template for calling Chargemol and making DDEC6 charges

    f = open("job_control.txt",'w')
    f.write( '<atomic densities directory complete path> \n' )
    f.write( '/home/bkelly08/Programs/chargemol_09_26_2017/atomic_densities/ \n' )
    f.write( '</atomic densities directory complete path> \n\n')
    f.write( '<input filename>\n' )
    f.write( '{0}.wfx \n'.format(molname) )
    f.write( '</input filename> \n\n' )
    f.write( '<charge type> \nDDEC6 \n</charge type> \n\n' )
    f.write( '<compute BOs> \n.false. \n</compute BOs>' )
    
    f.close()
    
def GenerateWFXFile():
    # Gaussian will generate the wfx file when it is called
    filename = "coords"
    
    GenerateZMatrixFile(filename)
 
    cmd = 'module load gaussian; g16 zMatrixInput.dat; module unload gaussian'
    p = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0] 
    
def GenerateZMatrixFile(filename):
    
    PrintXYZ(charge_replace, coords, filename + ".xyz")    # creates xyz file with coordinates
    ConvertXYZ2ZMat(filename)                     # creates Zmat file that is by default ".com"
    
    zMat = CopyZMatrix(filename + ".com")

    f = open("zMatrixInput.dat",'w')
    f.write( ' %chk={0}.chk \n '.format(molname) )
    f.write( '%mem=1000MB\n' )
    f.write( '%nproc=12 \n' )
    f.write( '#P {0}/{1} geom=connectivity guess=mix CHARGE DENSITY=CURRENT \n'.format(QMtheory,QMbasis) )    # PW91PW91
    f.write( '# scf=(fermi,conver=8,maxcycle=400) density=current output=wfx \n\n' )
    f.write( 'Truly witty remark \n\n' )
    f.write( '{0} {1} \n'.format(sumQQ, mult) )
    for line in zMat:
        f.write( '{0} \n'.format( str(line) ) )
        
    if background == True:
        for i, item in enumerate(solCoords) :
            f.write('{0:7.7} {1:7.7} {2:7.7} {3:7.7} \n'.format(solCoords[i][0], solCoords[i][1], solCoords[i][2], chargeGroup[i] ) )
    f.write(' \n')
    f.write( '{0}.wfx \n\n'.format( molname ) )
    f.close()
    
    
def ConvertXYZ2ZMat(filename):
    cmd = 'module load gaussian; newzmat -ixyz {0:s}; module unload gaussian'.format(filename)
    p = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0]
    
def CopyZMatrix(file):
    f = open(file,'r')

    zMatrixStuff = []
    
    for line in f:
        ln = line.split(",")
        if len(ln) > 1 :#ln[0] == sumQQ and ln[1] == mult:
            break
    for line in f:
        zMatrixStuff.append(line.strip())
        
    f.close()
    return zMatrixStuff
    
def PrintXYZ(atomNames, coords, filename):

    f = open(filename,'w')

    for i in range(len(atomNames)):
        f.write('{0} {1:7.7} {2:7.7} {3:7.7} \n'.format(atomNames[i][:1], coords[i][0], coords[i][1], coords[i][2]))
    f.write(' \n')
    
    f.close()
def GetDDEC6ChargesFromFile(): # Scan the output file, retrieve partial charges for DDEC6

    f = open("DDEC6_even_tempered_net_atomic_charges.xyz",'r')

    for line in f:
        if len( line.strip().split()) > 1:
            break
    iter = -1
    for line in f:
        iter += 1
        if iter == ( len( charges )  ):
            break
        charges[iter] = float( line.split()[4] )
    f.close()
    
def MakeMoleculeList(filename = "aqueous.g96"):
    # read the .g96 file and label what index number each molecule begins and ends at.
    f = open(filename, 'r')
    molList = []
    molCount = 0
    for line in f:
        if "VELOCITY" in line:
            break
        if len(line.split()) == 7:
            if int(line.split()[0]) == (molCount + 1): 
                molCount += 1
                molList.append( [ line.split()[3],line.split()[3] ] ) 
            elif molCount > 1:
                indx = len(molCount) 
                molList[indx] = [ molList[indx][0],int( line.split()[3] ) ] 
    f.close()
    molList = np.array( molList )  
    return molList 

def Make_itp_Template_4_newCharges(): # makes itp template
    
    files = os.listdir(os.getcwd())

    for file in files:

        if file.endswith(".itp") and "temp" not in file.lower(): 
            print("Oh yeah, we are triggered now!")
            # this is the droid we were looking for #
            fi=open(file,'r')
            fo = open(file.split("_")[0] + "_TEMP_GMX.itp",'w') # this assumes solute name is first part 
                                                                # of file name, and it is followed by _

            for line in fi:
                if '[ atoms ]' in line: 
                    fo.write(line)
                    break
                fo.write(line)

            for line in fi:
                if '[ bonds ]' in line:
                    fo.write(line) 
                    break

                if len(line.split(';')[0]) > 5:
                    atomName = line.split()[4]
                    dummyName = atomName + "C"
                    l = line.split()

                    lineout = '     {}   {}   {}    {}     {}     {}     {}     {} \n'.format(l[0], l[1],l[2],l[3],l[4],l[5],dummyName,l[7])
                else:
                    lineout = line
                fo.write(lineout)

            for line in fi:
                fo.write(line)

            fi.close()
            fo.close()



atomDict = {"H":1,"He":2,"Li":3,"Be":4, "B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,"Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,"K":19,"Ca":20} # needed for am1bcc
    
#############################################################
# Load saved np.array which has atom indexes for molecules
#############################################################
try:
    molList = np.load(input.dir + "/mArrayFile.npy")
except FileNotFoundError:
    molList = MakeMoleculeList()   # make list the manual way, pyjob.py didn't save it for us. (probably an old version got called?)
          
###############################
#
# Need to implement method for
# Getting atom names from file
#
###############################
""" We can find the dummy holder names for the charges in the dummy template
We scan the dummy topology until we find "name"_dummy, then scan the strings where the charges should be
"""
Make_itp_Template_4_newCharges() # make molname + '_TEMP_GMX.itp'

dummytemplate = molname + '_TEMP_GMX.itp'

f = open(dummytemplate,'r')

#Calculate charge on molecule
charge_replace=[]
for line in f:
    if '[ atoms ]' in line: 
        break
for line in f:
    if '[ bonds ]' in line: # we have scanned all atoms, and are now in the bonds section, terminate.
        break

    if MOL in line.split(';')[0] and len(line.split()) > 7:
        charge_replace.append( str(line.split()[6]) )

f.close()

BoxSize = float( FindLength( goutput ) ) * 10.0
cutoff = min(cutoff,BoxSize/2)  # Angstrom

######################################################
#
#         Get atom coordinates from GROMACS
#
###################################################### 

data=[]
solCoords=[] # store solvent coords
molGroup=[]
massAtom=[]
massSol = []

f = open(goutput,'r') #sys.stdin
############################
#
# Get Solute coords and COM
#
############################
stopRead = len(charge_replace) + 15  # the number 15 is just a dummy safety number, it should not be needed
for counter, line in enumerate(f):
    if counter > stopRead:
        break
    if MOL in line: 
        data.append(line)
        massSol.append( float( findMass( line.split()[2] ) ) )

datasplit=[]
coords=np.zeros( (len( charge_replace ),3) )
#coords = np.ascontiguousarray( coords )

for  x in data:              # split stats into chunks
    y = x.split()
    datasplit.append(y)

for i in range(len(charge_replace)):
    k=-1
    for j in range(4,7):
        k += 1
        coords[i,k] = float(datasplit[i][j])

coords = coords * 10.0         # convert gromacs coordinates from nanometer to Angstroms 

solCOM = np.zeros(( 3 ))                            # solute center-of-mass
solCOM = COM( coords,massSol, len(massSol) )

f.close()

############################
#
# Get Solvent coords and COM
#
############################
"""
Create temporary arrays holding coords and atom types and charges as we scan file.
Stop once all atoms of a solvent molecule are read in.
Calculate COM
Calculate distance between molecule COM and solute molecule COM
if within cutoff, save coords and charges to send to gaussian

Continue now with the next solvent molecule
"""


numSolv = 0
numRemoved = 0
numOutside = 0
buffer = []
solCoords = []
chargeGroup = []
elementType = []
solvCOM = np.zeros((3))
f=open(goutput,'r')

# I am not particularly proud of the spaghetti style for loop I have here, but... so it goes.

for counter, line in enumerate(f):
    if counter < 5:
        pass
    if "VELOCITY" in line:
        break
    if ("sol" or "wat" or "tip3p" or "spc") in line.lower(): # get X,Y,Z coords for all solvent atoms

        if len(buffer) < 1:
            buffer.append( line )

        elif len( buffer ) > 0 and ( line.split()[0] == buffer[-1].split()[0]):

            buffer.append(line)
        else:

            tempCoords = []
            tempMass = []
            tempCharge=[]
            tempAtomType=[]

            for count, item in enumerate(buffer):
                tempCoords.append( [float(item.split()[4])*10.0,float(item.split()[5])*10.0,float(item.split()[6])*10.0] ) 
                tempMass.append( float( findMass( item.split()[2] ) ) )
                """ get charges """
                if "o" in item.split()[2].lower():
                    tempCharge.append( q_oxygen )
                    tempAtomType.append("O")
                elif "h" in item.split()[2].lower():
                    tempCharge.append( q_hydrogen )
                    tempAtomType.append("H")
                else:
                    print("No solvent charge found, aborting")
                    tempCharge.append( oops ) # this will cause gaussian to abort for us. 
            numpArray = np.zeros( ( len(tempCoords),3) )
            for count, item in enumerate(tempCoords):
                numpArray[count,0] = float(tempCoords[count][0])
                numpArray[count,1] = float(tempCoords[count][1])
                numpArray[count,2] = float(tempCoords[count][2])
            tempMass   = np.asarray( tempMass )
            solvCOM = COM( numpArray, tempMass, len( tempMass ) )
            dist = DIST( solvCOM, solCOM, BoxSize )

            if int(Lambda) > 20:
                accept = False
                if dist < cutoff and dist < COMoverlap:
                    accept = AtomCOMCheck(coords,numpArray, BoxSize, overlap)

                elif dist < cutoff and dist > COMoverlap:
                    accept = True #numRemoved += 1
                else:
                    numOutside += 1
                                        
                if accept: #dist < cutoff and dist > overlap:  # protect against overlaps at nearly decouples states
                    numSolv += 1
                    solCoords += tempCoords
                    chargeGroup += tempCharge
                    elementType += tempAtomType                    
                else:
                    numRemoved += 1
            else:
                if dist < cutoff:
                    numSolv += 1
                    solCoords += tempCoords
                    chargeGroup += tempCharge 
                    elementType += tempAtomType
                else: 
                    numOutside += 1
                
            buffer = []
            buffer.append( line )  

##############################################
#
#  Scale MM background charges
#
##############################################
chargeGroup = np.asarray( chargeGroup )
chargeGroup *= (1.0 - mmScale) #
    
######################################################
#
#     Update GAUSSIAN Input File with new coords
#     this uses "from shutil import copyfile"
#
###################################################### 
oldtemplate = molname + '_GMX.itp'

f = open(oldtemplate,'r')

#Calculate charge on molecule, this is mostly for the case of ions, which are not part of this paper.
# This avoids hardcoding the charge on each molecule.
sumQQ = 0
for line in f:
    if '[ atoms ]' in line: 
        break
for line in f:
    if '[ bonds ]' in line: 
        break
    line = line.split(';')[0]
    if MOL in line and len(line.split()) > 7:

        sumQQ += float(line.split()[6])
f.close()

sumQQ = int(np.rint(sumQQ))

##################################################
#
#      Calculate Gaussian charges 
#         mulliken, mbs, esp, hi, ci, cm5, 
#      mbis and ddec6 have special call later, mbis uses this call though (ddec6 does not)
#
##################################################

if chargeMethod.lower() != "ddec6" and chargeMethod.lower() != "bcc":
    template = inputname + '_template.gjf'
    duplicate = inputname + '.' + end

    f = open(duplicate,'w')

    f.write('%mem=1000mb\n')
    f.write('%nproc=6\n')
    f.write('%chk=molecule\n')
    f.write(' \n')

    if background == True and chargeMethod == "resp":
        f.write('#T {0}/{1} pop={2} Charge density=current iop(6/33=2) iop(6/42=6) iop(6/50=1)\n'.format(QMtheory, QMbasis,pop))
    elif background == True:
        f.write('#T {0}/{1} pop={2} Charge density=current \n'.format(QMtheory, QMbasis,pop))
    else:
        f.write('#T {0}/{1} pop=ESP SCRF=(Solvent=Water)\n'.format(QMtheory, QMbasis)) # use implicit solvent

    f.write(' \n')
    f.write('Remark Section, Witty No Doubt \n')
    f.write(' \n')
    f.write('{0} {1} ! Molecule specification\n'.format(sumQQ,mult))

    for i in range(len(charge_replace)):
        f.write('{0} {1:7.7} {2:7.7} {3:7.7} \n'.format(charge_replace[i][:1], coords[i][0], coords[i][1], coords[i][2]))
    f.write(' \n')

    ######################################################
    #
    #     Print Solvent XYZ and partial charges
    #
    ###################################################### 

    if background == True:
        for i, item in enumerate(solCoords) :
            f.write('{0:7.7} {1:7.7} {2:7.7} {3:7.7} \n'.format(solCoords[i][0], solCoords[i][1], solCoords[i][2], chargeGroup[i] ) )
        f.write(' \n')

    if chargeMethod.lower() == "resp":
        """ Gaussian will now generate the .gesp file needed for resp fitting """
        f.write("molName" + ".gesp")
        f.write(' \n \n'.format())
        f.write("molName" + ".gesp")
        f.write(' \n'.format())
    f.close() 
    ######################################################
    #
    #               CALL GAUSSIAN FROM PYTHON
    #
    ######################################################  

    cmd = 'module load gaussian; {0} {1} ; module unload gaussian'.format(QMprogram, duplicate) 
    p = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0] 
elif "bcc" in chargeMethod.lower():
    # call QM program in Antechamber to get charges
    sqmInput = "sqmInput"
    f = open(sqmInput,'w')
    f.write("&qmmm\n")
    f.write( "qm_theory = 'AM1', \n" )
    f.write(' qmcharge = {0:d},\n'.format(sumQQ) )
    f.write(' maxcyc = {0:d},\n'.format( 0 ) )   # this stops sqm from optimizing geometry and using that geometry in charge calculations.
    f.write(' qmmm_int = {0:d},\n'.format( 1 ) )
    f.write(' / \n')
    for i in range(len(charge_replace)):
        f.write('{0} {1} {2:7.7} {3:7.7} {4:7.7} \n'.format(atomDict[ charge_replace[i][:1] ],charge_replace[i][:1], coords[i][0], coords[i][1], coords[i][2]))
    
    if background == True:
        f.write('#EXCHARGES\n')
        for i, item in enumerate(solCoords) :
            f.write('{0} {1} {2:7.7} {3:7.7} {4:7.7} {5:7.7} \n'.format(atomDict[ elementType[i] ], elementType[i],solCoords[i][0], solCoords[i][1], solCoords[i][2], chargeGroup[i] ) )
        f.write('#END')
    f.close()
    # duplicate the pdb file, update coordinates. This is used later by Antechamber to make a *.ac file """
    # Shouldn't actually be necessary to update coordinates, am1bcc shouldn't use them, it just adds an empirical charge correction """
    tempPDB=[]
    index = 0

    f=open(molname + ".pdb",'r')

    for line in f:
        if ("HETATM" in line) or ("ATOM" in line):
            templine = line.split()
            templine[5] = coords[index][0]
            templine[6] = coords[index][1]
            templine[7] = coords[index][2]
            if len(templine[2].strip()) == 3: # modilfy 4th column
                newline='{:<7} {:>3} {:>2} {:>4} {:>5} {:11.3f} {:7.3f} {:7.3f}'.format(templine[0],templine[1],templine[2],templine[3],templine[4],float(templine[5]),float(templine[6]),float(templine[7]))
            else:
                newline='{:<7} {:>3} {:>2} {:>5} {:>5} {:11.3f} {:7.3f} {:7.3f}'.format(templine[0],templine[1],templine[2],templine[3],templine[4],float(templine[5]),float(templine[6]),float(templine[7]))
            index += 1
        else:
            newline = line
        tempPDB.append( newline.rstrip() )
    f.close()
    sqmpdb = molname + "sqm"
    f=open(sqmpdb + ".pdb",'w')
    # write out pdb file for later use
    for item in tempPDB:
        f.write(item + "\n")
    f.close()    
######################################################
#
#     Extract charges from Gaussian output
#      
######################################################  
charges = np.zeros((len(charge_replace)))

if chargeMethod.lower() == "resp":
    """ This updates the mol2 file with the resp charges"""
    cmd = 'module load gcc/5.4.0; module load amber/18; antechamber -i {0:s}.gesp -fi gesp -o {1:s}.mol2 -fo mol2 -c {2:s} -at {3:s} -pf y'.format("molName","molName", chargeMethod, ffType)
    p = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0] 
    print(p.decode())  
    """ get the resp charges from the mol2 file """
    file = open("molName.mol2",'r')
    for line in file:
        if "@<TRIPOS>ATOM" in line:
            break
    for line in file:
        if "@<TRIPOS>BOND" in line:
            break
        charges[int(line.split()[0])-1] = line.split()[8] 
    file.close()
elif chargeMethod.lower() == "mbis":
    # Minimum Basis Iterative Stockholder charges

    # convert the (Gaussian?) checkpoint to a formatted checkpoint file
    cmd = 'module load gaussian; formchk {0:s}.chk {1:s}.fchk; module unload gaussian'.format("molecule","molecule")
    p1 = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0]
    #print( p1.decode() )

    GenerateHortonFile("mbis") # generates the... Horton ... file... it is kind of obvious..., takes as input the name of the python script to be produced
    # Now run Horton
    cmd = 'source ~/ENV/bin/activate; export PYTHONPATH=~/.local/lib/python2.7/site-packages/:$PYTHONPATH;python mbis.py'
    p = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0]  
   
    # A text file should now be present in the folder with the charges. It should also be called hortonCharges.txt
    file = open("hortonCharges.txt",'r')
    for iter, charge in enumerate(file):
        charges[iter] = format(float(charge.split()[0] ), ' 20.20f' ) # can customize with , ' 10.8f') etc...
    file.close()    
elif chargeMethod.lower() == "ddec6":
    print("Calling DDEC6 charges")
    GenerateWFXFile()
    GenerateDDEC6File()
    cmd = 'module load gcc/6.4.0 openmpi/2.1.1; ./Chargemol_09_02_2017_linux_parallel'
    p = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0] 
    print( p.decode() )
    GetDDEC6ChargesFromFile()
elif "bcc" in chargeMethod.lower():
    # call sqm 
    print("Running sqm")
    cmd = 'module load gcc/5.4.0; module load openmpi/2.1.1; module load amber/18;sqm -O -i {}   -o {}'.format(sqmInput,"sqmOutput")
    p = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0] 
    print(p.decode()) 
    print("Finished running sqm")
    
    ###########################
    #  Calculate am1bcc charges
    ###########################
    # First we need to generate a *.ac for am1-bcc 
    cmd = 'module load gcc/5.4.0; module load amber/18;antechamber -i {0:s}.pdb -fi pdb -o {1:s}.ac -fo ac -c {2:s} -at {3:s} -pf y'.format(sqmpdb,sqmpdb, chargeMethod, ffType)
    p = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0]
    print( p.decode() )
    # get the Mulliken charges from sqm output, and put in the *.ac file 
    sqmCharge = []
    f = open("sqmOutput",'r')
    for line in f:
        if "Mulliken Charge" in line:
            break
    for line in f:
        if "Total Mulliken Charge" in line:
            break
        sqmCharge.append(line.split()[2])

    sqmCharge = np.asarray( sqmCharge )
    index = 0
    tempAC = []
    f.close()
    # modfy .ac file with mulliken charges from sqm 
    f = open(sqmpdb + ".ac",'r')
    for line in f:
        if ("HETATM" in line) or ("ATOM" in line):
            templine = line.split()
            newline='{:<7} {:>3} {:>2} {:>5} {:>5} {:>11} {:>7} {:>7} {:9.6f} {:>9}'.format(templine[0],templine[1],templine[2],templine[3],templine[4],templine[5],templine[6],templine[7],float(sqmCharge[index]),templine[9])
            index += 1
        else:
            newline = line
        tempAC.append( newline.rstrip() )
    f.close()
    f = open("acNewCharge.ac",'w')
    for line in tempAC:
        f.write(line + "\n")
    f.close() 
    """ Now we need to run am1bcc to get the bond corrected charges """
    cmd = 'module load gcc/5.4.0; module load openmpi/2.1.1; module load amber/18;am1bcc -i {}.ac -f ac -o {}.ac -j 5'.format("acNewCharge","am1bcc")
    p = sub.Popen(cmd, shell=True, stderr = sub.STDOUT, stdout = sub.PIPE).communicate()[0] 
    print(p.decode()) 

    """ Now get charges from am1bcc.mol2 """
    f=open("am1bcc.ac",'r')
    index = 0
    for line in f:
        if "ATOM" in line:
            charges[index] = float( line.split()[8] ) 
            index += 1
    f.close()
    
else:
    gauss_output = inputname + '.log'

    filename = open(gauss_output)            # this is the log file output by Gaussian
    loop = True
    count = 0
    while loop and count < 10: # sometimes ESP is output more than once, we need the last one (G4 puts out 2... 10 is just a safety net)

        count += 1
        if pop == 'ESP':
            for line in filename:
                if 'ESP charges:' in line: 
                    gout = []
                    break
            
            for line in filename:  # This keeps reading the file
                if 'Sum of ESP charges' in line: 
                    break
                if ("" == line):
                    loop = False # reached EOF, exit

                gout.append(line)
    
        elif 'MBS' in pop:
            for line in filename:
                if 'Mulliken charges:' in line:
                    gout = []
                    break

            for line in filename:
                if 'Sum of Mulliken Charges' in line:
                    break
                if ("" == line):
                    loop = False # reached EOF, exit

                gout.append(line)

        elif pop == "CM5" or pop == "Hirshfeld":
            for line in filename:
                if ('Hirshfeld charges' in line) or ('CM5 charges' in line):
                    gout = []
                    break

            for line in filename:
                if 'Tot' in line:
                    break
                if ("" == line):
                    loop = False # reached EOF, exit

                if "Q-H" not in line:
                    gout.append(line)
        # Iterative Hirshfeld
        elif 'hi' in chargeMethod.lower() or 'ci' in chargeMethod.lower():
            for line in filename:
                if ('Iterated Hirshfeld charges, spin densities' in line):
                    gout = []
                    break

            for line in filename:
                if 'Tot' in line:
                    break
                if ("" == line):
                    loop = False # reached EOF, exit

                if "Q-H" not in line:
                    gout.append(line)
 
    goutsplit=[]

    for  x in gout:              # split stats into chunks
        y = x.split()
        if len(y) > 2:      # first few lines read in are not needed and are only single elements i.e. [i]
            goutsplit.append(y)
            # print(y)
 
    for i in range(len(charge_replace)):
        if "ci" in chargeMethod.lower() or "cm5" in chargeMethod.lower(): # take the I charge version
            charges[i] = float(goutsplit[i][7])
        else:
            charges[i] = float(goutsplit[i][2])
 
######################################################
#
#           Place charges in Gromacs *.itp file
#    First copy template.itp (has dummy charge names
#               for easy replacement
######################################################

gromacs_itp = molname + '_DUMMY_GMX.itp'
template_itp= molname + '_TEMP_GMX.itp'
copyfile(template_itp, gromacs_itp) # take copy template and call it name stored as gromacs_itp

filename = open(gromacs_itp)            # this is the log file output by Gaussian

for i in range( len(charge_replace) ) :
    with fileinput.FileInput(gromacs_itp, inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace(charge_replace[i], str(charges[i]) ), end='')

######################################################
#
#      Output charges to text file for histogram
#
######################################################
filename="charges.txt"

file=open(filename,"a")
file.writelines(["%s " % item for item in charges])
file.writelines(["\n"])
file.close()

######################################################
#
#    That should do it, the Gromacs topology file
#    should contain updated charges. We can now run
#    grompp to make a new *.tpr file and then
#    run gmx mdrun etc... to do our next batch of steps
#    before updating the charges again.
#
######################################################
#
#               A Quote to end with: 
#
#  "I feel more like I do now than I did when I first got here."
#                - anonymous man in a bar in Utah
#
#######################################################
