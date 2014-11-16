#!/usr/bin/env python

import sys
from openeye import oechem
from openeye import oeiupac
from openeye import oeomega

#=============================================================================================
def modifySubstructureName(mol2file, name):
   """Replace the substructure name (subst_name) in a mol2 file.

   ARGUMENTS
     mol2file (string) - name of the mol2 file to modify
     name (string) - new substructure name

   NOTES
     This is useful becuase the OpenEye tools leave this name set to <0>.
     The transformation is only applied to the first molecule in the mol2 file.

   TODO
     This function is still difficult to read.  It should be rewritten to be comprehensible by humans.
   """

   # Read mol2 file.
   file = open(mol2file, 'r')
   text = file.readlines()
   file.close()

   # Find the atom records.
   atomsec = []
   ct = 0
   while text[ct].find('<TRIPOS>ATOM')==-1:
     ct+=1
   ct+=1
   atomstart = ct
   while text[ct].find('<TRIPOS>BOND')==-1:
     ct+=1
   atomend = ct

   atomsec = text[atomstart:atomend]
   outtext=text[0:atomstart]
   repltext = atomsec[0].split()[7] # mol2 file uses space delimited, not fixed-width

   # Replace substructure name.
   for line in atomsec:
     # If we blindly search and replace, we'll tend to clobber stuff, as the subst_name might be "1" or something lame like that that will occur all over. 
     # If it only occurs once, just replace it.
     if line.count(repltext)==1:
       outtext.append( line.replace(repltext, name) )
     else:
       # Otherwise grab the string left and right of the subst_name and sandwich the new subst_name in between. This can probably be done easier in Python 2.5 with partition, but 2.4 is still used someplaces.
       # Loop through the line and tag locations of every non-space entry
       blockstart=[]
       ct=0
       c=' '
       for ct in range(len(line)):
         lastc = c
         c = line[ct]
         if lastc.isspace() and not c.isspace():
           blockstart.append(ct)
       line = line[0:blockstart[7]] + line[blockstart[7]:].replace(repltext, name, 1)
       outtext.append(line)
       
   # Append rest of file.
   for line in text[atomend:]:
     outtext.append(line)
     
   # Write out modified mol2 file, overwriting old one.
   file = open(mol2file,'w')
   file.writelines(outtext)
   file.close()

   return

def create_molecule(name, filename):
    # Open output file.
    ofs = oechem.oemolostream()
    ofs.open(filename)

    # Create molecule.
    mol = oechem.OEMol()

    # Speculatively reorder CAS permuted index names
    str = oeiupac.OEReorderIndexName(name)
    if len(str)==0:
        str=name

    done = oeiupac.OEParseIUPACName(mol, str)

    # Create conformation.
    omega = oeomega.OEOmega()
    omega.SetIncludeInput(False) # don't include input
    omega.SetMaxConfs(1) # set maximum number of conformations to 1
    omega(mol)

    # Set molecule name
    mol.SetTitle(name)
    
    # Write molecule.
    oechem.OEWriteMolecule(ofs, mol)

    # Close stream.
    ofs.close()
    
    return
                 
if __name__ == "__main__":
    # Parse command-line arguments.
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("--name", dest="name", default=None, help="molecule name", metavar="NAME")
    parser.add_option("--outfile", dest="output_filename", default=None, help="output filename", metavar="OUTPUT_FILENAME")    

    # Parse command-line arguments.
    (options, args) = parser.parse_args()
    
    # Check arguments for validity.
    if not (options.name and options.output_filename):
        parser.error("Molecule name and output filename must be specified.")

    create_molecule(options.name, options.output_filename)
    modifySubstructureName(options.output_filename, 'MOL')

