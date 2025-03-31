

lines = open( 'smiles_original.txt', "r" ).readlines()[::2]
file = open("smiles.txt", "w")
for line in lines:
    file.write(line)
file.close()