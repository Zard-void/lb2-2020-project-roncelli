# Custom function to remap secondary structures to simplified alphabet (C-H-E)

def ss_mapper(structure): 
    mapped_structure = []
    position = int()
    for secondary in structure:
        if secondary == ' ' or secondary == 'T' or secondary == 'S' or secondary == '-':
             secondary = 'C'
        elif secondary == 'H' or secondary == 'G' or secondary == 'I':
            secondary = 'H'
        elif secondary == 'B' or secondary == 'E':            
            secondary = 'E'
        else:
            if secondary == '\n':
                raise Exception('Character is a newline. Did you remember to rstrip?')
            else:
                raise Exception('Aminoacid in position ' + str(position) + ' is ' + "'" + secondary + "'")
        position += 1
        mapped_structure.append(secondary)
    mapped_structure = ''.join([str(elem) for elem in mapped_structure]) 
    return mapped_structure
