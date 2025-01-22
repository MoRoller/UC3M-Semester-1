import time
import pandas as pd
import re
import matplotlib.pyplot as plt
from mpi4py import MPI


def barplot(pat, values):
    x_values, y_values = zip(*values)  # unpack to tuples 
    
    plt.style.use('ggplot')
    plt.bar([str(index) for index in x_values], y_values, color='#0504aa', alpha=0.5) # turning x_values (int) into strings
        
    plt.xlabel('Protein ID')
    plt.xticks(rotation=90)
    plt.ylabel('Pattern Matches')
    plt.yticks(range(1, int(max(y_values)) + 1))
    plt.grid(True)
    plt.title(f'Proteins with most occurences of pattern {pat}')
    plt.show()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


if rank == 0:
    pattern = str(input('Please enter pattern: ')).upper()
    print(f'Pattern to be searched = {pattern}')
    print('____________________________________________________________________')
    t0 = time.time()
else:
    pattern = None

proteins_df = pd.read_csv('proteins.csv')
proteins = proteins_df.values

comm.barrier()
pattern = comm.bcast(pattern, root = 0)

# chunk data evenly 
chunk_size = len(proteins) // size

# starting/ending index for each process
start = rank * chunk_size 
end = start + chunk_size

matches = {}
occurences = 0

for index, sequence in proteins[start:end]:
  if len([m.start() for m in re.finditer(pattern, str(sequence))]) > 0:  
    matches[index] = len([m.start() for m in re.finditer(pattern, str(sequence))])

# Gather the results from all processes ->  list of dictionaries
# for each process/chunk one dictionary
gathered_matches = comm.gather(matches, root=0)

if rank == 0:
    # Combine results from all processes into a single dictionary
    final_matches = {}
    for dic in gathered_matches:
        final_matches.update(dic)  # add dictionaries together
    #output: dictionary with all matches
        
    execution_time = time.time() - t0
    print(f'total execution time: {round(execution_time, 3)} seconds')
    print('____________________________________________________________________')

    if final_matches:    
        sorted_matches = sorted(final_matches.items(), key= lambda x: x[1], reverse=True)
        top10 = sorted_matches[:10]
        print(f'Protein with most occurences of pattern: ID {top10[0][0]}')
        print(f'Number of occurences: {top10[0][1]}')
        barplot(pat = pattern, values = sorted_matches[:10])
       
    else:
        print(f'No matches found for pattern {pattern}')
















