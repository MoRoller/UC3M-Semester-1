import time
import pandas as pd
import re
import matplotlib.pyplot as plt

def barplot(pat, values):
    x_values, y_values = zip(*values)  
    
    plt.style.use('ggplot')
    plt.bar([str(index) for index in x_values], y_values, color='#0504aa', alpha=0.5) 
        
    plt.xlabel('Protein ID')
    plt.xticks(rotation=90)
    plt.ylabel('Pattern Matches')
    plt.yticks(range(1, int(max(y_values)) + 1))
    plt.grid(True)
    plt.title(f'Proteins with most occurences of pattern {pat}')
    plt.show()


pattern = str(input('Please enter pattern: ')).upper()
print(f'Pattern to be searched = {pattern}')
print('____________________________________________________________________')
t0 = time.time()

proteins_df = pd.read_csv('proteins.csv')
proteins = proteins_df.values

matches = {}
occurences = 0

for index, sequence in proteins:
  if len([m.start() for m in re.finditer(pattern, str(sequence))]) > 0:   
    matches[index] = len([m.start() for m in re.finditer(pattern, str(sequence))])

t1 = time.time()
print(f'total execution time: {round(t1-t0, 3)} seconds')
print('____________________________________________________________________')

if matches:    
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    print(f'Protein with most occurences of pattern: ID {sorted_matches[0][0]}')
    print(f'Number of occurences: {sorted_matches[0][1]}')
    barplot(pat = pattern, values = sorted_matches[:10])
else:
    print(f'No matches found for pattern {pattern}')





