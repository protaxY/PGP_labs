
import sys
import argparse
import random

# parser = argparse.ArgumentParser()
# parser.add_argument("n")
# args = parser.parse_args()

n = [2**5, 2**10, 2**15, 2**20, 2**25]

for k in range(len(n)):
    print(n[k])
    with open(str(n[k])+'.txt', 'w') as fp:
        fp.write(str(n[k])+'\n')
        for i in range(n[k]):
            fp.write(str(random.uniform(-100, 100))+' ')
        fp.write('\n')
        for i in range(n[k]):
            fp.write(str(random.uniform(-100, 100))+' ')