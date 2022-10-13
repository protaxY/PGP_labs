import sys
import argparse
import random

n = [2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14]

for k in range(len(n)):
    print(n[k])
    with open(str(n[k])+'.data', 'wb') as fp:

        fp.write(n[k].to_bytes(4, "little"))
        fp.write(n[k].to_bytes(4, "little"))

        for i in range(n[k]):
            for j in range(n[k]):
                r = int(random.uniform(0, 255)).to_bytes(1, "little")
                g = int(random.uniform(0, 255)).to_bytes(1, "little")
                b = int(random.uniform(0, 255)).to_bytes(1, "little")
                w = (0).to_bytes(1, "little")
                fp.write((r+g+b+w))