import random

n = [10**2, 10**3, 10**4, 10**5, 10**6, 10**7]

for k in range(len(n)):
    print(n[k])
    with open(str(k+2)+'.in', 'wb') as fp:
        fp.write(n[k].to_bytes(4, "little"))

        for i in range(n[k]):
            val = int(random.uniform(0, 4294967295)).to_bytes(4, "little")
            fp.write(val)