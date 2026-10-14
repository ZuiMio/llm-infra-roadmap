

import os
import shutil
import glob
import sys
import argparse
import re
import math
import random
import statistics

os.getcwd()      # Return the current working directory
os.chdir('/server/accesslogs')   # Change current working directory
os.system('mkdir today')   # Run the command mkdir in the system shell

dir(os) # List all the attributes of the os module
help(os) # Show the help of the os module

shutil.copyfile('data.db', 'archive.db')    # 'archive.db'

shutil.move('/build/executables', 'installdir') # 'installdir'

glob.glob('*.py')   # ['primes.py', 'random.py', 'quote.py']


# File demo.py
print(sys.argv)

parser = argparse.ArgumentParser(
    prog='top',
    description='Show top lines from each file')
parser.add_argument('filenames', nargs='+')
parser.add_argument('-l', '--lines', type=int, default=10)
args = parser.parse_args()
print(args)

for filename in args.filenames:
    print(f"=== {filename} ===")

    with open(filename, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.lines:
                break
            print(line, end="")


for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        print("空行", file=sys.stderr)
    else:
        print(line, file=sys.stdout)


re.findall(r'\bf[a-z]*', 'which foot or hand fell fastest')
re.sub(r'(\b[a-z]+) \1', r'\1', 'cat in the the hat')
'tea for too'.replace('too', 'two')


math.cos(math.pi / 4)

math.log(1024, 2)

random.choice(['apple', 'pear', 'banana'])

random.sample(range(100), 10)   # 重複なしでサンプリング

random.random()    # [0.0, 1.0) の区間内のランダムな float

random.randrange(6)    # range(6) からランダムに選ばれた integer

data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
statistics.mean(data)

statistics.median(data)

statistics.variance(data)