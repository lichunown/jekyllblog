import os,re

filename = '2018-12-25-Deterministic-Policy-Gradient-Algorithms.md'

f = open(filename, encoding='utf8')
data = f.readlines()
f.close()
out = ''
doublenum = 0
for line in data:
    if line=='$$\n':
        doublenum += 1
        if doublenum % 2 == 0:
            out += '$$\n\n'
        else:
            out += '\n$$\n'
    elif '$' in line:
        out += line.replace('$','$$').replace('$$$$','$$')
    else:
        out += line
        
with open('aa.md', 'w') as f:
    f.write(out)