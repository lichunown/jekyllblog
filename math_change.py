import sys

filename = './Deterministic Policy Gradient Algorithms笔记.md'
outname = ''

def change(filename, outname):
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
            out += line.replace('$','\n$$\n').replace('$$$$','$$')
        else:
            out += line
    with open(outname, 'w', encoding='utf8') as f:
        f.write(out)
        

if __name__=='__main__':
    arglen = len(sys.argv) - 1
    if arglen == 2:
        change(*sys.argv[1:])
    if arglen == 1:
        filename = sys.argv[1]
        change(filename, filename)
    