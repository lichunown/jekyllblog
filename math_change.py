import sys, os

file_path = './for_myself/'
outname_path = './_posts'


def change(filename, outname):
    f = open(filename, encoding='utf8')
    data = f.readlines()
    f.close()
    
    out = ''
    doublenum = 0
    for line in data:
        line = line.replace('img/in-post/', '/img/in-post/')

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
        file_name, out_name = sys.argv[1:]
        change(os.path.join(file_path, file_name), os.path.join(outname_path, out_name))
    if arglen == 1:
        filename = sys.argv[1]
        change(os.path.join(file_path, file_name), os.path.join(outname_path, file_name))
    else:
        for filename in os.listdir(file_path):
            if filename.split('.')[-1] == 'md' or filename.split('.')[-1] == 'markdown':
                print(f"change {os.path.join(file_path, filename)} to {os.path.join(outname_path, filename)}")
                change(os.path.join(file_path, filename), os.path.join(outname_path, filename))
    