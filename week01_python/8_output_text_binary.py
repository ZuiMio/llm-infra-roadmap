import reprlib
print(set('supercalifragilisticexpialidocious'))
print(reprlib.repr(set('supercalifragilisticexpialidocious')))


import pprint
t = [[[['black', 'cyan'], 'white', ['green', 'red']], [['magenta',
    'yellow'], 'blue']]]
pprint.pprint(t, width=30)

import textwrap
doc = """The wrap() method is just like fill() except that it returns
a list of strings instead of one big string with newlines to separate
the wrapped lines."""
print(textwrap.fill(doc, width=40))

import locale
loc = locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
print(loc)
conv = locale.localeconv()
x = 1234567.8
print(locale.format_string("%d", x, grouping=True))
print(locale.format_string("%s%.*f", (conv['currency_symbol'], conv['frac_digits'], x), grouping=True))


from string import Template
t = Template('${village}folk send $$10 to $cause.')
print(t.substitute(village='Nottingham', cause='the ditch fund'))
t = Template('Return the $item to $owner.')
d = dict(item='unladen swallow')
try:
    print(t.substitute(d))
except KeyError as e:
    print(t.safe_substitute(d))

import time, os.path
photofiles = ['img_1074.jpg', 'img_1076.jpg', 'img_1077.jpg']
class BatchRename(Template):
    delimiter = '%'
# fmt = input('Enter rename style (%d-date %n-seqnum %f-format):  ') # %n_%d%f
fmt = '%{n}_%d%f'
t = BatchRename(fmt)
date = time.strftime('%d%b%y')
for i, filename in enumerate(photofiles):
    base, ext = os.path.splitext(filename)
    newname = t.substitute(d=date, n=i, f=ext)
    print('{0} --> {1}'.format(filename, newname))


# import struct

# with open('week01_python/myfile.zip', 'rb') as f:
#     data = f.read()

# start = 0
# for i in range(3):                      # show the first 3 file headers
#     start += 14
#     fields = struct.unpack('<IIIHH', data[start:start+16])
#     crc32, comp_size, uncomp_size, filenamesize, extra_size = fields

#     start += 16
#     filename = data[start:start+filenamesize]
#     start += filenamesize
#     extra = data[start:start+extra_size]
#     print(filename, hex(crc32), comp_size, uncomp_size)

#     start += extra_size + comp_size     # skip to the next header

import zipfile

with zipfile.ZipFile('week01_python/myfile.zip', 'r') as z:
    for info in z.infolist():
        print(
            info.filename,
            hex(info.CRC),
            info.compress_size,
            info.file_size,
            info.compress_type,
            info.flag_bits,
        )

