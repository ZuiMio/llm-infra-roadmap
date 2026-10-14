

import threading, zipfile

class AsyncZip(threading.Thread):
    def __init__(self, infile, outfile):
        super().__init__()
        self.infile = infile
        self.outfile = outfile

    def run(self):
        with zipfile.ZipFile(self.outfile, 'w', zipfile.ZIP_DEFLATED) as f:
            f.write(self.infile)
        print('Finished background zip of:', self.infile)

background = AsyncZip('mydata.txt', 'myarchive.zip')
background.start()
print('The main program continues to run in foreground.')

background.join()    # Wait for the background task to finish
print('Main program waited until background was done.')


import logging
logging.debug('Debugging information')
logging.info('Informational message')
logging.warning('Warning:config file %s not found', 'server.conf')
logging.error('Error occurred')
logging.critical('Critical error -- shutting down')

import weakref, gc

class A:
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return str(self.value)

# 普通の dict
a = A(10)
normal = {}
normal['primary'] = a
del a
gc.collect()
print(normal['primary'])   # 10  ← まだ生きている

# WeakValueDictionary
a = A(20)
weak = weakref.WeakValueDictionary()
weak['primary'] = a
wp = weak['primary']
del a
gc.collect()
print(weak.get('primary')) # None ← 消えている
print(wp)                  # 20  ← まだ生きている




