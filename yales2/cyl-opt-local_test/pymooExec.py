# from pymooCFD.runPreProc import runPreProc
# runPreProc()

#from pymooCFD.optPreProc.gen0Map import gen0Map
#gen0Map()

# from pymooCFD.util.handleData import archive
# archive('dump', 'archive')

# from pymooCFD.util.handleData import archive
# archive('dump')

from pymooCFD.runOpt import runOpt
runOpt(restart=True, hasTerminated=False)

# from pymooCFD import runPostProc
# runPostProc()

# from pymooCFD.util.handleData import compressDir
# compressDir('/dump')
