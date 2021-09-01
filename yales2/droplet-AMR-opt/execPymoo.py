def main():
    #from pymooCFD.preProcOpt import meshStudy
    #meshStudy(restart=False)    

    #from pymooCFD.preProcOpt import runGen1
    #runGen1(restart=False)
    
    #from pymooCFD.setupOpt import xl, xu
    # from pymooCFD.preProcOpt import runCornerCases
    # runCornerCases(xl, xu)
    
    #from pymooCFD.preProcOpt import runCornerCases
    #runCornerCases(xl, xu)

    from pymooCFD.runOpt import runOpt
    runOpt(restart=False)
    
    # from pymooCFD.setupCFD import runCase
    # runCase('preProcOpt/maxTimeSim', [0.05, 10])
    
    #from pymooCFD.setupCFD import postProc
    #postProc('preProcOpt/maxTimeSim', [0.05, 10])
    
    # from pymooCFD.util.handleData import compressDir
    # compressDir('/dump')
    
    # from pymooCFD.util.handleData import loadCP
    # alg = loadCP()
    # alg.display.do(alg.problem, alg.evaluator, alg)
    
    #from pymooCFD.util.handleData import archive
    #archive('dump', background=False)
    
    #from pymooCFD.util.handleData import compressDir
    #compressDir('dump/chackpoint.npy')


if __name__ == '__main__':
    main()
