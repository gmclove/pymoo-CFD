import numpy as np
from scipy.interpolate import griddata
import os
import h5py
import warnings


class GridInterp:
    def __init__(self, dumpPrefix, dom_xmin, dom_xmax, dom_ymin, dom_ymax, 
                 t_begin, t_end, t_resol, x_resol = 100j):
        # new grid to interpolate onto
        x_resol = x_resol
        dx = abs(dom_xmax - dom_xmin)
        dy = abs(dom_ymax - dom_ymin)
        y_resol = x_resol*(dy/dx)
        self.grid_x, self.grid_y = np.mgrid[dom_xmin:dom_xmax:x_resol, 
                                            dom_ymin:dom_ymax:y_resol]
        self.dumpPrefix = dumpPrefix
        self.t_begin = t_begin
        self.t_end = t_end
        self.t_resol = t_resol
        

    def getGrids(self, caseDir): #, t_resol):
         # sort dump files to find latest mesh and solution files
        path = os.path.join(caseDir, 'dump')
        ents = os.listdir(path)
        ents.sort()

        t_begin_str = self.getTimeStr(self.t_begin)
        t_end_str = self.getTimeStr(self.t_end)
        # t_begin_str = self.getTimeStr(t_begin)
        # t_end_str = self.getTimeStr(t_end)
        i_init = ents.index(f'{self.dumpPrefix}.sol{t_begin_str}.xmf')
        i_final = ents.index(f'{self.dumpPrefix}.sol{t_end_str}.xmf')
        ents_bnd = ents[i_init:i_final]
        
        # store latest mesh from dump before t_begin
        for ent in ents[:i_init]:
            if ent.endswith('.mesh.h5'):
                latestMesh = ent
    
        
        solnFiles = []
        for ent in ents_bnd:
            if ent.endswith('.mesh.h5'):
                latestMesh = ent
            if ent.endswith('.sol.h5'):
                latestSoln = ent
                solnFiles.append([latestSoln, latestMesh])
        
        n_solns = len(solnFiles)
        # use t_resol to reduce number of solution files 
        if t_resol > n_solns:
            warnings.warn('t_resol too high, make t_resol <= {n_solns}')
        t_indices = np.linspace(0, n_solns-1, self.t_resol)
        t_indices = [int(t_i) for t_i in t_indices]
        solnFiles = [solnFiles[i] for i in t_indices]
        
        grids = []
        for soln in solnFiles:
            solnDat = soln[0]
            solnMesh = soln[1]
            grid = self.getGrid(caseDir, solnDat, solnMesh)
            grids.append(grid)
        
        grids = np.array(grids, dtype=object)
        return grids
    
    def getGrid(self, caseDir, solnDat, solnMesh):
        with h5py.File(os.path.join(caseDir, 'dump', solnDat), 'r') as f:
            ls_phi = f['Data']['LS_PHI'][:]
            t = f['Data']['TOTAL_TIME'][:]
        with h5py.File(os.path.join(caseDir, 'dump', solnMesh), 'r') as f:
            coor = f['Coordinates']['XYZ'][:][:, :2]
        
        grid = griddata(coor, ls_phi, (self.grid_x, self.grid_y), 
                        method='cubic')
        return grid, t

    def meanDiff(self, grids1, grids2):
        mean_diffs = []
        for i, grid1 in enumerate(grids1):
            grid2 = grids2[i]
            t1 = grid1[1]
            t2 = grid2[1]
            if t1 != t2:
                t_diff = abs(t1-t2)
                if t_diff > 1e-10:
                    print('GRID TIMES DO NOT MATCH: {t1}, {t2} off by {t_diff} seconds')
                # warnings.warn(f'GRID TIMES DO NOT MATCH: off by {t_diff} seconds')
                # raise Exception('GRID COMPARISON FAILED: grid times do not match')
            grid2 = grid2[0]
            grid1 = grid1[0]
            mean_diff = np.mean(abs(grid1 - grid2))
            mean_diffs.append(mean_diff)
        mean_diff_all = np.mean(mean_diffs)
        return mean_diff_all
    
    def getTimeStr(self, t):
        t_str = str(int(t*1000))
        t_str = t_str.zfill(6)
        return t_str



# import time    
# #### TEST #####
# dom_xmin = -0.5
# dom_xmax = 0.5
# dom_ymin = -0.5
# dom_ymax = 1.5

# t_begin = 0.5
# t_end = 1
# t_resol = 2 # evaluate t_resol time steps

# gridInterp = GridInterp('droplet_convection', 
#                         dom_xmin, dom_xmax, dom_ymin, dom_ymax,
#                         t_begin, t_end, t_resol
#                         )
# #################################
# ####### HQ DATA EXTRACT #########
# caseDir = '.'
# start = time.time()
# hq_grids = gridInterp.getGrids(caseDir) #, t_begin, t_end)
# print('hq_grids calc time: ', time.time()-start)
# #######################################
# ####### AMR DATA EXTRACT ##########
# caseDir = '../../../base_case'
# initSolnFile = 'droplet_convection.sol000000_1.sol.h5'
# with h5py.File(os.path.join(caseDir, 'dump', initSolnFile), 'r') as f:
#     t_init = f['Data']['TOTAL_TIME'][:]
# start = time.time()
# amr_grids = gridInterp.getGrids(caseDir) #, t_begin, t_end)
# print('amr_grids calc time: ', time.time()-start)

# # print(amr_grids)

# print('amr first time step: ', amr_grids[0,1])
# print('hq first time step: ', hq_grids[0,1])
# amr_grids[:,1] = amr_grids[:,1] - t_init
# print(amr_grids[0,1])
# start = time.time()
# meanDiff = gridInterp.meanDiff(hq_grids, amr_grids)
# print('meanDiff calc time: ', time.time()-start)
