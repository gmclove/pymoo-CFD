from distutils.dir_util import copy_tree

caseDir = './cases/ics_2D_cylinder'
indDir = './cases/ics_2D_cylinder/gen0/ind0'
copy_tree(caseDir + '/base-case', indDir)
