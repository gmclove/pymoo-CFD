import gmsh
import sys
import math

projName = '2D_cylinder'

roomH = 5
roomL = 13

cylD = 1
cylCenterX = 2

meshSize = 0.05

########################################################################################################################
# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize()
# gmsh.clear()
# By default Gmsh will not print out any messages: in order to output messages
# on the terminal, just set the "General.Terminal" option to 1:
gmsh.option.setNumber("General.Terminal", 1)
# Next we add a new model (if gmsh.model.add() is not called a new
# unnamed model will be created on the fly, if necessary):
gmsh.model.add(projName)
# We can log all messages for further processing with:
gmsh.logger.start()
########################################################################################################################
# create room rectangle
rectTag = gmsh.model.occ.addRectangle(0, 0, 0, roomL, roomH)
# print('room tag: %i' % roomTag)
# add circle to rectangular domain to represent cylinder
cirTag = gmsh.model.occ.addCircle(cylCenterX, roomH / 2, 0, cylD / 2)  # 1-dim. entity
# use 1-D circle to create curve loop entity
cirLoopTag = gmsh.model.occ.addCurveLoop([cirTag])
# print('cyl. tag: %i' % cylTag)
# create plane surface between rectangle and circle
domainTag = gmsh.model.occ.addPlaneSurface([rectTag, cirLoopTag])
# remove original rectangular surface
gmsh.model.occ.remove([(2, rectTag)])

# We finish by synchronizing the data from OpenCASCADE CAD kernel with the Gmsh model:
gmsh.model.occ.synchronize()
########################################################################################################################
#################################
#    Physical Group Naming      #
#################################
# print(gmsh.model.getBoundary([(2, domainTag)]))
domBWall = 1
domRWall = 2
domTWall = 3
domLWall = 4
cylWall = 5

grpTag = 1
gmsh.model.addPhysicalGroup(1, [domLWall])
gmsh.model.setPhysicalName(1, grpTag, 'x0')
grpTag += 1
gmsh.model.addPhysicalGroup(1, [domRWall])
gmsh.model.setPhysicalName(1, grpTag, 'x1')
grpTag += 1
gmsh.model.addPhysicalGroup(1, [domTWall])
gmsh.model.setPhysicalName(1, grpTag, 'y0')
grpTag += 1
gmsh.model.addPhysicalGroup(1, [domBWall])
gmsh.model.setPhysicalName(1, grpTag, 'y1')
grpTag += 1
gmsh.model.addPhysicalGroup(1, [cylWall])
gmsh.model.setPhysicalName(1, grpTag, 'cyl')
########################################################################################################################
#################################
#           MESHING             #
#################################
# To identify points or other bounding entities you can take advantage of the
# `getEntities()', `getBoundary()' and `getEntitiesInBoundingBox()' functions:

# Assign number of nodes on each curve
# NN = 40
# for c in gmsh.model.getEntities(1):
#     gmsh.model.mesh.setTransfiniteCurve(c[1], NN)

#
# # NN_domTBWalls = 150
# # gmsh.model.mesh.setTransfiniteCurve(domTWall, NN_domTBWalls, coef=1.3)
# # gmsh.model.mesh.setTransfiniteCurve(domBWall, NN_domTBWalls, coef=1.3)
#
# #
# # # create mesh construction lines
# # NN_constL = 50
# # pt1 = gmsh.model.occ.addPoint(0, roomH/2, 0)
# # pt2 = gmsh.model.occ.addPoint(cylCenterX - cylD/2, roomH/2, 0)
# # constL = gmsh.model.occ.addLine(pt1, pt2)
# # gmsh.model.occ.synchronize()
# # gmsh.model.mesh.setTransfiniteCurve(constL, NN_constL, coef=1.2)
# #


# gmsh.model.mesh.setSizeFromBoundary(2, domainTag, 5)
# gmsh.model.mesh.setTransfiniteAutomatic(gmsh.model.getEntities(2))

# gmsh.model.mesh.createGeometry([])

# Set this to True to build a fully hex mesh:
#transfinite = True
# transfinite = False
# transfiniteAuto = False
#
# if transfinite:
#     NN = 30
#     for c in gmsh.model.getEntities(1):
#         gmsh.model.mesh.setTransfiniteCurve(c[1], NN)
#     for s in gmsh.model.getEntities(2):
#         gmsh.model.mesh.setTransfiniteSurface(s[1])
#         gmsh.model.mesh.setRecombine(s[0], s[1])
#         gmsh.model.mesh.setSmoothing(s[0], s[1], 100)
#     gmsh.model.mesh.setTransfiniteVolume(v1)
# elif transfiniteAuto:
#     gmsh.option.setNumber('Mesh.MeshSizeMin', 0.5)
#     gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)
#     # setTransfiniteAutomatic() uses the sizing constraints to set the number
#     # of points
#     gmsh.model.mesh.setTransfiniteAutomatic()
# else:
#     gmsh.option.setNumber('Mesh.MeshSizeMin', 0.05)
#     gmsh.option.setNumber('Mesh.MeshSizeMax', 0.05)
#

# Assign a mesh size to all the points:
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)

NN_cylWall = 120
gmsh.model.mesh.setTransfiniteCurve(cylWall, NN_cylWall)
#
NN_domTB = 250
gmsh.model.mesh.setTransfiniteCurve(domTWall, NN_domTB)#, coef=1.2)
gmsh.model.mesh.setTransfiniteCurve(domBWall, NN_domTB)#, coef=1.2)

NN_domLR = 200
gmsh.model.mesh.setTransfiniteCurve(domRWall, NN_domLR, meshType='Bump', coef=.5)
gmsh.model.mesh.setTransfiniteCurve(domLWall, NN_domLR, meshType='Bump', coef=.5)

# gmsh.option.
# We can then generate a 2D mesh...
gmsh.model.mesh.generate(1)
gmsh.model.mesh.generate(2)
# ... and save it to disk
gmsh.write(projName + '.msh22')

# Inspect the log:
# log = gmsh.logger.get()
# print("Logger has recorded " + str(len(log)) + " lines")
# gmsh.logger.stop()

# To visualize the model we can run the graphical user interface with
# `gmsh.fltk.run()'. Here we run it only if the "-nopopup" is not provided in
# the command line arguments:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
# This should be called when you are done using the Gmsh Python API:
gmsh.finalize()
