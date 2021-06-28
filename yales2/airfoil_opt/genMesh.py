import gmsh
import sys
import numpy as np

import matplotlib.pyplot as plt

def genMesh(indDir, geoVar, meshSF):
    projName = '2D_airfoil'

    # domain size
    dom_dx = 20
    dom_dy = 10

    # extract geometric variables
    mu_x = geoVar[0]
    mu_y = geoVar[1]

    # airfoil "center"
    af_cent = [0, 0]
    # number of points used to discretize airfoil edge
    num_pt = 500
    af_x, af_y = joukowski_map(mu_x, mu_y, num_pt)

    # mid, top, bot, left, right = joukowski_map(mu_x, mu_y)
    top_y = 1
    bot_y = -1
    left_x = -2
    right_x = 3


    meshSizeMax = 0.15
    # meshSF = 0.4
    ####################################################################################################################
    # Before using any functions in the Python API, Gmsh must be initialized:
    gmsh.initialize()
    gmsh.clear()
    # By default Gmsh will not print out any messages: in order to output messages
    # on the terminal, just set the "General.Terminal" option to 1:
    gmsh.option.setNumber("General.Terminal", 1)
    # Next we add a new model (if gmsh.model.add() is not called a new
    # unnamed model will be created on the fly, if necessary):
    gmsh.model.add(projName)
    # We can log all messages for further processing with:
    gmsh.logger.start()
    ####################################################################################################################
    # Make sure "Recombine all triangular meshes" is unchecked so only triangular elements are produced
    gmsh.option.setNumber('Mesh.RecombineAll', 0)
    ####################################################################################################################
    # create domain rectangle
    dom = gmsh.model.occ.addRectangle(-(1/5)*dom_dx, -dom_dy/2, 0, dom_dx, dom_dy)

    af_pts = np.empty(len(af_x))
    for i in range(len(af_x)):
        af_pts[i] = gmsh.model.occ.addPoint(af_x[i], af_y[i], 0)
    af_lines = np.empty(len(af_pts)-1)
    for i in range(len(af_pts)-1):
        af_lines[i] = gmsh.model.occ.addLine(int(af_pts[i]), int(af_pts[i+1]))

    af_loop = gmsh.model.occ.addCurveLoop(af_lines)
    # create plane surface between rectangular domain and airfoil loop
    domain = gmsh.model.occ.addPlaneSurface([dom, af_loop])
    # remove original rectangular surface
    gmsh.model.occ.remove([(2, dom)])
    # We finish by synchronizing the data from OpenCASCADE CAD kernel with the Gmsh model:
    gmsh.model.occ.synchronize()
    ####################################################################################################################
    #################################
    #    Physical Group Naming      #
    #################################
    # print(gmsh.model.getBoundary([(2, domain)]))
    domBWall = 1
    domRWall = 2
    domTWall = 3
    domLWall = 4

    # yales2 gmsh reader requires physical group tags to start at 0
    # 2-D physical groups
    dim = 2
    grpTag = gmsh.model.addPhysicalGroup(dim, [domain])
    # gmsh.model.setPhysicalName(dim, grpTag, 'dom')
    # 1-D physical groups
    dim = 1
    grpTag = gmsh.model.addPhysicalGroup(dim, [domLWall])
    gmsh.model.setPhysicalName(dim, grpTag, 'x0')
    grpTag = gmsh.model.addPhysicalGroup(dim, [domRWall])
    gmsh.model.setPhysicalName(dim, grpTag, 'x1')
    grpTag = gmsh.model.addPhysicalGroup(dim, [domBWall])
    gmsh.model.setPhysicalName(dim, grpTag, 'y0')
    grpTag = gmsh.model.addPhysicalGroup(dim, [domTWall])
    gmsh.model.setPhysicalName(dim, grpTag, 'y1')

    af_tags = np.array(range(4, 4+num_pt))
    print(af_tags)
    grpTag = gmsh.model.addPhysicalGroup(dim, af_tags)
    gmsh.model.setPhysicalName(dim, grpTag, 'af')

    ####################################################################################################################
    #################################
    #           MESHING             #
    #################################
    # We could also use a `Box' field to impose a step change in element sizes
    # inside a box
    # boxF = gmsh.model.mesh.field.add("Box")
    # gmsh.model.mesh.field.setNumber(boxF, "VIn", meshSizeMax/5)
    # gmsh.model.mesh.field.setNumber(boxF, "VOut", meshSizeMax)
    # gmsh.model.mesh.field.setNumber(boxF, "XMin", af_cent[0])
    # gmsh.model.mesh.field.setNumber(boxF, "XMax", af_cent[0]+5)
    # gmsh.model.mesh.field.setNumber(boxF, "YMin", np.min(af_y)-0.4)
    # gmsh.model.mesh.field.setNumber(boxF, "YMax", np.max(af_y)+0.4)
    # # Finally, let's use the minimum of all the fields as the background mesh field:
    # minF = gmsh.model.mesh.field.add("Min")
    # gmsh.model.mesh.field.setNumbers(minF, "FieldsList", [boxF])
    #
    # gmsh.model.mesh.field.setAsBackgroundMesh(minF)

    # Set transfinite curve
    # gmsh.model.mesh.setTransfiniteCurve(af_loop, 2, meshType='Progression', coef=1.3)
    for tag in af_tags:
        gmsh.model.mesh.setTransfiniteCurve(tag,3)#, meshType='Progression', coef=1.3)


    # Set minimum and maximum mesh size
    # gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
    gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)

    # Set number of nodes along airfoil wall
    # gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 1000)

    # Set size of mesh at every point in model
    # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)

    # Set mesh size factor
    gmsh.option.setNumber('Mesh.MeshSizeFactor', meshSF)

    # We can then generate a 2D mesh...
    # gmsh.model.mesh.generate(1)
    gmsh.model.mesh.generate(2)
    ####################################################################################################################
    # To force Gmsh to save all elements,
    # whether they belong to physical groups or not, set `Mesh.SaveAll=1;'
    gmsh.option.setNumber('Mesh.SaveAll', 1)
    # ... and save it to disk
    gmsh.write(indDir + '/' + projName + '.msh22')

    # Inspect the log:
    log = gmsh.logger.get()
    print("Logger has recorded " + str(len(log)) + " lines")
    gmsh.logger.stop()

    # To visualize the model we can run the graphical user interface with
    # `gmsh.fltk.run()'. Here we run it only if the "-nopopup" is not provided in
    # the command line arguments:
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    # This should be called when you are done using the Gmsh Python API:
    gmsh.finalize()


def joukowski_map(mu_x, mu_y, num_pt):
    # center of circle in complex plane
    comp_cent = np.array([mu_x, mu_y])
    # radius of circle in complex plane /
    # distance from center to point (-1,0) in complex plane
    r = np.sqrt((comp_cent[0]-1)**2 + (comp_cent[1]-0)**2)

    # Circle coordinates calculations
    angle = np.linspace(0, 2*np.pi, num_pt) # 500 points along circle [0, 2*pi]
    comp_r = comp_cent[0] + r*np.cos(angle) # real coordinates along circle (horz.)
    comp_i = comp_cent[1] + r*np.sin(angle) # imaginary coordinates along circle (vert.)

    # Cartesian components of the Joukowsky transform
    x = ((comp_r)*(comp_r**2+comp_i**2+1))/(comp_r**2+comp_i**2)
    y = ((comp_i)*(comp_r**2+comp_i**2-1))/(comp_r**2+comp_i**2)
    # plt.plot(x,y)
    # plt.show()

    ########################################
    # change chord length to be from x=0 to 1
    # Compute the scale factor (actual chord length)
    c = np.max(x)-np.min(x)
    # Leading edge current position
    LE = np.min(x/c)
    # Corrected position of the coordinates
    x = x/c-LE # move the leading edge
    y = y/c

    # return 500 points that make up airfoil shape
    return x, y


genMesh('.', [-0.2, 0.075], 1)


'''
# Defintion of the search spaces limits
x_low = -0.3
x_high = -0.1
y_low = 0.0
y_high = 0.15

# Type of the constraints (including the boundary constraints)
constVal = [x_high, x_low, y_high, y_low]
compMode = ['leq', 'geq', 'leq', 'geq']
'''
