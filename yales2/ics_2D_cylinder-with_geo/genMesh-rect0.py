import gmsh
import sys
from os.path import join


def genMesh(path, geoVar):
    projName = '2D_cylinder'

    room_dx = 40
    room_dy = 20

    cylD = geoVar
    r = cylD / 2

    meshSizeMax = 0.25
    # meshSF = 0.4
    ####################################################################################################################
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
    ####################################################################################################################
    # Make sure "Recombine all triangular meshes" is unchecked so only triangular elements are produced
    gmsh.option.setNumber('Mesh.RecombineAll', 0)
    ####################################################################################################################
    # create room rectangle
    rectTag = gmsh.model.occ.addRectangle(-(1 / 6)
                                          * room_dx, -room_dy / 2, 0, room_dx, room_dy)
    # print('room tag: %i' % roomTag)
    # add circle to rectangular domain to represent cylinder
    cirTag = gmsh.model.occ.addCircle(0, 0, 0, cylD / 2)  # 1-dim. entity
    # use 1-D circle to create curve loop entity
    cirLoopTag = gmsh.model.occ.addCurveLoop([cirTag])
    # print('cyl. tag: %i' % cylTag)
    # create plane surface between rectangle and circle
    domainTag = gmsh.model.occ.addPlaneSurface([rectTag, cirLoopTag])
    # remove original rectangular surface
    gmsh.model.occ.remove([(2, rectTag)])
    # We finish by synchronizing the data from OpenCASCADE CAD kernel with the Gmsh model:
    gmsh.model.occ.synchronize()
    ####################################################################################################################
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
    gmsh.model.addPhysicalGroup(2, [domainTag])

    grpTag += 1
    gmsh.model.addPhysicalGroup(1, [domLWall])
    gmsh.model.setPhysicalName(1, grpTag, 'x0')
    grpTag += 1
    gmsh.model.addPhysicalGroup(1, [domRWall])
    gmsh.model.setPhysicalName(1, grpTag, 'x1')
    grpTag += 1
    gmsh.model.addPhysicalGroup(1, [domTWall])
    gmsh.model.setPhysicalName(1, grpTag, 'y0')
    # self.output.append('Best Drag [N]', algorithm.pop.get("F")[:, 0].min())
    grpTag += 1
    # self.output.append('Best Drag [N]', np.mean(algorithm.pop.get("F")[:, 1].min()))
    # if
    gmsh.model.addPhysicalGroup(1, [domBWall])
    gmsh.model.setPhysicalName(1, grpTag, 'y1')
    grpTag += 1
    gmsh.model.addPhysicalGroup(1, [cylWall])
    gmsh.model.setPhysicalName(1, grpTag, 'cyl')
    ####################################################################################################################
    #################################
    #           MESHING             #
    #################################
    # We could also use a `Box' field to impose a step change in element sizes
    # inside a box
    # boxF = gmsh.model.mesh.field.add("Box")
    # gmsh.model.mesh.field.setNumber(boxF, "VIn", meshSizeMax/10)
    # gmsh.model.mesh.field.setNumber(boxF, "VOut", meshSizeMax)
    # gmsh.model.mesh.field.setNumber(boxF, "XMin", cylD/3)
    # gmsh.model.mesh.field.setNumber(boxF, "XMax", cylD/3+cylD*10)
    # gmsh.model.mesh.field.setNumber(boxF, "YMin", -cylD)
    # gmsh.model.mesh.field.setNumber(boxF, "YMax", cylD)
    # # Finally, let's use the minimum of all the fields as the background mesh field:
    # minF = gmsh.model.mesh.field.add("Min")
    # gmsh.model.mesh.field.setNumbers(minF, "FieldsList", [boxF])
    #
    # gmsh.model.mesh.field.setAsBackgroundMesh(minF)

    # Set number of nodes along cylinder wall
    # gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 150)

    # Set size of mesh at every point in model
    # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)

    # start by specifying a distance field from the obstacle surface
    obstacles = [cylWall]
    f_dist_obst = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_obst, "CurvesList", obstacles)
    gmsh.model.mesh.field.setNumber(f_dist_obst, "NumPointsPerCurve", 400)

    # next step is to use a threshold function vary the resolution from these surfaces in the following way:
    # LcMax -                  /--------
    #                      /
    # LcMin -o---------/
    #        |         |       |
    #       Point    DistMin DistMax
    resolution = r / 10
    f_thres_obst = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thres_obst, "IField", f_dist_obst)
    # gmsh.model.mesh.field.setNumber(f_thres_obst, "SizeMin", resolution)
    # gmsh.model.mesh.field.setNumber(f_thres_obst, "SizeMax", 20 * resolution)
    gmsh.model.mesh.field.setNumber(f_thres_obst, "LcMin", resolution)
    gmsh.model.mesh.field.setNumber(f_thres_obst, "LcMax", 20 * resolution)
    gmsh.model.mesh.field.setNumber(f_thres_obst, "DistMin", 0.5 * r)
    gmsh.model.mesh.field.setNumber(f_thres_obst, "DistMax", r)

    # add a similar threshold at the inlet to ensure fully resolved inlet flow
    inlet = domLWall
    f_dist_in = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_in, "CurvesList", [inlet])
    f_thres_in = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thres_in, "IField", f_dist_in)
    gmsh.model.mesh.field.setNumber(f_thres_in, "LcMin", 5 * resolution)
    # gmsh.model.mesh.field.setNumber(f_thres_in, "SizeMin", 5 * resolution)
    gmsh.model.mesh.field.setNumber(f_thres_in, "LcMax", 10 * resolution)
    # gmsh.model.mesh.field.setNumber(f_thres_in, "SizeMax", 10 * resolution)
    gmsh.model.mesh.field.setNumber(f_thres_in, "DistMin", 0.1)
    gmsh.model.mesh.field.setNumber(f_thres_in, "DistMax", 0.5)

    # combine these fields by using the minimum field
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thres_obst, f_thres_in])
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    # Set minimum and maximum mesh size
    # gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
    # gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)

    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(2)
    ####################################################################################################################
    # ... and save it to disk
    gmsh.write(join(path, f'{projName}.msh22'))

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


# genMesh('base_case/meshes', 0.01517)  # 0.05 , 0.01517 -> Re=150
# genMesh('base_case/meshes', 0.05)  # 0.05 , 0.01517 -> Re=150
genMesh('.', 1)
