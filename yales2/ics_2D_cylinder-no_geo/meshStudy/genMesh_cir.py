import gmsh
import sys

def genMesh(indDir, geoVar, meshVar):
    projName = '2D_cylinder'

    dom_D = 20

    cylD = geoVar

    # meshSizeMax = 0.2
    meshSF = meshVar[0]
    curveNN = meshVar[1]
    ####################################################################################################################
    # Before using any functions in the Python API, Gmsh must be initialized:
    gmsh.initialize()
    #gmsh.clear()
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
    centPt = gmsh.model.occ.addPoint(0, 0, 0)
    botPt = gmsh.model.occ.addPoint(0, -dom_D/2, 0)
    topPt = gmsh.model.occ.addPoint(0, dom_D/2, 0)
    inlet = gmsh.model.occ.addCircleArc(botPt, centPt, topPt)
    outlet = gmsh.model.occ.addCircleArc(topPt, centPt, botPt)
    domLoop = gmsh.model.occ.addCurveLoop([inlet, outlet]) # creates 2-D entity
    # add circle to rectangular domain to represent cylinder
    cylCir = gmsh.model.occ.addCircle(0, 0, 0, cylD/2)  # 1-dim. entity
    # use 1-D circle to create curve loop entity
    cylLoop = gmsh.model.occ.addCurveLoop([cylCir]) # creates 2-D entity
    # create plane surface between rectangle and circle
    dom = gmsh.model.occ.addPlaneSurface([domLoop, cylLoop])

    # We finish by synchronizing the data from OpenCASCADE CAD kernel with the Gmsh model:
    gmsh.model.occ.synchronize()
    ####################################################################################################################
    #################################
    #    Physical Group Naming      #
    #################################
    gmsh.model.addPhysicalGroup(2, [dom])
    grpTag = gmsh.model.addPhysicalGroup(1, [inlet])
    gmsh.model.setPhysicalName(1, grpTag, 'x0')
    grpTag = gmsh.model.addPhysicalGroup(1, [outlet])
    gmsh.model.setPhysicalName(1, grpTag, 'x1')
    grpTag = gmsh.model.addPhysicalGroup(1, [cylCir])
    gmsh.model.setPhysicalName(1, grpTag, 'cyl')
    ####################################################################################################################
    #################################
    #           MESHING             #
    #################################
    # To determine the size of mesh elements, Gmsh locally computes the minimum of
    # 1) the size of the model bounding box;
    # 2) if `Mesh.MeshSizeFromPoints' is set, the mesh size specified at
    #    geometrical points;
    # 3) if `Mesh.MeshSizeFromCurvature' is positive, the mesh size based on
    #    curvature (the value specifying the number of elements per 2 * pi rad);
    # 4) the background mesh size field;
    # 5) any per-entity mesh size constraint.
    # This value is then constrained in the interval [`Mesh.MeshSizeMin',
    # `Mesh.MeshSizeMax'] and multiplied by `Mesh.MeshSizeFactor'.  In addition,
    # boundary mesh sizes (on curves or surfaces) are interpolated inside the
    # enclosed entity (surface or volume, respectively) if the option
    # `Mesh.MeshSizeExtendFromBoundary' is set (which is the case by default).
    #
    # When the element size is fully specified by a background mesh size field (as
    # it is in this example), it is thus often desirable to set
    # Mesh.MeshSizeExtendFromBoundary = 0;
    # Mesh.MeshSizeFromPoints = 0;
    # Mesh.MeshSizeFromCurvature = 0;

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

    # Set minimum and maximum mesh size
    #gmsh.option.setNumber('Mesh.MeshSizeMin', meshSizeMin)
    #gmsh.option.setNumber('Mesh.MeshSizeMax', meshSizeMax)

    # Set mesh size factor
    gmsh.option.setNumber('Mesh.MeshSizeFactor', meshSF)

    # Set number of nodes along cylinder wall
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', curveNN)
    # gmsh.option.setNumber('Mesh.MeshSizeFromCurvatureIsotropic', 1)

    # Set size of mesh at every point in model
    # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), meshSize)


    # gmsh.model.mesh.setTransfiniteCurve(cylCir, 150, coef=1.1)
    # We can then generate a 2D mesh...
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.generate(2)
    ####################################################################################################################
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

    # scale factor, num. nodes along curve
meshVar = [1, 400]
    # cyl. diameter
geoVar = 1
genMesh('.', geoVar, meshVar)
