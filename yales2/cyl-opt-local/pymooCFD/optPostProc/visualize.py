



if pf is not None:
    if heatmap == True:
        from pymoo.visualization.heatmap import Heatmap
        plot = Heatmap(#figsize=(10,30),
                       # title=("Optimization", {'pad': 15}),
                       # bound=[0,1],
                       # order_by_objectives=0,
                       # y_labels=None,
                       # labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"],
                       # cmap="Greens_r"
                       )
        plot.add(F) # aspect = 0.2
        plot.save(f'{plotDir}/heatmap.png')

    if petalplot == True:
        from pymoo.visualization.petal import Petal

        plot = Petal(title = 'Petal Plot of Objectives',
                    # bounds=[0, 1],
                    cmap="tab20",
                    labels=["profit", "cost", "sustainability", "environment", "satisfaction", "time"],
                    title=("Solution A", {'pad': 20})
                    )
        plot.add(F)
        plot.save(f'{plotDir}/petalplot.png')
