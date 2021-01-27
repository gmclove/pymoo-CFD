from subprocess import check_output


# test base case
out = check_output(['sbatch', './cases/ics_2D_cylinder-no_geo/base-case/jobslurm.sh'])
batchID = int(out[20:])
print(batchID)
waiting = True
while waiting:
    out = check_output('squeue | grep --count %i || :' % batchID, shell=True)
    # print(out)
    print(int(out))
    if int(out) == 0:
        waiting = False
