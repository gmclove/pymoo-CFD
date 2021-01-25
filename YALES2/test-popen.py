from subprocess import check_output

out = check_output(['sbatch', './cases/base-case/'])
print(out)