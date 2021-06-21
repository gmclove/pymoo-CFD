import os
from pymooIN import procLim, job_name


def findKeywordLine(kw, file_lines):
    kw_line = -1
    kw_line_i = -1

    for line_i in range(len(file_lines)):
        line = file_lines[line_i]
        if line.find(kw) >= 0:
            kw_line = line
            kw_line_i = line_i

    return kw_line, kw_line_i


os.system('source activate pymoo-CFD')
os.system('rm -rfv gen*/ checkpoint* output.dat __pycache__')
os.system('rm -rfv base_case/output.dat base_case/dump/')  # base_case/'+dataFile)
# os.remove()
# os.rmdir()

########################################################################################################################
###### MOO JOBSLURM ######
##########################
# change jobslurm.sh to correct procLim
with open('./jobslurm.sh', 'r') as f_orig:
    job_lines = f_orig.readlines()

# use keyword 'cd' to find correct line
keyword = 'cd'
keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
# create new string to replace line
newLine = 'cd ' + os.getcwd() + '\n'
job_lines[keyword_line_i] = newLine

# use keyword 'cd' to find correct line
keyword = '#SBATCH --nodes='
keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
# create new string to replace line
newLine = keyword + str(procLim) + '\n'
job_lines[keyword_line_i] = newLine

# change slurm job-name
keyword = 'job-name='
keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
# create new string to replace line
newLine = keyword_line[:keyword_line.find(keyword)] + keyword + job_name + '\n'
job_lines[keyword_line_i] = newLine

with open('./jobslurm.sh', 'w') as f_new:
    f_new.writelines(job_lines)

# ########################################################################################################################
# ##### BASE CASE JOBSLURM ######
# ###############################
# # change jobslurm.sh to correct directory and change job name
# with open('./base_case/jobslurm.sh', 'r') as f_orig:
#     f_lines = f_orig.readlines()
# # individuals jobslurms
# keyword = '#SBATCH --nodes='
# keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
# # create new string to replace line
# newLine = keyword + str(nProc) + '\n'
# job_lines[keyword_line_i] = newLine
#
# # use keyword 'cd' to find correct line
# keyword = 'cd'
# keyword_line, keyword_line_i = findKeywordLine(keyword, f_lines)
# # create new string to replace line
# newLine = 'cd ' + os.getcwd() + '\n'
# f_lines[keyword_line_i] = newLine
#
# keyword = 'job-name='
# keyword_line, keyword_line_i = findKeywordLine(keyword, f_lines)
# # create new string to replace line
# newLine = keyword_line[:keyword_line.find(keyword)] + keyword + jobName + '\n'
# f_lines[keyword_line_i] = newLine
# with open('./base_case/jobslurm.sh', 'w') as f_new:
#     f_new.writelines(f_lines)
