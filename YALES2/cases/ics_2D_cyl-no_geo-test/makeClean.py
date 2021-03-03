import os

os.system('rm -rfv gen* checkpoint*')


def editJobslurm(wd, jobName):
    def findKeywordLine(kw, file_lines):
        kw_line = -1
        kw_line_i = -1

        for line_i in range(len(file_lines)):
            line = file_lines[line_i]
            if line.find(kw) >= 0:
                kw_line = line
                kw_line_i = line_i

        return kw_line, kw_line_i

    # change jobslurm.sh to correct directory and change job name
    with open(wd + '/jobslurm.sh', 'r') as f_orig:
        f_lines = f_orig.readlines()
    # use keyword 'cd' to find correct line
    keyword = 'cd'
    keyword_line, keyword_line_i = findKeywordLine(keyword, f_lines)
    # create new string to replace line
    newLine = 'cd ' + wd + '\n'
    f_lines[keyword_line_i] = newLine

    keyword = 'job-name='
    keyword_line, keyword_line_i = findKeywordLine(keyword, f_lines)
    # create new string to replace line
    newLine = keyword_line[:keyword_line.find(keyword)] + keyword + jobName + '\n'
    f_lines[keyword_line_i] = newLine
    with open(wd + '/jobslurm.sh', 'w') as f_new:
        f_new.writelines(f_lines)


# cwd = '~/Simulations/yales2/pymoo-CFD/YALES2/cases/ics_2D_cyl-no_geo-test'
cwd = os.getcwd()
editJobslurm(wd=cwd, jobName='test-moo')
editJobslurm(wd=cwd+'/base_case', jobName='base_case')
