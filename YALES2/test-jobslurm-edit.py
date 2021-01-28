from os import getcwd

def findKeywordLine(kw, file_lines):
    kw_line = -1
    kw_line_i = -1

    for line_i in range(len(file_lines)):
        line = file_lines[line_i]
        if line.find(kw) >= 0:
            kw_line = line
            kw_line_i = line_i

    return kw_line, kw_line_i


# omega = 5
# freq = 100
# # open and read YALES2 input file to array of strings for each line
# with open('./cases/ics_2D_cylinder/base-case/2D_cylinder.in', 'r') as f:
#     in_lines = f.readlines()
#     # find line that must change using a keyword
#     keyword = 'CYL_ROTATION_PROP'
#     keyword_line, keyword_line_i = findKeywordLine(keyword, in_lines)
#     # create new string to replace line
#     newLine = keyword + ' = ' + str(omega) + ' ' + str(freq) + '\n'
#     in_lines[keyword_line_i] = newLine
# with open('./cases/ics_2D_cylinder/base-case/2D_cylinder.in', 'w') as f_new:
#     f_new.writelines(in_lines)

ind = 0
gen = 0
indDir = './cases/ics_2D_cylinder-no_geo/base_case'
# change jobslurm.sh to correct directory
with open(indDir + '/jobslurm.sh', 'r') as f_orig:
    job_lines = f_orig.readlines()


# find cd line
# keyword = 'cd'
# keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
# # create new string to replace line
# i = keyword_line.find('base-case')
# print(i)
# print(keyword_line[i]) # = '/gen%i/ind%i' % (gen, ind)
# newLine = keyword_line[:keyword_line.find('base-case')] + 'gen%i/ind%i' % (gen, ind) + '\n'
# job_lines[keyword_line_i] = newLine

# find cd line
keyword = 'job-name='
keyword_line, keyword_line_i = findKeywordLine(keyword, job_lines)
# create new string to replace line
newLine = keyword_line[:keyword_line.find(keyword)] + keyword + 'gen%i.ind%i' % (gen, ind) + '\n'
job_lines[keyword_line_i] = newLine


with open(indDir + '/jobslurm.sh', 'w') as f_new:
    f_new.writelines(job_lines)
