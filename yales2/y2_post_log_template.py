#!/usr/bin/env python3

time = []                  # Total time - all runs


import sys,os
yales2_home = os.environ['YALES2_HOME']
if yales2_home=='':
   print("Couldn't find YALES2_HOME env var")
   sys.exit()
sys.path.append(yales2_home+'/tools')
import y2_post_log

my_cases =  [
               "./output/RUN_01/solver01_rank0000.log",\
               "./output/RUN_02/solver01_rank0000.log",\
               "./output/RUN_03/solver01_rank0000.log"
            ]
output_folder = "./temporal"
y2_post_log.plot_log(my_cases,output_folder,save_pdf=True,show_figs=False,send_by_mail=True,fsize=(16,12))
