import os
import numpy as np
#exclude_machines = ''

''' MCMC hyper-parameters '''
size = [100*i for i in range(1, 11)]

count = 0

fo_submit = open('submit.sh', 'w+')

if not os.path.exists('./scripts_run_time/'):
	os.makedirs('./scripts_run_time/')

for nr in size:
	nc = nr

	fo = open('./scripts_run_time/{}_{}.sh'.format(str(nr), str(nc) ), 'w+')
	fo.write('#!/bin/bash -l\n')
	fo.write('\n')
	fo.write('#SBATCH --nodes=1\n')
	#fo.write('#SBATCH --exclude={}\n'.format(exclude_machines))
	fo.write('#SBATCH --ntasks=1\n')
	fo.write('#SBATCH --cpus-per-task=1\n')
	fo.write('#SBATCH --ntasks-per-node=1\n')
	fo.write('#SBATCH --mem=6000M\n')
	fo.write('#SBATCH --time=48:00:00\n')
	fo.write('#SBATCH --export=ALL\n')
	fo.write('#SBATCH --output=R_RT-{}_{}.%j.out\n'.format(str(nr), str(nc)))
	fo.write('#SBATCH --error=R_RT-{}_{}.%j.err\n'.format(str(nr), str(nc)))
	fo.write('# -cwd\n')
	fo.write('conda activate PIOT\n')
	fo.write('python3 PIOT_run_time.py ./input_files/scripts_run_time//{}_{}.in\n'.format(str(nr), str(nc), ))
	fo.close()

	in_filename = './scripts_run_time/{}_{}.in'.format(str(nr), str(nc))
	fo_in = open(in_filename, 'w+')
	fo_in.write('{},{},./data_run_time/'.format(str(nr), str(nc) ) )
	fo_in.close()

	fo_submit.write('sbatch ./input_files/scripts_run_time/{}_{}.sh\n'.format(str(nr), str(nc)))
	
	count += 1

fo_submit.close()
