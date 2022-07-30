import os
import numpy as np
exclude_machines = ''

''' MCMC hyper-parameters '''
idx_i = 6
idx_j = 2

# MCMC parameters
shift = 0.1
lam = 320.0
num_burn_in = 10000
num_lag = 1000
num_sample = 10000

count = 0

fo_submit = open('submit.sh', 'w+')

if not os.path.exists('./scripts/{}_{}'.format(str(idx_i), str(idx_j))):
	os.makedirs('./scripts/{}_{}'.format(str(idx_i), str(idx_j)))

for value in range(100, 25000, 250):
	print(value)

	filename = '{}.sh'.format(str(count))

	fo = open('./scripts/{}_{}'.format(str(idx_i), str(idx_j))+'/'+filename, 'w+')
	fo.write('#!/bin/bash -l\n')
	fo.write('\n')
	fo.write('#SBATCH --nodes=1\n')
	#fo.write('#SBATCH --nodelist=hal0115\n')
	#fo.write('#SBATCH --exclude={}\n'.format(exclude_machines))
	fo.write('#SBATCH --ntasks=1\n')
	fo.write('#SBATCH --cpus-per-task=1\n')
	fo.write('#SBATCH --ntasks-per-node=1\n')
	fo.write('#SBATCH --mem=6000\n')
	fo.write('#SBATCH --time=48:00:00\n')
	fo.write('#SBATCH --export=ALL\n')
	fo.write('#SBATCH --output=R-{}_{}_{}.%j.out\n'.format(str(idx_i), str(idx_j), str(value)))
	fo.write('#SBATCH --error=R-{}_{}_{}.%j.err\n'.format(str(idx_i), str(idx_j), str(value)))
	fo.write('# -cwd\n')
	fo.write('conda activate PIOT\n')
	fo.write('python3 PIOT_EU_migration.py ./input_files/scripts/{}_{}/{}.in\n'.format(str(idx_i), str(idx_j), str(count)))
	fo.close()

	in_filename = './scripts/{}_{}/{}.in'.format(str(idx_i), str(idx_j), str(count))
	fo_in = open(in_filename, 'w+')
	fo_in.write('{},{},{},{},{},{},{},{},{},./data_C/missing_value/{}_{}/'.format(str(idx_i), str(idx_j), str(value),\
		str(shift), str(lam), str(num_burn_in), str(num_lag), str(num_sample), str(count), str(idx_i), str(idx_j) ) )
	fo_in.close()

	fo_submit.write('sbatch ./input_files/scripts/{}_{}/{}.sh\n'.format(str(idx_i), str(idx_j), str(count)))
	
	count += 1

fo_submit.close()
