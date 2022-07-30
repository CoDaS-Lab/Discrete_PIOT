# Discrete Probabilistic Inverse Optimal Transport
Paper accepted to ICML 2022: 
https://proceedings.mlr.press/v162/chiu22b.html  
  
Folders:  
./SRC : source codes for performing PIOT simulations  
./synthetic : contains python codes for running batch jobs, Jupyter Notebooks for simulations and plots, and pickle files for data in the paper  

Required packages and version  
\----------------------------  
conda  
matplotlib                3.3.4  
numpy                     1.19.1  
pandas                    1.1.1   
pickle  
python                    3.8.5  
pytorch                   1.4.0   
scikit-learn              0.24.1  
scipy                     1.5.2   
seaborn                   0.11.1  
tqdm                      4.54.0  

### How to run batch jobs:
1. Check \*.py files in ./synthetic/input_files/ and tune the parameters in it (slurm only)
2. Generate batch job files with "python FILE_NAME.py" inside ./synthetic/input_files/ 
3. Submit jobs by "sh ./synthetic/submit_all.sh" in ./synthetic
