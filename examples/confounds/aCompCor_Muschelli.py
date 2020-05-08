import os
import pandas as pd
from jtnipyutil.confounds import aCompCor_Muschelli

# Calculate aCompCor for CSF/WM (WITHOUT APPLYING A TEMPORAL FILTER, AS IN FMRIPREP) then append the top 5 PCA
# components for each to the confounds.tsv file for each task.

base_dir = '/home/neuro/workdir/2020-05-07_nofilter_aCompCor'
data_dir = os.path.join(base_dir, 'data')
out_dir = os.path.join(base_dir, 'output')
subj_list = ['sub-001', 'sub-002', 'sub-003']
task_list = ['stress', 'wm']
ventricle_file = '/home/neuro/atlases/ALVIN_v1/ALVIN_mask_v1.img'

for subj in subj_list:
    for task in task_list:
        # extract PCA components using Muschelli 2014 method.
        aCompCor_Muschelli(data_dir, out_dir, subj, task, ventricle_file)

        # grab confounds file
        conf_file = os.path.join(data_dir, subj, 'func', subj+'_task-'+task+'_bold_confounds.tsv')
        conf_data = pd.read_csv(conf_file, sep='\t')

        # drop any prior PCA components, if they exist.
        conf_data = conf_data[conf_data.columns.drop(list(conf_data.filter(regex='^aCompCorWMnoFilter')))]
        conf_data = conf_data[conf_data.columns.drop(list(conf_data.filter(regex='^aCompCorCSFnoFilter')))]

        # add CSF top 5 PCA components to confounds file
        conf_data = pd.concat([conf_data,
            pd.read_csv(os.path.join(out_dir, subj, 'PCA', subj+'_task-'+task+'_CSF-aCompCor.csv')).iloc[:,0:5]],
            axis=1)
        # add WM top 5 PCA components to confound file
        conf_data = pd.concat([conf_data,
            pd.read_csv(os.path.join(out_dir, subj, 'PCA', subj+'_task-'+task+'_WM-aCompCor.csv')).iloc[:,0:5]],
            axis=1)
        # save new confounds file
        conf_data.to_csv(os.path.join(data_dir, subj, 'func', subj+'_task-'+task+'_bold_confounds.tsv'),
                        sep='\t', index=False)
