import subprocess

files = [
    'fig_lanczos_polynomial',
    'fig_level_curves',
    'fig_sqrt',
    'fig_pcr',
    'fig_pcp',
    'fig_sqrt_fp',
    'fig_log_quadform',
    'fig_step_quadform',
]

for file in files:
    subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 {file}.ipynb',shell=True)

