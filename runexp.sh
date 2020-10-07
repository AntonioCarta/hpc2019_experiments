export KMP_SETTINGS
KMP_SETTINGS=TRUE
ch-run --set-env=latest_image/config.env --cd=/home/carta/hpc2019_experiments latest_image/antonio.latest /opt/conda/bin/python /home/carta/hpc2019_experiments/src/benchmark_sparse.py
