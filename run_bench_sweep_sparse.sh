# export KMP_SETTINGS=TRUE  # Output OpenMP environment variables
export KMP_AFFINITY
export OMP_NUM_THREADS
export KMP_BLOCKTIME
export MKL_NUM_THREADS
export MKLDNN_VERBOSE

MKLDNN_VERBOSE=1

PY_MAIN=./src/benchmark_sparse.py

vals_nthreads="1 2 4 7 14 28 56 112"
#vals_nthreads="1 2 4"
vals_kmp_blocktime="0 1 10 50 200"
#for i in $vals_nthreads; do
#    for j in $vals_kmp_blocktime; do
#      echo OMP_THREADS=$i, KMP_BLOCKTIME=$j, AFFINITY=0
#      OMP_NUM_THREADS=$i
#      MKL_NUM_THREADS=$i
#      KMP_BLOCKTIME=$j
#      /opt/conda/bin/python $PY_MAIN
#    done
#done

###################################
## KMP_AFFINITY
###################################
#KMP_AFFINITY=granularity=fine,compact,1,0
#for i in $vals_nthreads; do
#    for j in $vals_kmp_blocktime; do
#      echo OMP_THREADS=$i, KMP_BLOCKTIME=$j, AFFINITY=1
#      OMP_NUM_THREADS=$i
#      MKL_NUM_THREADS=$i
#      KMP_BLOCKTIME=$j
#      /opt/conda/bin/python $PY_MAIN
#    done
#done

##################################
# KMP_AFFINITY
##################################
KMP_AFFINITY=noverbose,warnings,respect,granularity=core,none
for i in $vals_nthreads; do
    for j in $vals_kmp_blocktime; do
      echo OMP_THREADS=$i, KMP_BLOCKTIME=$j, AFFINITY=2
      OMP_NUM_THREADS=$i
      MKL_NUM_THREADS=$i
      KMP_BLOCKTIME=$j
      /opt/conda/bin/python $PY_MAIN
    done
done

###################################
## NUMA
###################################
#echo "NUMA Control"
#echo "KMP_AFFINITY SET."
#KMP_AFFINITY=granularity=fine,compact,1,0
#for i in $vals_nthreads; do
#    for j in $vals_kmp_blocktime; do
#      echo OMP_THREADS=$i, KMP_BLOCKTIME=$j
#      OMP_NUM_THREADS=$i
#      MKL_NUM_THREADS=$i
#      KMP_BLOCKTIME=$j
#      numactl --cpunodebind=1 --membind=1 python /opt/conda/bin/python src/benchmark_rnn.py
#    done
#done