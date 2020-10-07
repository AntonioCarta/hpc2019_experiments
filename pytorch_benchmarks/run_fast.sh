###
### Pytorch with jemalloc
###

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
export LD_PRELOAD=/home/carta/lib/libjemalloc.so

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export $KMP_SETTING
echo -e "\n### using $KMP_SETTING"

vals_nthreads="1 2 4 7 14 28 56 112"
#vals_nthreads="1 2 4"
vals_kmp_blocktime="0 1 10 50 200"

for i in $vals_nthreads; do
    export OMP_NUM_THREADS=$i
    echo -e "### using OMP_NUM_THREADS=$OMP_NUM_THREADS"
    python -u benchmark_rnn.py
done
