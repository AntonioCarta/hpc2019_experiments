for i in $(seq 1 1 21); do
    /opt/conda/bin/python src/benchmark_rnn.py --threads=$i
done
