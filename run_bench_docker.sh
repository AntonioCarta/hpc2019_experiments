for i in $(seq 1 5 81); do
    /opt/conda/bin/python src/benchmark_rnn.py --threads=$i
done
