for i in $(seq 1 5 81); do
    python src/benchmark_rnn.py --threads=$i
done