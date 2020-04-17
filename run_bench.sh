# for i in $(seq 1 5 81); do
for i in $(seq 1 1 21); do
    python src/benchmark_rnn.py --threads=$i
done
