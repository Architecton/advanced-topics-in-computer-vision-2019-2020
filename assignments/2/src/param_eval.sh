LANG=en_US
LC_NUMERIC="en_US.UTF-8"

# for alpha in $(seq 0 0.001 0.1)
# do
#     printf "alpha=%.3f\n" $alpha
#     for seq in $(ls ../data/vot2014) 
#     do
#         python3.7 run_tracker.py --tracker ms --seq $seq --n_bins 30 --alpha $alpha
#     done
#     python3.7 res_stats.py --param $alpha
#     rm ../results/results.txt
# done

for n_bins in {10..50}
do
    printf "n_bins=%d\n" $n_bins
    for seq in $(ls ../data/vot2014) 
    do
        python3.7 run_tracker.py --tracker ms --seq $seq --n_bins $n_bins --alpha 0.006
    done
    python3.7 res_stats.py --param $n_bins
    rm ../results/results.txt
done

