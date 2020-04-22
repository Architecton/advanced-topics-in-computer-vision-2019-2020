LANG=en_US
LC_NUMERIC="en_US.UTF-8"

for param in $(seq 0 1 100)
do
    for seq in $(ls ../data/vot2014) 
    do
        python3.7 run_tracker.py --seq $seq --rotate --training-iter $param
    done
    printf "done %d\n" $param
    echo -en '#\n' >> ../results/results.txt
done
