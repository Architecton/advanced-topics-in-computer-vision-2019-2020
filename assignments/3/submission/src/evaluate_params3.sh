LANG=en_US
LC_NUMERIC="en_US.UTF-8"

for param in $(seq 0.0 0.01 1.0)
do
    for seq in $(ls ../data/vot2014) 
    do
        python3.7 run_tracker.py --seq $seq --alpha $param
    done
    printf "done %f\n" $param
    echo -en '#\n' >> ../results/results.txt
done
