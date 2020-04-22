LANG=en_US
LC_NUMERIC="en_US.UTF-8"

for param in $(seq 5 5 200)
do
    for seq in $(ls ../data/vot2014) 
    do
        python3.7 run_tracker.py --seq $seq --sigma $param
    done
    printf "done %f\n" $param
    echo -en '#\n' >> ../results/results.txt
done
