for seq in $(ls ../data/vot2014) 
do
    python3.7 run_tracker.py --tracker ms --seq $seq --n_bins 22 --alpha 0.006 >> n_failures_ms.txt
    if [ $? -eq 0 ]
    then
        python3.7 run_tracker.py --tracker ncc --seq $seq  >> n_failures_ncc.txt
        echo $seq >> seq.txt
    fi
done

