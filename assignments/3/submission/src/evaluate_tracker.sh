LANG=en_US
LC_NUMERIC="en_US.UTF-8"

for seq in $(ls ../data/vot2014) 
do
    echo $seq
    python3.7 run_tracker.py --seq $seq --scale 0.93
done
