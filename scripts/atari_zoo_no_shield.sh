readarray -t envs < <(python get_gamelist.py --repeat)
readarray -t prop < <(python get_gamelist.py --prop)
methods=("APEX" "DQN-D" "Rainbow-D" "IMPALA-U" "A2C") 
max_frames=108000
output="../results/violations.csv"
noops=30
len=${#prop[@]}
for ((i=0; i<$len;i++)); do
        for method in "${methods[@]}"; do
                python check_noops.py --method $method --env ${envs[i]} --prop ${prop[i]} --max_frames $max_frames --max_noops $noops --output $output --verbose
        done
done

