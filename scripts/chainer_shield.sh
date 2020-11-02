readarray -t envs < <(python get_gamelist.py --repeat)
readarray -t prop < <(python get_gamelist.py --prop)
methods=("DQN-C" "Rainbow" "IQN" "A3C")
max_frames=108000
output="../results/shielded_rewards.csv"
noops=30
len=${#prop[@]} 
max_shield=5
for ((i=0; i<$len;i++)); do
	for method in "${methods[@]}"; do
		for ((j=5; j<$shield+1;j++)); do
			python check_noops.py --method $method --env ${envs[i]} --prop ${prop[i]} --max_frames $max_frames --max_noops $noops --output $output --lookahead $j
		done
	done
done
