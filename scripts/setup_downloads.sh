readarray -t games < <(python get_gamelist.py)
if [ -z "$1" ]; then path=../models; else path="$1"; fi
pretrained_chainer=("DQN" "Rainbow" "A3C" "IQN")
for alg in "${pretrained_chainer[@]}"; do
        for game in "${games[@]}"; do
                python download_pretrained.py --alg $alg  --env $game
        done
done
mkdir -p $path/chainer
rsync -a ~/.chainer/dataset/pfnet/chainerrl/models/DQN/* $path/chainer/DQN-C/ --remove-source-files
rsync -a ~/.chainer/dataset/pfnet/chainerrl/models/Rainbow/* $path/chainer/Rainbow/ --remove-source-files
rsync -a ~/.chainer/dataset/pfnet/chainerrl/models/A3C/* $path/chainer/A3C/ --remove-source-files
rsync -a ~/.chainer/dataset/pfnet/chainerrl/models/IQN/* $path/chainer/IQN/ --remove-source-files

