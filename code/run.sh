mode=$1
basedir='.'

python3 -u $basedir/train.py -c $basedir/config.json -l logdir/ -m learn2sing_2
