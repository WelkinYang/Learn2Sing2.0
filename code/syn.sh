out=$1
spk=$2
sty=$3
step=$4
device=$5
use_fast_maximum_likelihood_sampling=$6

python3 inference.py logdir/learn2sing_2 testdata/test_labels $out $spk $sty $step $device $use_fast_maximum_likelihood_sampling
