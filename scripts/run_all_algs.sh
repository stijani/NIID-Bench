# run: ./scripts/run_all_algs.sh > /dev/null 2>&1 &
# #testing_end_2_end
device='cuda:3'
alg=gradiance
n_parties=10
partition=homo

python experiments.py \
    --exp_category benchmarking/cifar10/$partition/client_$n_parties/replicate_results_from_branch_added_gradiance_algorithm \
    --device $device \
    --alg $alg \
    --model simple-cnn \
    --dataset cifar10 \
    --lr 0.01 \
    --beta_ 0.5 \
    --batch-size 64 \
    --num_local_steps 1000 \
    --n_parties $n_parties \
    --mu 0.01 \
    --rho 0.9 \
    --comm_round 100 \
    --partition $partition \
    --beta 0.5 \
    --logdir './logs/' \
    --noise 0 \
    --sample 1 \
    --init_seed 0 \
    --exp_title $alg