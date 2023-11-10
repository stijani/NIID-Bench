# run: ./scripts/run_all_algs.sh > /dev/null 2>&1 &
n_parties=100
device="cuda:5"
sample=0.1
partition="noniid-#label1"
model="lenet"
dataset="cifar10"
comm_round=500
num_local_steps=50
# "gradiance" "fedprox" "fedavg" "scaffold" "moon"

for alg in "gradiance"
do
    python experiments.py \
        --exp_category benchmarking/cifar10/$partition/clients_100_round_500_localsteps_50_manual_update \
        --device $device \
        --alg $alg \
        --model $model \
        --dataset $dataset \
        --lr 0.01 \
        --beta_ 0.9 \
        --batch-size 64 \
        --num_local_steps $num_local_steps \
        --n_parties $n_parties \
        --mu 0.01 \
        --rho 0.9 \
        --comm_round $comm_round \
        --partition $partition \
        --beta 0.5 \
        --logdir "./logs/" \
        --noise 0 \
        --sample $sample \
        --init_seed 0 \
        --exp_title $alg
done








#######################

# algorithms=("fedavg" "fedprox" "gradiance")
# devices=("cuda:2" "cuda:3" "cuda:4")



# length=${#algorithms[@]}
# for ((i = 0; i < length; i++)); do
#     alg="${algorithms[i]}"
#     device="${devices[i]}"

#     python experiments.py \
#         --exp_category benchmarking/cifar10/homo/clients_10 \
#         --device "$device" \
#         --alg "$alg" \
#         --model simple-cnn \
#         --dataset cifar10 \
#         --lr 0.01 \
#         --beta_ 0.9 \
#         --batch-size 64 \
#         --num_local_steps 150 \
#         --n_parties 10 \
#         --mu 0.01 \
#         --rho 0.9 \
#         --comm_round 200 \
#         --partition homo \
#         --beta 0.5 \
#         --logdir './logs/' \
#         --noise 0 \
#         --sample 1 \
#         --init_seed 0 \
#         --exp_title "$alg"
# done


# algorithms=("scaffold" "moon")
# devices=("cuda:5" "cuda:6")

# length=${#algorithms[@]}
# for ((i = 0; i < length; i++)); do
#     alg="${algorithms[i]}"
#     device="${devices[i]}"

#     python experiments.py \
#         --exp_category benchmarking/cifar10/homo \
#         --device "$device" \
#         --alg "$alg" \
#         --model simple-cnn \
#         --dataset cifar10 \
#         --lr 0.01 \
#         --beta_ 0.9 \
#         --batch-size 64 \
#         --num_local_steps 150 \
#         --n_parties 10 \
#         --mu 0.01 \
#         --rho 0.9 \
#         --comm_round 200 \
#         --partition homo \
#         --beta 0.5 \
#         --logdir './logs/' \
#         --noise 0 \
#         --sample 1 \
#         --init_seed 0 \
#         --exp_title "$alg"
# done





