# run: ./scripts/run_all_algs.sh > /dev/null 2>&1 &
n_parties=1000
device="cuda:3"
sample=0.2
niid=1
model="lenet"
#model="resnet"
dataset="cifar10"
num_local_steps=10
beta_=0.9
output_path=benchmarking/cifar10/niid-$niid/clients_$n_parties
root="/home/stijani/projects/phd/paper-2/phd-paper2-code/NIID-Bench/exp_metrics"

# plots
plot_title="Test-Accuracy-vs-Comms-Round"  

# "gradiance" "fedprox" "fedavg" "scaffold" "moon"
# --exp_category benchmarking/cifar10/$partition/clients_$n_parties \

for alg in "gradiance" "fedavg" "fedprox" "fednova" "scaffold"
do
    python experiments.py \
        --exp_category $output_path \
        --device $device \
        --alg $alg \
        --model $model \
        --dataset $dataset \
        --lr 0.01 \
        --beta_ $beta_ \
        --batch-size 10 \
        --num_local_steps $num_local_steps \
        --n_parties $n_parties \
        --mu 0.01 \
        --rho 0.9 \
        --comm_round 1000 \
        --niid $niid \
        --beta 0.9 \
        --logdir "./logs/" \
        --noise 0 \
        --sample $sample \
        --init_seed 0 \
        --exp_title $alg
        #--partition $partition \

    python visualization.py \
        --plot_title $plot_title \
        --metric_filename $root/$output_path/test_acc.csv \
        --save_path $root/$output_path/metric_plot.png \
        --save_path_hist $root/$output_path/hist_plot.png
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





