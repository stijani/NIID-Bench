current_directory=$(pwd)
n_parties=1000
device="cuda:5"
sample=0.1
niid=1
batch_size=32 #10 # for 10 and 100 clients
#batch_size=10 # only for 1000 clients runs
model="lenet"
dataset="cifar10"
num_local_steps=10
beta_=0.9
output_path=benchmarking/$dataset/niid-$niid/clients_$n_parties
root=$current_directory/exp_metrics
local_data_path=$HOME/projects/dataset
plot_title="Test-Accuracy-vs-Comms-Round"


# for alg in "fedavg" "scaffold"
for alg in "gradiance" "fedavg" "fedprox" "fednova" "scaffold"
#for alg in "scaffold"
do
    python get_dataset.py \
        --root_dir $local_data_path \
        --datasetname $dataset

    python experiments.py \
        --exp_category $output_path \
        --device $device \
        --alg $alg \
        --model $model \
        --dataset $dataset \
        --lr 0.01 \
        --beta_ $beta_ \
        --batch-size $batch_size \
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
        --exp_title $alg \
        --local_data_path $local_data_path

    python visualization.py \
        --plot_title $plot_title \
        --metric_filename $root/$output_path/test_acc.csv \
        --save_path $root/$output_path/metric_plot.png \
        --save_path_hist $root/$output_path/hist_plot.png
done