# run: ./scripts/run_all_algs.sh > /dev/null 2>&1 &
n_parties=1000
device="cuda:3"
sample=0.1
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

# for alg in "fedprox" "fednova" #"gradiance" "fedavg" #"fedprox" "fednova" "scaffold"
#for alg in "gradiance" "fedavg" "fedprox" "scaffold" "fednova"
for alg in "fedavg"
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