partition=noniid-#label2
unique_dir=clients_100_newest
plot_title=Initial-Code-Testing
path="/home/stijani/projects/phd/paper-2/phd-paper2-code/NIID-Bench/exp_metrics/benchmarking/cifar10/$partition/$unique_dir"

python visualization.py \
    --plot_title $plot_title \
    --metric_filename $path/test_acc.csv \
    --save_path $path/metric_plot.png \
    --save_path_hist $path/hist_plot.png