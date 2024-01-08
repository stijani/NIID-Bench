partition=niid-1
unique_dir=testing # beta_num_local_steps/100_clients
plot_title=Initial-Code-Testing
# path="/home/stijani/projects/phd/paper-2/phd-paper2-code/NIID-Bench/exp_metrics/hyper-tunning/gradiance/cifar10/$partition/$unique_dir"
path="/home/stijani/projects/phd/paper-2/phd-paper2-code/NIID-Bench/exp_metrics/benchmarking/cifar10/niid-5/clients_10"
python visualization.py \
    --plot_title $plot_title \
    --metric_filename $path/test_acc.csv \
    --save_path $path/metric_plot.png \
    --save_path_hist $path/hist_plot.png \
    --window_size 30