path="/home/stijani/projects/phd/paper-2/phd-paper2-code/NIID-Bench/exp_metrics/benchmarking/cifar10/noniid-#label3/clients_300_round_500_localsteps_50_manual_update"

python visualization.py \
    --plot_title "hyper tunning" \
    --metric_filename $path/test_acc.csv \
    --save_path $path/metric_plot.png \
    --save_path_hist $path/hist_plot.png \
    #--window_size 10