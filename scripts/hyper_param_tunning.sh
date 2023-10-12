for beta_ in 0.1 0.3 0.5 0.7 0.9
do
	for num_local_steps in 10 50 70 100 150 200 500 1000
	do
		python experiments.py \
            --exp_category hyper-tunning/gradiance/cifar10/beta_vs_num_local_steps \
            --device 'cuda:2' \
            --alg gradiance \
            --model simple-cnn \
            --dataset cifar10 \
            --lr 0.01 \
            --beta_ $beta_ \
            --batch-size 64 \
            --num_local_steps $num_local_steps \
            --n_parties 10 \
            --mu 0.01 \
            --rho 0.9 \
            --comm_round 100 \
            --partition homo\
            --beta 0.5 \
            --logdir './logs/' \
            --noise 0 \
            --sample 1 \
            --init_seed 0 \
            --exp_title beta:{$beta_}num_local_steps:{$num_local_steps}
	done
done
