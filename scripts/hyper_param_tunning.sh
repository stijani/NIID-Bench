partition=noniid-#label3
for beta_ in 0.0
#for beta_ in 0.5 0.9 0.1 0.7 0.3 0.99
# for num_local_steps in 7 10 20 50 100 150 500
do
	for num_local_steps in 7 50 20 10 100 150 500
	do
		python experiments.py \
            --exp_category hyper-tunning/gradiance/cifar10/$partition/beta_num_local_steps/100_clients \
            --device 'cuda:7' \
            --alg gradiance \
            --model lenet \
            --dataset cifar10 \
            --lr 0.01 \
            --beta_ $beta_ \
            --batch-size 64 \
            --num_local_steps $num_local_steps \
            --n_parties 100 \
            --mu 0.01 \
            --rho 0.9 \
            --comm_round 100 \
            --partition $partition \
            --beta 0.5 \
            --logdir './logs/' \
            --noise 0 \
            --sample 0.1 \
            --init_seed 0 \
            --exp_title "beta: $beta_ num_local_steps: $num_local_steps"
	done
done
