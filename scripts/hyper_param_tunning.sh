niid = 1
# for beta_ in 0.0
#for beta_ in 0.5 0.9 0.1 0.7 0.3 0.99
#for num_local_steps in 7 10 20 50 100 150 500
#do
for num_local_steps in 75 7 50 15 20 10 100 150 200
do
      python experiments.py \
      --exp_category hyper-tunning/gradiance/cifar10/niid-$niid/num_local_steps/1000_clients \
      --device 'cuda:1' \
      --alg gradiance \
      --model lenet \
      --dataset cifar10 \
      --lr 0.01 \
      --beta_ 0.9 \
      --batch-size 32 \
      --num_local_steps $num_local_steps \
      --n_parties 1000 \
      --mu 0.01 \
      --rho 0.9 \
      --comm_round 500 \
      --niid 1 \
      --beta 0.5 \
      --logdir './logs/' \
      --noise 0 \
      --sample 0.1 \
      --init_seed 0 \
      --exp_title "num_local_steps_$num_local_steps"
      #--partition $partition \
done
#done
