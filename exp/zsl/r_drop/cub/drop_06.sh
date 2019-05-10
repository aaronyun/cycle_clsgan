python3 ./r_clswgan.py --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 100 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB1 --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --nrh 4096 --nrh1 1024 --nrh2 512 --nrh3 312 --nrh4 156 --drop_rate 0.6