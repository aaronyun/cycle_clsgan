python3 ./dwgan.py --manualSeed 3483 --cls_weight 0.01 --consistency_weight 1 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 70 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --reverse_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --lr 0.0001 --syn_num 300 --classifier_lr 0.001 --outname cub