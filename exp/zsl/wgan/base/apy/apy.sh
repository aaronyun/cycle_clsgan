python3 ./wgan.py --manualSeed 1995 --cls_weight 0.01 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset APY --batch_size 64 --nz 64 --attSize 64 --resSize 2048 --lr 0.00001 --syn_num 1000 --classifier_lr 0.001 --outname apy