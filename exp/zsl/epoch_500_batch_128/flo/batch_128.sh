python3 ./wgan.py --manualSeed 806 --cls_weight 0.1 --syn_num 300 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 500 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 128 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --outname flowers --nrh 4096 --nrh1 1024 --nrh2 512 --nrh3 312 --nrh4 156 --drop_rate 0.2
