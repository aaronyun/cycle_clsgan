python3 ./rwgan.py --gzsl --manualSeed 3483 --cls_weight 0.01 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 512 --nz 312 --attSize 312 --resSize 2048 --lr 0.0001 --classifier_lr 0.001 --syn_num 900 --nclass_all 200 --outname cub --r_hl 3 --drop_rate 0.2