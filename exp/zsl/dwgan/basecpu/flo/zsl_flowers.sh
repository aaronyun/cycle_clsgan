python3 ./dwgan.py --manualSeed 806 --cls_weight 0.1 --val_every 1 --preprocessing --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 100 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --syn_num 300 --classifier_lr 0.001 --outname flowers