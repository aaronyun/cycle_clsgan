python3 ./wgan.py --manualSeed 4115 --cls_weight 0.01 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 54 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --syn_num 100 --classifier_lr 0.0005 --outname sun