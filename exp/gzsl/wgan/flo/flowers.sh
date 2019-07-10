python3 ./wgan.py --gzsl --manualSeed 806 --cls_weight 0.1 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 80 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --classifier_lr 0.001 --syn_num 1200 --nclass_all 102 --outname flowers