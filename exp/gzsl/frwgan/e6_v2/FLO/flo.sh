python3 ./model/frwgan/frwgan_base.py --gzsl --cls_weight 0.1 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 512 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 --classifier_lr 0.001 --syn_num 800 --nclass_all 102 --outname flowers --r_hl 4 --drop_rate 0.2 --triplet_num 1500