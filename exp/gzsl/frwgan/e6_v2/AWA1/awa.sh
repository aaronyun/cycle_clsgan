python3 ./model/frwgan/frwgan_v2.py --gzsl --cls_weight 0.01 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 200 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA1 --batch_size 512 --nz 85 --attSize 85 --resSize 2048 --lr 0.00001 --classifier_lr 0.001 --syn_num 3200 --nclass_all 50 --outname awa --r_hl 4 --drop_rate 0.2 --triplet_num 1500