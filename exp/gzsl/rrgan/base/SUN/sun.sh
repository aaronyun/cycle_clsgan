python3 ./model/rrgan/rrgan_base.py --gzsl --manualSeed 4115 --cls_weight 0.01 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 2048 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.001 --syn_num 600 --nclass_all 717 --outname sun --r_hl 4 --drop_rate 0.2 --ntrain_class 645