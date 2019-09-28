python3 ./model/rwgan/rwgan_v1.1.py --gzsl --manualSeed 1995 --cls_weight 0.01 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 100 --gen_hu 4096 --dis_hu 4096 --lambda1 10 --critic_iter 5 --dataset APY --batch_size 256 --nz 64 --att_size 64 --res_size 2048 --lr 0.00001 --classifier_lr 0.001 --syn_num 1400 --nclass_all 32 --outname apy --r_hl 2 --drop_rate 0.2