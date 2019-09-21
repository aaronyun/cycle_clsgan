python3 ./model/rwgan/rwgan_v1.py --gzsl --manualSeed 3483 --cls_weight 0.01 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 100 --gen_hu 4096 --dis_hu 4096 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --att_size 312 --res_size 2048 --lr 0.0001 --classifier_lr 0.001 --syn_num 800 --nclass_all 200 --outname cub --r_hl 3 --drop_rate 0.2 --netAtt './model/_pretrain_data/rn/cub_att_net.pkl' --netRN './model/_pretrain_data/rn/cub_rn_net.pkl'