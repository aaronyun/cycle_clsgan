CUDA_VISIBLE_DEVICES=7 python3 ../R_clswgan.py --manualSeed 9182 --cls_weight 0.01 --r_weight 1 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 100 --syn_num 300 --ngh 4096 --nrh 4096 --nrh1 1024 --nrh2 512 --ndh 4096 --drop_rate 0.2 --lambda1 10 --critic_iter 5 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa