python3 ./rwgan.py --gzsl --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 500 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA2 --batch_size 512 --nz 85 --attSize 85 --resSize 2048 --lr 0.00001 --syn_num 1800 --nclass_all 50 --outname awa2 --nrh 4096 --nrh1 1024 --nrh2 512 --nrh3 312 --nrh4 156 --drop_rate 0.2