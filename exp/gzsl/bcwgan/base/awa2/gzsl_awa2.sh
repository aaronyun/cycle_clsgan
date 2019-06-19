python3 ./wgan.py --gzsl --manualSeed 9182 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 100 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA2 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --lr 0.00001 --classifier_lr 0.001 --syn_num 1800 --nclass_all 50 --outname awa2 --bc True