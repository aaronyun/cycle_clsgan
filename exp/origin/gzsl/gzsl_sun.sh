python3 ./clswgan.py --gzsl --manualSeed 4115 --cls_weight 0.01 --preprocessing --val_every 1 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 40 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN1 --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.001 --syn_num 400 --nclass_all 717 --outname sun 