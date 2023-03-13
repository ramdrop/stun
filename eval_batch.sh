
# for((i=0;i<60;i=i+5));  
for((i=31;i<35;i=i+1));  
do
    resume=logs/tri_train_stu_0312_174824/ckpt_e_$i.pth.tar
    echo "running ${i}.."
    python main.py --phase='test_stu' --split='val' --resume=${resume}
    python main.py --phase='test_stu' --split='test' --resume=${resume}
    python eval_ece_sh.py --split='val' --epoch=${i} --resume=${resume} --network='res50'
    python eval_ece_sh.py --split='test' --epoch=${i} --resume=${resume} --network='res50'
done





