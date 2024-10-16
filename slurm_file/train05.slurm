#!/bin/bash
#SBATCH -J NAR-cap_4            # 任务名字是 gpu-job
#SBATCH --cpus-per-task=10    # 申请 10 个 cpu 核心，这里防止内存不够必须增加 CPU 核心数
#SBATCH --partition all
#SBATCH --mem 15000
#SBATCH --gres=gpu:1         # 申请 1 块 GPU 卡，必须填写，SLURM 默认是不分配 GPU 的
#SBATCH -t 10-00:00:00         # 最长运行时间是 10 天
#SBATCH -o /public/home/zhangy/v2_Non-Autoregressive-Video-Captioning-master/log_file/submit3_2.out          # 加入迭代的trick,去除lstm中的dropout=0。5
               # 作业使用的 QoS 为 normal #nodelist=node10指定节点

module load jdk/1.8.0
source activate cap

echo Start slurm job at `date`

#python 1.py
#python train.py --default --dataset MSRVTT --method ARB

python train.py --default --dataset MSRVTT --method NAB --pos_attention --group_dependency True --version v2

#python 1.py
#python translate.py --default --dataset MSRVTT --method NACF --use_ct --paradigm mp
#python translate.py --default --dataset MSRVTT --method NACF --use_ct --paradigm ef
#python translate.py --default --dataset MSRVTT --method NACF --use_ct --paradigm l2r

echo End slurm job at `date`