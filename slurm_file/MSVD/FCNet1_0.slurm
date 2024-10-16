#!/bin/bash
#SBATCH -J y2t_FCNet1.0-cap            # 任务名字是
#SBATCH --cpus-per-task=12    # 申请 10 个 cpu 核心，这里防止内存不够必须增加 CPU 核心数
#SBATCH --partition all
#SBATCH --mem 15000
#SBATCH --gres=gpu:1         # 申请 1 块 GPU 卡，必须填写，SLURM 默认是不分配 GPU 的
#SBATCH -t 10-00:00:00         # 最长运行时间是 10 天
#SBATCH -o /public/home/zhangy/v2_Non-Autoregressive-Video-Captioning-master/log_file/y2t_FCNet1_0.out          # 将屏幕的输出结果保存到当前文件夹
               # 作业使用的 QoS 为 normal #nodelist=node10指定节点

module load jdk/1.8.0
source activate cap

echo Start slurm job at `date`


python train.py --default --dataset Youtube2Text --method ARB --is_FCNet True


echo End slurm job at `date`