# 论文阅读报告总览

## **1. classic_ctrl/FC-coupled_model（文件路径以heart_ctrl为根路径，且已经注释，无需重复添加）**
- **论文标题:** Cardiac index adaptive physiological control system for continuous-flow left ventricular assist device
- **作者:** Hongtao Liu, Aili Guan, Fayu Xing, Yunpeng Zhang, Wencheng Wang, Guoxu Liu
- **发表日期:** 2025-06-12
- **标签:** VAD, Physiological Control, Cardiac Index, Adaptive Control
- **一句话总结:** 本文提出了一种基于心脏指数（Cardiac Index, CI）的自适应生理控制策略（CIAPC），用于持续流动左心室辅助装置（CFLVAD），旨在根据患者个体特征（身高、体重、心率）实现个性化治疗，并能在不同生理状态（休息、运动、睡眠）下动态调节泵速，以维持血液动力学稳定，同时有效防止心室吸引和泵反流。

## **2. classic_ctrl/FC**
- **论文标题:** Supervised Adaptive Fuzzy Control of LVAD with Pulsatility Ratio Modulation
- **作者:** Milad Azizkhani, Yue Chen
- **发表日期:** 2022-08-20
- **标签:** LVAD, Fuzzy Control, Adaptive Control, Pulsatility Ratio
- **一句话总结:** 本文提出了一种监督自适应模糊控制器，使用脉动比率（Pulsatility Ratio, PR）作为控制指标，旨在提高LVAD控制器的响应速度、解决模糊逻辑调参难题，并有效防止心室抽吸。

## **3. classic_ctrl-ml/FC-CNN**
- **论文标题:** Deep learning methods for the estimation of preload and physiological control of heart assist devices
- **作者:** Fetanat Fard Haghighi, Masoud
- **发表日期:** 2021
- **标签:** Deep learning, physiological control systems, LVADs, machine learning, CNN, sensorless control
- **一句话总结:** 本文旨在设计、开发和评估用于预负荷估计和LVADs无传感器生理控制系统的新方法，结合了深度CNN进行预负荷估计和模型无关自适应控制（MFAC）进行泵速调节。

## **4. classic_ctrl-ml/PPSMC-ffn**
- **论文标题:** An Advanced Physiological Control Algorithm for Left Ventricular Assist Devices
- **作者:** Mohsen Bakouri
- **发表日期:** 2023-10-24
- **标签:** LVAD, 生理控制, 极点配置, 滑模控制, 神经网络
- **一句话总结:** 本文提出了一种结合了极点配置和滑模控制的先进生理控制算法（PP-SMC），并利用神经网络来补偿系统不确定性，旨在为LVAD提供一个能适应不同生理条件的鲁棒控制器。

## **5. ml/CNN**
- **论文标题:** A Sensorless Control System for an Implantable Heart Pump Using a Real-Time Deep Convolutional Neural Network
- **作者:** Masoud Fetanat, Michael Stevens, Christopher Hayward, and Nigel H. Lovell
- **发表日期:** 2021-02-23
- **标签:** CNN, LVAD, FFDL-MFAC, sensorless
- **一句话总结:** 本文提出了一种基于深度卷积神经网络（CNN）的无传感器生理控制系统，通过实时估计前负荷并结合全动态形式的无模型自适应控制（FFDL-MFAC），以在安全的生理范围内维持患者的血液动力学稳定。

## **6. ml/KNN-SVM-ANN**
- **论文标题:** Intelligent reflux and suction detection system for ventricular assist devices: in silico study
- **作者:** Bruno J. Santos, Idagene A. Cestari
- **发表日期:** 2024-12-10
- **标签:** LVAD, 生理控制, 反流, 吸引, 结构化算法, 人工智能
- **一句话总结:** 本文提出了一种结合结构化算法（SA）和人工智能模型（KNN, SVM, ANN）的智能检测系统，用于检测LVAD的反流和吸引事件，以提高设备的安全性。

## **7. mix/tmp**
- **论文标题:** 经皮左心室辅助装置的反流模型和控制研究
- **作者:** 殷安云 (武汉大学)
- **发表日期:** 2023-12
- **标签:** VAD, PLVAD, 反流, 自适应控制, 等效电路模型, 状态估计
- **一句话总结:** 本文系统地研究了经皮左心室辅助装置（PLVAD）在全支持状态下由泵位置偏移引起的反流问题，建立了能够模拟瓣膜动态响应的反流模型，并开发了相应的状态估计算法和预防性控制策略。

## **8 RL/critic-actor-lstm**
- **论文标题:** Enhancing cardiac hemodynamic and pulsatility in heart failure via deep reinforcement learning: An in-silico and in-vitro validation study of percutaneous ventricular assist devices
- **作者:** Yuyang Shi, Zhike Xu, Chenghan Chen, Feng He, Pengfei Hao, Xiwen Zhang
- **发表日期:** 2025-07-18
- **标签:** pVAD, DRL, Heart Failure, Hemodynamics, Pulsatility, Adaptive Control
- **一句话总结:** 本文为经皮心室辅助装置(pVAD)设计了一种基于深度强化学习(DRL)的自适应控制器，该控制器能够恢复生理脉动性血液动力学，并能自主适应不同的心力衰竭状况和心率波动。

### **9 RL/economic_RL**
- **论文标题:** A Physiological Control System for Pulsatile Ventricular Assist Device Using an Energy-Efficient Deep Reinforcement Learning Method
- **作者:** Te Li, Member, IEEE, Wenbo Cui , Xingjian Liu , Member, IEEE, Xu Li , Nan Xie , and Yongqing Wang
- **发表日期:** 2023-05-24
- **标签:** VAD, DRL, Energy-Efficient
- **一句话总结:** 本文提出了一种基于深度强化学习（DRL）的能量效率高的搏动式心室辅助装置（PVAD）生理控制系统，该系统通过使用AdderNet替代全连接网络，在不降低性能的情况下，将传统DRL算法的能耗降低至44.8%。

### **10 RL/PPO**
- **论文标题:** Physiological control for left ventricular assist devices based on deep reinforcement learning
- **作者:** Diego Fernández-Zapico, Thijs Peirelinck, Geert Deconinck, Dirk W. Donker, Libera Fresiello
- **发表日期:** 2024-09-17
- **标签:** VAD, Reinforcement Learning, Control Systems
- **一句话总结:** 本文提出了一种基于深度强化学习（DRL）的左心室辅助装置（LVAD）控制器，旨在通过模拟生理控制，防止心室抽吸并允许主动脉瓣开放。

### **11 RL/soft-critic-actor**
- **论文标题:** Intelligent and strong robust CVS-LVAD control based on soft-actor-critic algorithm
- **作者:** Te Li, Wenbo Cui, Nan Xie, Heng Li, Haibo Liu, Xu Li, Yongqing Wang
- **发表日期:** 2022-04-22
- **标签:** DRL, Heart Failure, LVAD, Physiological Control, Soft-Actor-Critic
- **一句话总结:** 本文提出了一种基于深度强化学习（DRL）的LVAD生理控制器，该控制器使用Soft-Actor-Critic（SAC）算法，通过模拟Frank-Starling机制来快速、鲁棒地响应不同生理需求，其性能优于传统的PID控制器。
