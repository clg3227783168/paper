使用压力-体积（PV）环模型来描述不同临床综合征的心室功能和心肌氧供需情况。PV环提供了关于收缩功能、舒张特性、每搏输出量、心脏做功和心肌氧消耗的信息。

使用计算机仿真in-silico(Simulink)和模拟循环回路in-vitro

实验场景:静息 运动'高风险PCI、急性心肌梗死、心源性休克和急性失代偿性心力衰竭 改变体循环阻力（Rs）、心率（HR）、最大心肌弹性（Emax）和目标压力（Ptarget）等参数实现这些场景

模拟不同体型患者（瘦、正常、肥胖）

思考到底设置哪些传感器 如果设置的少可以说系统依赖于主动脉压力信号，而长期、稳定、无创地获取该信号在临床上仍是一个挑战。设置多可以设计更加复杂的奖励函数

基于临床数据集记录的LVAD流量波形，采用先进的机器和深度学习方法设计了改进的非侵入性前负荷估计器-----这也就是说根据临床流量波形数据集设计前负荷估计器 实际上人体里面是没有这个传感器的（离线学习）->要是有一个高度仿真的模拟环境可以用来收集数据 迭代数据就好了

类似heart_ctrl/RL/economic_RL 如何修改算法做到economic 降低功耗

Samsky MD, Milano CA, Pamboukian S, Slaughter MS, Birks E, Boyce S, Najjar SS, Itoh A, Reid B, Mokadam N, Aaronson KD, Pagani FD, Rogers JG. The Impact of Adverse Events on Functional Capacity and Quality of Life After HeartWare Ventricular Assist Device Implantation. ASAIO J. 2021 Oct 1;67(10):1159-1162. doi: 10.1097/MAT.0000000000001378. PMID: 33927085; PMCID: PMC8478694. https://pubmed.ncbi.nlm.nih.gov/33927085/     引用这个说明尽管CFLVADs（连续流左心室辅助装置）在技术上取得了显著进步，但患者的生活质量和不良事件仍然是CFLVAD治疗的重要限制因素(见heart_ctrl/complications/12ae.md)

对某种负面事件进行建模 比如对反流进行集总参数模型建模

传统模型通常假设心脏和血管壁是刚体，而实际上它们具有一定的弹性，这可以影响血流动力学->这里希望能够找到引用
这种简化模型通常无法模拟心脏瓣膜的动态响应，如打开和关闭速度的变化。实际上，心脏瓣膜的开合速度可以随心脏周期的不同阶段而变化，但传统模型无法反应这种特性。

整体流量仅由 PLVAD 提供，这称为全支持状态

卸载左心室可以减轻心脏的负担，有助于受损的心肌得到休息和恢
复[99, 101]。然而，随着时间的推移， PLVAD 患者的心脏可能会显示出改善和恢复的迹
象。这意味着心脏可能开始自行承担更多的泵血工作，不再完全依赖于 PLVAD。在这
种情况下，逐渐过渡到部分支持可能更有利，以促使心脏重新获得自身的功能 -->也就是说考虑VAD的长期使用时需要考虑这一点