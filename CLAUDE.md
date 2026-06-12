# 项目介绍
科研论文写作
# 目录结构

```text
paper/
├── conference-latex-template_10-17-19/   # 论文模板
├── IEEECS/                               # 基于论文模板的完成正式写作的目录，在此目录下包含论文的所有文件
├── heart_ctrl/                           # 论文相关素材与研究积累
├── ppt/                                 # 汇报材料
└── CLAUDE.md                            # 项目说明与协作约束
```

# 任务
完成论文写作

# 方式
修改IEEECS目录下的文件

# 图片
- Fig. 1 ：CNN-Transformer Actor-Critic 总体框架图。 已有
- Fig. 2 ：Mock circulatory loop 实验平台图。 已有
- Fig. 3 ：Training and evaluation pipeline。
- Fig. 4 ：Comparative control performance under physiological transitions。 已有
- Fig. 5 ：Robustness and safety analysis under disturbances。 已有
- Fig. 6 ：Ablation study of CNN and Transformer modules。 已有

# 编译命令示例
PATH=/Users/bytedance/Library/TinyTeX/bin/universal-darwin:$PATH /Users/bytedance/Library/TinyTeX/bin/universal-darwin/latexmk -g -xelatex -interaction=nonstopmode -file-line-error conference_101719.tex

# 注意
1. python 路径为 /usr/bin/python3
