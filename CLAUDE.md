# 项目介绍
科研论文写作
# 目录结构

```text
paper/
├── LaTeX-Template/                       # ACM acmart 官方示例模板 (sample-sigconf.tex)
├── conference-latex-template_10-17-19/   # 旧 IEEEtran 模板（已弃用）
├── IEEECS/                               # 正式写作目录，包含论文的所有文件
├── texmf/                                # 项目本地 TeX 宏包树（acmart 及依赖）
├── texmf-var/                            # 项目本地 TeX 字体/map 缓存
├── heart_ctrl/                           # 论文相关素材与研究积累
├── ppt/                                 # 汇报材料
└── CLAUDE.md                            # 项目说明与协作约束
```

# 任务
完成论文写作

# 方式
修改IEEECS目录下的文件

# 论文格式
- 使用 **ACM acmart sigconf** 模板（`\documentclass[sigconf,nonacm]{acmart}`），与 `LaTeX-Template/sample-sigconf.tex` 保持一致。
- 参考文献使用 BibTeX：`references.bib` + `\bibliographystyle{ACM-Reference-Format}`。
- 表格使用 booktabs（`\toprule/\midrule/\bottomrule`），表标题在表上方。
- acmart 及依赖（含手工从 CTAN 安装的 hyperxmp）已安装到项目本地 `texmf/`，字体 map 缓存在 `texmf-var/`。

# 图片
- Fig. 1 ：CNN-Transformer Actor-Critic 总体框架图。 已有
- Fig. 2 ：Mock circulatory loop 实验平台图。 已有
- Fig. 3 ：Training and evaluation pipeline。
- Fig. 4 ：Comparative control performance under physiological transitions（仅 MAP 面板）。 已有
- Fig. 5 ：Robustness and safety analysis under disturbances（仅 MAP 面板）。 已有
- Fig. 6 ：Generalization sweep（preload-afterload 双热力图）。 已有

# 编译命令示例
acmart 需要 pdflatex + bibtex 流程（不要用 xelatex）。务必设置本地 texmf 环境变量：

```bash
cd /Users/bytedance/paper/IEEECS
export TEXMFHOME=/Users/bytedance/paper/texmf
export TEXMFVAR=/Users/bytedance/paper/texmf-var
export PATH=/Users/bytedance/Library/TinyTeX/bin/universal-darwin:$PATH
pdflatex -interaction=nonstopmode -file-line-error conference_101719.tex
bibtex conference_101719
pdflatex -interaction=nonstopmode -file-line-error conference_101719.tex
pdflatex -interaction=nonstopmode -file-line-error conference_101719.tex
```

# 注意
1. python 路径为 /usr/bin/python3
2. 编译前必须导出 TEXMFHOME=/Users/bytedance/paper/texmf 与 TEXMFVAR=/Users/bytedance/paper/texmf-var，否则找不到 acmart 与字体 map。
