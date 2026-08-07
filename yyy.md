我已经把两代引擎的编译/加载路径都读过了。先给结论： 你的困惑来自一个隐含假设——「把依赖倒推挪到编译阶段 = 编译阶段变重」。但 vt3 挪进去的那部分其实极其廉价，真正贵的部分反而被它从运行期删掉了。 下面用代码说明。

## 关键区别：依赖倒推在两代里"贵不贵"完全不同
vt2（vt_jit）——依赖倒推在运行期，且很贵

vt2 的依赖倒推是 value_tree.cpp 里的 get_target_consts() （约 L2025）。它是 每次请求 都对整棵 expr_list_ 执行图做 逆序反向可达性遍历 ，逐个表达式实时调 rt->get_vars() / get_string_consts() 累积依赖；在 IF3 分支模式下还要先 do_eval 再按 op 做位运算做分支敏感裁剪。也就是说，倒推是「运行期 × 每 target × 每请求」重复算的。

而 vt2 的编译期（加载期）本身只做一件事：在 value_tree.h 的 add_expr 里对 每条表达式 单独 jit.compile(expression) 。

vt3——依赖倒推在编译期，但只是一次线性前向传播

vt3 在加载时 value_tree.cpp 调 build_introspection_metadata_from_run_graph() 。看它的实现 expr_manager.cpp ：它只是 对已经编译好的 run graph 顺序走一遍 ，用一个 field_consts_by_index 向量把每个节点的 get_string_consts() 沿字段依赖链 做一次并集向上传播 ，把结果塞进 formula_consts_map_ 。

这是 O(节点+边) 的一次前向 set-union ，不重新 JIT、不做逐 target 反向遍历。运行期的 get_target_consts() 因此退化成一次 map 查表。

## 为什么"挪进编译阶段"没让编译变慢，反而整体更快
1. 挪进去的东西本来就便宜。 JIT codegen（asmjit 生成机器码）才是编译阶段的大头，两代都得做。vt3 新增的那趟依赖传播只是读一遍现成 runtime 的元数据，相对 codegen 可以忽略。所以"编译阶段多了倒推 → 编译变慢"这个直觉高估了倒推的成本。
2. vt3 的表达式编译组织方式更省。 vt3 先由 Executor/Session 把 baseline+patch 合并、重排、去重成分层执行图 merged_layered_*_exprs_ （见 expr_manager.h 注释 "ready-to-run expressions after merge baseline and reordered"），再统一走 Session 编译；而 vt2 是在建树过程中零散地逐条 jit.compile 。
3. AB/patch 模型直接继承 baseline 的元数据，不重算。 expr_manager.cpp 的 failover_to_baseline 直接把 formula_consts_map_ 、 target_formulas_ 、 target_rule_runtimes_ 从 base_em_ 拷过来。日常「patch 叠在 baseline 上」的加载几乎不需要再跑倒推。
4. 本质是把「运行期重复算」换成「加载期算一次」。 被移动的不是额外新增的重活，而是把 vt2 那份昂贵的、每请求重复的反向遍历，替换成一次廉价的前向预计算。所以加载/编译阶段几乎没变重，运行期反而卸掉了大头。
消费侧对这套倒推结果的入口是 gen_model.cpp 的 prepare_utility_derivation() （约 L270），vt2/vt3 通过 value_tree_formula.h 统一门面分派——接口一样，但底层 vt3 是查表、vt2 是遍历。

## 一句话
不是"编译阶段变重了还更快"，而是：vt3 把 廉价的一次前向依赖传播 放进编译期（配合合并/分层执行图和 patch 元数据继承），同时 删掉了 vt2 运行期那份昂贵的逐请求反向遍历 ——所以编译阶段几乎没多花时间，整体反而更快。

补充一点澄清：这里的"编译阶段"指的是 VT 模型加载时的 JIT 编译期 ，跟 build.sh 编译 C++ 二进制那个"编译"无关；依赖倒推的挪动不会影响 mono_predict 的 C++ 构建耗时。如果你观察到的其实是 C++ 构建变快，那是另一码事（跟链接的引擎库有关），可以再告诉我，我另行查。