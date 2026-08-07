Agent

参考了 1 个上下文

让我把两代编译阶段的实际代码读透，才能给你逐行对得上的解释。

命令已执行 cd /cloudide/workspace && echo "===== vt3 expr_manager.h members/methods =====" && grep -n "void \|Runtime\|merged_\|base_em_\|parent_em_\|patch\|merge\|reorder\|dedup\|layer\|do_load\|failover\|build_run_graph\|need_field\|external_runtime" data/valuetree-v3/valuetree/v3/expr_manager.h | head -80

命令已执行 cd /cloudide/workspace && echo "===== vt3 loader.h/.cpp =====" && ls data/valuetree-v3/valuetree/v3/ 2>/dev/null && echo "---loader grep---" && grep -rn "load_from_json\|set_base_em\|base_vt\|baseline\|version2trees\|cache\|LruCache\|class ValueTreeLoader\|patch" data/valuetree-v3/valuetree/v3/loader.h data/valuetree-v3/valuetree/v3/loader.cpp 2>/dev/null | head -40

命令已执行 cd /cloudide/workspace && echo "===== vt3 value_tree_loader.h =====" && grep -n "load_from_json\|set_base_em\|base\|baseline\|cache\|Cache\|class ValueTreeLoader\|patch\|ValueTreePtr\|shared_ptr<ValueTree>\|map<" data/valuetree-v3/valuetree/v3/value_tree_loader.h data/valuetree-v3/valuetree/v3/value_tree_loader.cpp 2>/dev/null | head -50

让我读 loader 的 do_load_from_json (带 base_name 的 patch 版本)和 ValueTree 里 set_base_em 的接线，把 baseline 继承的完整链路读清楚。

data/valuetree-v3/valuetree/v3/value_tree_loader.cpp

现在让我读 vt3 value_tree.cpp 的 set_base_vt / do_load_from_json 和 vt3 set_base_em ，看 baseline 的 session 和已编译 runtime 如何被 patch 复用。

data/valuetree-v3/valuetree/v3/value_tree.cpp

data/valuetree-v3/valuetree/v3/expr_manager.cpp

我已经把两代的编译加载路径读透了。下面把这一点拆开逐层解释——同时先做个诚实的澄清： set_base_em （ expr_manager.cpp ）才是"patch 继承 baseline"的正路；我上一轮引用的 failover_to_baseline 其实是 加载 patch 失败时的兜底 （ value_tree_loader.cpp catch 分支里才调），两者继承的字段一样，但触发时机不同。下面按正确的机制讲。

## 一、先明确"baseline"和"patch"是什么
线上一个 VT 模型 = 一个 baseline 公式 （大部分表达式，很少变）+ 一份 AB patch （少量覆盖/新增，随实验切换）。 load_ab_json(vt, loader, formula_base, formula) 里， formula_base 是 baseline 名， formula 是这份 patch 的 JSON。两代都在 FormulasInit::load_vt （ formulas_init.cpp ）触发。

关键区别就在于： 加载一份 patch 时，baseline 的那堆表达式要不要重新 codegen。

## 二、vt2：baseline + patch 合并成一颗新树，逐条 add_expr
vt2 的 load_ab_json 最终走 do_load_from_json （ value_tree.cpp ），它把 base JSON 和 AB JSON 合并后，对合并结果里的 每一条 表达式调用 add_expr ：

这里的建树是"平铺遍历、一条一条 compile"。baseline 有几百上千条表达式，patch 只改其中几条，但 vt2 的循环会把 整棵合并后的树 都过一遍 jit.compile 。

它唯一的"救济"是 jit 是进程级单例 static nc::Jit jit （ value_tree.h ），内部有 LruCache<hash(expr), RuntimePtr> （ jit.h ）。所以 baseline 那些表达式字符串如果之前编译过，会命中缓存，不会真正重新 codegen。

但请注意 ：命中缓存省的是"codegen"， add_expr 这层的循环、hash 计算、 get_string_consts 塞 consts_in_expr_ 、建 expr_list_ / ns_expr_list_ 这些"建树"工作，vt2 每次加载 每条都还得做一遍 。也就是缓存只挡住了最底层 codegen，上层的"逐条组织"省不掉。

## 三、vt3：patch 以 baseline 为父，直接接管已编译产物，只编增量
vt3 的路径完全不同。加载 patch 时先 set_base_vt → set_base_em(base_vt_->expr_manager_) （ value_tree.cpp ）。看 set_base_em 干了什么（ expr_manager.cpp ）：

注意这里拷的都是 const Runtime* 指针和 map， 不是重新编译 。baseline 的编译产物（机器码 runtime）和 formula_consts_map_ （上一轮讲的依赖倒推结果）被 patch 直接接管。

更关键的是 session 的共享——patch 作为 child，它的 effective_session_ 直接指向 baseline 的 session（ value_tree.cpp set_parent ，以及 loader 里 if (!vt.parent()) begin_session() ， value_tree_loader.cpp ）。Session 内部对表达式有内容缓存（ jit.h ），所以 patch 只对 自己真正新增/覆盖的那几条 表达式 compile ，baseline 的绝大多数表达式一次都不用碰。

然后 do_load_from_json 只对 patch 增量做： build_run_graph （重排/分层，产出 merged_layered_*_exprs_ ）→ build_introspection_metadata_from_run_graph （补算被 patch 影响的依赖）→ flush （ value_tree.cpp ）。

## 四、两者差在哪，一句话对照
vt2 vt3 baseline+patch 组织 合并成新树， 逐条 遍历 add_expr patch 以 baseline 为父， set_base_em 指针继承 已编译 runtime baseline 表达式 codegen LruCache 命中可免 codegen，但 每条仍要走建树循环 完全不碰，直接复用父 session 产物 patch 实际编译量 整棵树的组织 只有增量那几条 依赖倒推 formula_consts_map_ 无此结构，运行期每请求重算 直接从 base_em_ 拷贝继承 ，patch 只补增量

所以你原来的理解要点是对的—— vt2 逐条建树、vt3 继承 baseline 只编增量 。补一个精度修正： failover_to_baseline 是 patch 加载 失败时的降级兜底 （ value_tree_loader.cpp ），正常路径的"baseline 不重编"靠的是 set_base_em 的指针继承 + 父 session 共享，二者继承的字段集合相同。

## 五、落到你最初的问题
正因为 patch 加载时 vt3 只编译增量、且直接继承 baseline 的编译产物和依赖倒推结果，所以 即便把依赖倒推放进编译期 ，vt3 每次加载/切 AB 的编译工作量反而比 vt2"整棵逐条过一遍"更小——这就是"依赖倒推挪到编译阶段却没变慢、整体还更快"的第二层原因（第一层是运行期删掉了每请求的重复倒推）。