7.16
ack迁移feature_dumper
7.14
ack功能现状梳理和迁移feature dumper方案
7.10
https://code.byted.org/data/mono_sort/merge_requests/36350
7.9
补列式注册字段+列式代码固化
7.8
Feature dumper
步骤
做什么
1. 解析配置
读 JSON 配置：在哪些阶段采样、怎么打标、发到哪
2. 建 GroupData
从 ctx 里，把各个阶段的物料列表(gid+score)整理出来
3. 采样 + 打标
按配置在每个阶段挑 gid，给它们打 label
4. 发送
把 gid/uid/label 发给远端 feature dumper 服务

【lite 微头条/文字流】sort&predict 合并
https://data.bytedance.net/libra/flight/5401938/edit
7.7
没有关联的meego链接
[图片]
7.6
对照头条LR规范，完善LR实验分析部分
7.2-3
学习feature dumperFeature Dump 架构与模块分层
准备列式LR精排数据协议改造
7.1
修复昨天的问题
6.30
列式 build_group_rsp_with_col 只回传 utilities 对应的 cu_ 字段，而行式 build_group_rsp_with_row 回传
  model_pred 中所有 cu_ 开头字段
6.29
热点功能渗透 又负向显著
[图片]
6.25
_CTR direct only
https://code.byted.org/data/mono_predict/merge_requests/3030/

6.24
【lite small】sort&predict 合并
https://data.bytedance.net/libra/flight/5336864/edit
延时avg正收益，但p99负收益，放量继续观察
_CTR
本地复现_CTR字段落分同时出现logit和direct类型的情况
5.6-6.23
vt3 core问题排查
精排列式数据改造实验导致了vt3模块的core问题
[图片]
core的位置始终在eval_target_rule附近，该函数的作用是取出group_tag_set的tag执行rule的表达式运算，得到本次请求中实际命中的target分支。
e.g.
"targets": {
      "topic": {
          "rule": "E(group_tag_set, '19880')",
          "formula": "xs_19880_formula"
      }
 }
仅当tag等于19880时执行表达式xs_19880_formula
core原因
1. vt默认会删除24小时没有被使用且不在白名单内的runtime（表达式）。但是target_rule_runtime这种程序启动时加载，后续请求消费的runtime没有加进白名单。
引发core的场景：当读一个被删除的runtime时，会引发core。
2. eval_target_rule函数所需输入group_tag_set的生命周期本应该是请求级别，每一次调用 new 一个局部的临时 Context ，让这次请求关于group_tag_set的写和读都发生在这块只属于自己的内存上。但是group_tag_set被设置成了vt对象内。vt对象是不同请求之间共享的，在设计上应该是一个只读的编译产物，而非可变的求值状态。这会导致并发请求之间的数据竞争。
引发core的场景：请求中附带了新的targets表达式时，可能触发 group_tag_set 扩容，此时如果并发读就会导致core。
由于以上两个原因的特殊性，core很难在本地稳定复现。需要大流量长时间才能复现。
排查记录
vt3 core 排查记录
列式链路用户指标下降排查
精排列式数据改造实验没有core之后，在用户指标上出现了下降。排查的思路是对比列式+vt3链路与基线的diff
rsp字段缺少排查
通过对比predict的rsp.groups[i].pred发现列式链路少了许多字段
排查思路
1. 确定该字段的生产正常
  1. 源自sort，sort需要成功发给predict
  2. 源自predict的不直接由marine返回的字段，列式链路需要成功注册并调用
  3. 源自predict的直接由marine返回的字段，这类字段一般不会出错，次优先级考虑
2. 确定该字段的消费正常，成功写入了rsp，否则使用dump_scores.xxx.to_rsp来补充
原因1:
列式有一个优化是fakemodel和static的字段是不进rsp的
[图片]
实验发现这类字段缺失会导致“热点内容vv占比”和“热点功能渗透”指标下降，故不使用该改动
原因2:
冷启动相关的部分字段没有在predict中注册，导致冷启动相关指标下降，补充_fill_cold_boost_score后恢复正常。
[图片]
原因3:
行式和列式存储字段的数据结构完全不同
1. 行式
Instance::model_pred CustomMap<std::string, float> 模型原始预估分的第一落点
object_utility_score unordered_map<string, vector<double>> 融合打分的输入表
2. 列式
CommonRequestContext::model_pred_table shared_ptr<monarch::data::FeatureTable> 模型原始预估分的第一落点
CommonRequestContext::help_score_table 融合打分的输入表
append_model_white_list
这个参数的作用是控制白名单字段强制写入rsp，逻辑是从model_pred_table读取字段值并写入g.pred。但是列式的白名单字段，由于vt算分需要，直接存在了help_score_table中，出于拷贝开销的考虑，model_pred_table中并无副本，故会丢字段。
[图片]
解决方式是使用dump_scores.xxx.to_rsp来补充。
universal_objs_cold_boost_dump 字段的原因同上
[图片]
原因4:
两条路径对 cu_xxx字段 的处理不同，导致cu_tf_multi_impr_share_pos字段缺失
- sort消费cu_tf_multi_impr_share_pos的地方
[图片]
- 行式build_group_rsp_with_row ：遍历model_pred ， 凡是 cu_ 开头的全部回传 ，所以 cu_tf_multi_impr_share_pos 进 rsp。（send_model_pred_uniform_cu_formula_vars_back参数默认开启）
[图片]
- 列式build_group_rsp_with_col ：只对 universal_param.utilities 回传 cu_xxx， _pos 这种排序位次派生分根本不在 utilities 里，也就不回传。
[图片]
原因5:
在实验期间，行式逻辑注册了新字段传给predict，列式需要追新

vt算分初始化时取模型分数的diff排查
此处diff行式的逻辑不对，列式是对的。是一个优化，保留。
行式路径的取模型分时的默认值填充逻辑
把值为0也当作logit类型的缺失，补-15。事实上logit0对应概率值0.5，是一个正常值。
[图片]
列式路径的取模型分时的默认值填充逻辑
logit类型的缺失补-15
[图片]
[图片]
Preidct lib化
https://bytediff.bytedance.net/report?task_id=38265964&case_id=11142839
Small lite predict lib化实验因为predict代码中的ERROR日志过多导致review阶段的diff检查不通过，与各个ERROR日志处代码owner沟通修改，找不到人的就自己修改。
4.27
- result_field().name ：这个表达式 产出谁
- get_vars() ：这个表达式 依赖谁
- get_string_consts() ：这个表达式里 写死了哪些字符串常量

valuetree-v3测试方式
单开一个workspace，防止库污染
1. git clone valuetree-v3
2. cd valuetree-v3
3. ./dev.sh test --target :valuetree_loader_test(或者其他目标) --toolchain x86_64-x86_64-clang-1606
4.19
https://www.diffchecker.com/zh-Hans/zSWzJkTI/
4.15
score
https://www.diffchecker.com/zh-Hans/YrLtDeP9/
23example
https://www.diffchecker.com/zh-Hans/M5uCGCXO/
4.9
发送给marine：模型集合 + 样本批 + 每模型的 example_index
目前只裁剪了样本，不裁剪 utility

get_consts_in_expr() ：全局出现过的字符串常量集合，用于模型依赖粗粒度汇总与审计
get_formula_consts(formula) ：某公式的“传递闭包”字符串常量集合，用于从公式反推出模型 key

rule 在 load 时编译一次，存入target_rule_runtimes_，运行时 O(1) 查表并 eval；不需要再解析/编译
4.8
libchapter
- result_field()：这个表达式最终把结果写到哪个字段
- get_vars()：这个表达式依赖了哪些输入变量名
- get_string_consts()：这个表达式里出现了哪些字符串常量
对每个 runtime，它只需要知道：
- 这个 runtime 产出谁：result_field().name
- 这个 runtime 直接依赖谁：get_vars()
- 这个 runtime 自己携带哪些字符串常量：get_string_consts()
编译阶段顺手把“反射所需最小信息”缓存进 Runtime。
后面的 ValueTree introspection 不用再碰 parser/AST，只需要遍历最终生效的 runtimes，就能分析依赖和常量。

Runtime
- 概念：  是“已编译表达式”的可执行对象。把一段表达式字符串（如公式）交给 JIT 编译器，得到一个 Runtime ；之后对它调用 eval(context) ，它就会在给定的 Context 上读取输入变量、计算，并把结果写回到 Context 。
- 反射信息：包含“产出的字段是什么、输入了哪些变量、表达式里有哪些字符串常量”等

  ContextSchema::Field*
  Field 定义在 data/lib9chapter-v3/nc/v3/context_schema.h:83，核心字段：
- index：在 Context.variables_ 中的槽位
- type：类型 ID
- name：字段名
- is_mutable/is_scalar：推导和执行属性
调用链位置 ： ValueTree::do_load_from_json() 里先 expr_manager_.do_load_from_json(doc) （解析/编译），再 executor.build_run_graph(ctx_, expr_manager_) （合并/重排/建图），然后才 expr_manager_.refresh_introspection_metadata() ，见 value_tree.cpp
4.7
mono_predict/apps/toutiao/conf/parameters/toutiao.json -> toutiaopredict运行时读的AB参数配置文件
IF3 的假分支值是显式传入的第三个参数；IF2 假分支固定是 0，IF2_1 假分支固定是 1

  vt2 当前支持“从公式依赖反推所需模型变量，并进一步做 target/rule 级别的模型依赖裁剪”；
  vt3 当前只支持公式执行和输出结果获取，不支持这套依赖反推链路。

数据流
  配置里的公式
    -> FormulasInit 解析公式
    -> 提取公式依赖的 utility 名
    -> required_vars

  required_vars
    -> delete_unused_model_types
    -> 删掉没用的 utility
    -> 删掉没用的 model

  剩下的 models
    -> ModelLoader
    -> ModelConfigBuilder::gen_model_cfg
    -> 生成本次请求真正要跑的模型列表

  如果开启自动推导，且是 vt2:
    公式 target/rule
      -> 判断每个 gid 命中哪个分支
      -> 判断每个分支依赖哪些 utility
      -> 给每个 model 生成 example_index

  model_configs + example_index
    -> 发模型请求
    -> 模型只给指定样本打分
    -> 回包时按 example_index 写回每个样本
4.3
现状
1. common/src/helper/marine_sdk_executor.cpp 中的 convert_pred_io将 Marine predict response 转成 instance 中的 model_pred 字段
2. common/src/helper/utility_score_util.cpp 中的 UtilityScoreUtil::get_utility_score将基于 Instance 的行式数据提取成了列式的 std::unordered_map<std::string, std::vector<double>> object_utility_score
FeatureTable 
一个强类型语义的列式表
在普通 ColumnTable 之上，给每列加了特征语义元信息：feature_id、feature_version、is_shared、is_raw_feature、is_dense。
共三层
1. Schema
记录列名、类型、tag_id、meta。
2. columns_
真正的数据列，类型是 std::shared_ptr<Column>
3. feature_columns_
是 std::vector<FeatureColumn>
  每个 FeatureColumn 本身不拥有数据，只保存：
    - 列名
    - 指向真实列的裸指针 Column* raw_column_
    - FeatureProperty 
4.2
引流报错->已解决
[图片]

4.1
              +---------------------------+
              |            行             |
              +---------------------------+
                            |
                            v
              +---------------------------+
              |           sort            |
              +---------------------------+
                            |
                            v
              +---------------------------+
              |        predict入口        |
              +---------------------------+
                            |
                            v
              +---------------------------+
              |            BFS            |
              +---------------------------+
                            |
                            v
              +---------------------------+
              |      列 ExampleBatch      |
              +---------------------------+
                  /                   \
                 /                     \
                v                       v
+-----------------------+   +-----------------------+
|     (行 Instance)     |   |        Marine         |
+-----------------------+   +-----------------------+
                                        |
                                        v
                            +-----------------------+
                            |          列           |
                            +-----------------------+
                                        |
                                        v
                            +-----------------------+
                            |      行 instance      |
                            +-----------------------+
                                        |
                                        v
                            +-----------------------+
                            | 列 object_utility_score|
                            +-----------------------+
                                        |
                                        v
                            +-----------------------+
                            |          vt           |
                            +-----------------------+
marine的模型分输出直接构造feature table
https://code.byted.org/data/mono_predict/tree/clz_dev_fresh?ref_type=heads
一个预估目标（CTR、CVR、...）对应一个UtilityPredProto

// Per utility prediction values.
message UtilityPredProto {
  //  If target is not specified for the model: $utility_name
  //  If target is specified for the model:
  //  - oracle model: ${utility_name}_${target}
  //  - tf model: ${utility_name}:${target}
  string utility_name = 1;

  // Specify which model provides this utility prediction
  // 指明该预测结果是由哪个模型产生的
  string model_name = 2;

  // Per instance prediction values with the specified model. Should have same
  // number of results as the input instances
  // 核心字段：每个 Instance 对应的模型预估值（可以是标量 float 或者是 Tensor）
  repeated InstancePredProto pred = 3;

  // if use example_index, use this for the index of results
  // 用于索引对齐：指明 pred 中的值分别对应原始请求 batch 中的哪个 instance
  repeated uint32 example_index = 4;

  // 0 by default, meaning successful requests.
  // If any exception occurred, return the corresponding error code.
  // 错误码，0表示成功
  int32 ret_code = 5;

  // extra model pred infos (包含 PS 分片失败数量等附加信息)
  ExtraPredInfo extra_pred_info = 6;

  ServiceInfo service_info = 7;

  // model meta中runstep的fetch所对应的head_name
  string head_name = 8;
}

// Per utility per instance prediction values.
message InstancePredProto {
   oneof pred {
    // Scalar output
    // 标量输出，也就是你在 `model_pred` 字典中存入的 float 类型分数
    float value = 1;

    // Tensor output
    // 张量输出，通常序列化后存入 `model_embedding`
    tensorflow.TensorProto tensor = 2;
  }
  int64 instance_info = 10;
}
3.31
Predict req 的g侧数据
行式
struct GroupFeature {
    1: i64 id,
    2: list<i16> tags,
    3: GroupInfo group_info,
    ...
    13: optional map<string, string> info,
    14: optional map<string, double> model_weight,
    ...
    30: optional i32 impr_cnt, // 同一用户，同一长视频的展现次数
}

struct Req {
    // 共享的用户侧特征 (U侧)
    1: string user,
    24: optional i32 ut,
    25: optional i64 uid,
    ...
    // 物品侧特征列表 (G侧)
    20: optional list<GroupFeature> group_features=[],
}

Predict 向 BFS 发送的并不是一个巨大的特征矩阵，而是一个 轻量级的“指令与引子”集合 。它告诉 BFS：“这是用户A，这是100个物品的ID及基础属性，我准备跑模型B和C，请帮我把所需的全部特征拉回来，用 Rosetta 处理好，最后打包成 ExampleBatch 返回。”
bool BFSFeatureExtractor::build_bfs_service_req(std::shared_ptr<lagrange::bfs::BfsReq> bfs_req_ptr) {
    lagrange::bfs::BfsReq &bfs_req = *bfs_req_ptr;
    
    // 1. 基础信息
    bfs_req.user.ut = ctx.uid_type;
    bfs_req.user.uid = ctx.uid_int;
    bfs_req.did = req->did;
    bfs_req.channel_id = ctx.chn_id;
    bfs_req.app_id = ...;
    bfs_req.bfs_id = get_bfs_id(); // 路由ID

    // 2. 控制选项
    bfs_req.extract_option.use_rosetta = true;
    bfs_req.extract_option.use_example_batch = true;

    // 3. 候选集与透传特征
    build_items_list(bfs_req.items); // 填充 gids
    
    // 填充从 sort 带来的 group_features，通过拷贝实现
    build_group_features(bfs_req.items_features); 
   

    // 4. 按需抽取控制 
    if (abtest_params.enable_subextract) {
        auto& subextract_opt = bfs_req.extract_option.subset_extract_option;
        subextract_opt.enable_subset_feature_extract = true;
        // 告诉 BFS 当前要跑哪些模型
        //BFS 会动态解析这些模型依赖的特征槽位（Slots）， 
        //只抽取这些模型需要的特征 ，而不是全量抽取。
        for (const auto& mgc : ctx.model_load_ptr_->model_group_configs) {
            for (auto& model : mgc.model_configs) {
                subextract_opt.online_model_names.emplace_back(model.name);
            }
        }
    }
    return true;
}

目前mono_predict中FeatureTable相关的逻辑
在打分的准备阶段，已经将基于 Instance 的行式数据提取成了列式的 std::unordered_map<std::string, std::vector<double>> object_utility_score 。
紧接着，在 value_tree_ranker.cpp 中，如果开启了 enable_column_data ，就会调用 convert_to_table 方法将其转换为 FeatureTable 。
void convert_to_table(std::unordered_map<std::string, std::vector<double>>& object_utility_score, FeatureTable& help_score_table) {
  for (const auto& obj_util : object_utility_score) {
    // 1. 将 std::vector<double> 包装为 Feature1D<double>
    auto col = ::std::make_shared<Feature1D<double>>();
    col->push_n(obj_util.second); 
    
    // 2. 将列添加到 FeatureTable 中，obj_util.first 是特征名
    help_score_table.add_featurend(obj_util.first, col);
  }
  
  // (后续的遍历代码主要是为了触发 schema 的构建和校验)
  for (size_t i = 0; i < help_score_table.num_columns(); i++) {
    std::shared_ptr<const Feature1D<double>> col = help_score_table.get_featurend<Feature1D<double>>(i);
    // ...
  }
}

object_utility_score
std::unordered_map<std::string, std::vector<double>>& 
为valuetree打分而生，拼装出来输入到value tree，算完之后结果给instance
存放的东西
1. 原始的模型打分结果（Model Preds）
这是最主要的一部分。它是底层 Marine 模型预估出来的原始分数（通常是从 ins->model_pred 这个 Map 中提取出来的）。
- 例子 ： "click_score" (点击率)、 "finish_rate" (完播率)、 "staytime" (预期停留时长) 等。 2. 经过转换（Transform/Activate）后的模型分
有些模型分在直接参与乘法公式前，需要做平滑、截断或者非线性变换。
- 例子 ：经过 Sigmoid 激活的得分、或者限制在 [min, max] 区间内的得分。 3. 统计类/属性类的静态特征（Static Objs / Tags）
在很多业务场景（如小说、视频）中，公式不仅依赖模型分，还需要根据物料的某些客观属性来进行加权或降权（Boost/Deboost）。这些属性也会作为“一列”被塞进这个结构中。
- 例子 ：
  - "video_duration" ：视频的时长。
  - "book_score" ：小说的书籍质量分（客观属性）。
  - "author_fans_count" ：作者粉丝数。
  - "item_type" ：物料类型（图文、视频、小说等）。 4. 上下文或状态特征
- 例子 ： "is_following" （用户当前是否关注了该作者）、请求时间偏移量、请求的网络状态等。
object_utility_score为什么是列式？
- 为了喂给ValueTree当公式要求计算 click_score * 2.0 时，引擎不是一个一个物料去算，而是直接把click_score= [0.12, 0.05, 0.88] 这一整块连续内存加载到 CPU 的 SIMD 寄存器中，一条指令同时算出 [0.24, 0.10, 1.76] 。 std::vector<double> 提供了这种内存连续性。
- 便于做全局排序/分布特征 ：在这个 Map 里，可以很容易地获取所有物料的某个分数分布。object_map_multiobj_score_to_rank_index ，直接拿到 score 这一列数组，进行全局降序排列，然后把每个物料在这个分数上的排名rank计算出来，作为新的一列 xxx_pos 塞回到 object_utility_score 中，供后续公式使用。如给点击率排名前10%的物料加权。
生命周期
创建：进入到 ValueTreeRanker 这个计算节点时
装载：ValueTreeRanker::prepare_input()
ValueTree V3开启时，convert_to_table(object_utility_score, help_score_table);
计算完毕后，随着ValueTreeRanker析构被释放回收

数据流动中的行列变化
行-> sort->predict入口->BFS->列 ExampleBatch->（行 Instance）->Marine->列->行 instance->列 object_utility_score->vt
由于 ValueTree 计算引擎是列式计算的，所以在准备打分前，需要把分散在各个 Instance 里的特征重新抠出来，拼装成连续的数组。在这个过程中，还会顺便做一些分数的截断（min/max）或非线性变换（Sigmoid 等）。
3.30
 value tree3底层依赖lib9chapter3
valuetree-jit + lib9chapter改进方案

sort数据流向
行（bfs中取到）->行（predict req）->列（valuetree计算） 

ExampleBatch
package idl.euclid.common;

// 特征列表类型，用于区分该特征是共享的（如用户侧特征）还是独立的（如物品侧特征）
enum FeatureListType {
    INDIVIDUAL = 0; // 独立特征：每个样本对应一个值，数组长度 = batch_size
    SHARED = 1;     // 共享特征：整个 batch 共享一个值，数组长度 = 1
}

// 命名的特征列表（即某一列特征，比如 "gender" 或 "author_id"）
message NamedFeatureList {
    optional string name = 1;
    optional uint64 id = 2;
    optional FeatureListType type = 3 [default = INDIVIDUAL];
    // 特征值数组。如果 type 是 SHARED，这里通常只有 1 个元素；如果是 INDIVIDUAL，则有 batch_size 个元素
    repeated Feature feature = 4;
}

// 完整的特征批次结构
message ExampleBatch {
    // 批次大小，表示这个 Batch 里包含了多少个样本（例如 100 篇文章）
    optional int32 batch_size = 1;
    
    // 列式存储的特征列表。包含了解析后的模型可用特征
    repeated NamedFeatureList named_feature_list = 2;
    
    // （可选）原始特征列表，通常包含未经 Rosetta 处理的字符串等
    repeated NamedRawFeatureList named_raw_feature_list = 3;
}

保留 ExampleBatch 完整的列式存储形态，让本次请求的所有 Instance共享一个FeatureCollection
int i = 0;
auto fc_ptr = ctx.fc_ptr.get();
for (const auto &g_feature : req->group_features) {
    auto instance_ptr =  std::make_shared<seraph::Instance>();
    instance_ptr->clear();
    instance_ptr->id = seraph::get_group_feature_id(g_feature);
    
    // 关键点 1：记录当前物品在这个大 Batch 中的索引位置
    instance_ptr->order_in_request = i++; 
    
    // 关键点 2：所有的 Instance 都指向同一个 FeatureCollection
    instance_ptr->feature_collection_ = fc_ptr; 
    
    ctx.instance_ptrs_.emplace_back(instance_ptr);
    ctx.instances_.emplace_back(instance_ptr.get());
}
3.27
研发安全原则考试
学习 https://code.byted.org/data/lib9chapter
一个高效数学表达式编译器，将字符串数学表达式编译成机器码
高效的原因：JIT（Just-In-Time Compilation，即时编译）
JIT原理：以a + b * c 为例
非即时编译
在内存中构建AST
      [+] (AddNode)
     /   \
    /     \
[a]        [*] (MulNode)
(VarNode) /   \
         /     \
       [b]     [c]
  (VarNode)  (VarNode)
节点继承自基类Node，通过虚函数表查找（对象 -> vptr -> vtable -> 函数地址 -> 跳转），难以分支预测
每次计算都要遍历整棵树

即时编译
直接在编译期写死函数指针调用的顺序（先乘后加）
template <typename OP>
class NumericBinaryOp {
public:
    static int32_t func(int64_t ctx_ptr, int64_t ret, int64_t a1, int64_t a2) {
        char* ac1 = (char*)a1;  // 参数1的内存指针
        char* ac2 = (char*)a2;  // 参数2的内存指针
        Context* ctx = (Context*)ctx_ptr;
        
        // 1. 动态类型分发：查表
        // ac1[0] 和 ac2[0] 存的是变量的类型（ 1代表Double, 4代表Long）
        std::pair<ValueType, void(*)(int64_t, int64_t, int64_t)> op;
        if (!OPS.get_op(ac1[0], ac2[0], op)) {
            // 如果找不到对应的类型组合（比如数字加上一个字符串），报错
            return 1;
        }
        
        // 2. 准备返回值类型
        char* c_ret = (char*)ret;
        c_ret[0] = static_cast<char>(op.first);
        
        // 3. 执行真正的计算
        try {
            // 调用上面查表拿到的函数指针
            (*op.second)((int64_t)(c_ret + 1),
                         (int64_t)(ac1 + 1),
                         (int64_t)(ac2 + 1));
        } catch (std::exception& err) {
            // ...
            return 1;
        }
        return 0; // 成功返回0
    }
    
    // 静态成员：类型分发表，预先分配好的二维数组，对于op = ops_[t1][t2] 
    // 仅需一次基础的指针偏移计算： 基址 + t1*宽 + t2，而非使用ifelse/switch进行类型判断
    static BinaryOpSpace OPS;
};
 value tree2底层依赖lib9chapter
3.26
开源合规培训考试
predict代码阅读
1. 向bfs_service请求用户特征和物品特征
2. 送入多个模型打分，得到_STAYTIME_MERGE（播放时长）、tf_comment（评论率）、tf_share（分享率）、tf_finish（完播率）等等等
3. 根据体裁将分数归一化，因为不同体裁同一指标的分数的分布不同，难以比较
4. 根据不同场景使用不同的分数融合公式，进行分数融合，得到最终得分
以mono_predict/apps/toutiao/conf/formula_base/audio_bansui.json的几个公式为例
"ctr_value": "(ctr_bias + ctr_beta * _CTR) ^ ctr_alpha",
"st_duration_debias": "_STAYTIME_MERGE / (MAX(tt_audio_duration + 0.1, 0.1) ^ 0.3)",时长去偏，惩罚长视频在时长上的分数优势
"staytime_value": "(1 + staytime_beta * st_duration_debias) ^ staytime_alpha",利用时长去偏的结果算出最终的时长得分
准备串讲
3.25
昨日编译报错2 解决办法：
去远程cpputil/log仓搜索发现，default分支没有LOGF_ID_INFO_EVERY_N，但是trustlog-1.1.0分支有https://code.byted.org/cpputil/log/blob/trustlog-1.1.0/log.h?ref_type=heads
所以直接在BUILD中指明cpputil/log:trustlog-1.1.0@//cpputil/log:log
实现消重、精排
https://code.byted.org/chenliangzhu/bootcamp
完成研发流程规范考试 
https://wj.bytedance.com/q/v2/316163/q564l395/d97d/#/
predict代码阅读
Value Tree 2 使用文档
3.24
环境配置
咨询trae相关同学，trae的vscode插件不支持gemini、gptcodex等高级模型，遂开始使用trae IDE开发
参考VSCode 改用 clangd 做 C++项目的代码补全，配置trae
bootcamp
测试时报错，错误码80103
[图片]
原因：超时
解决办法：增加超时时间 archon::common::RequestOptions.set_timeout_options(timeout)
实现viking recall
编译报错1
[图片]
解决办法：显式指明编译工具链x86_64-gcc830，默认工具链gcc4.9 不支持 __has_cpp_attribute(x)

编译报错2 viking_utils/log_utils.h使用的宏LOGF_ID_INFO_EVERY_N和ERROR_ID在cpputil/log/log.h中没有
[图片]
3.23
学习blade
实现bootcamp，
实现不包含业务逻辑的server框架，运行server时遇到error
[图片]
原因：LC_XX与当前运行的 glibc 库版本不匹配
[图片]
解决办法：export LC_ALL=C，覆盖掉 cloudIDE的LC_ALL配置，C是系统内置的最基础环境
实现UFSProvider
异步获取user_feature