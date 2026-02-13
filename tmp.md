#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# API配置
API_URL="https://gw.1route.ai/v1/chat/completions"
API_KEY="sk-37fbb5a2-989a-4364-ac81-e58281d31e13"

# 待测模型列表
MODELS=(
    "us.anthropic.claude-opus-4-5-20251101-v1:0"
    "openai.gpt-oss-120b-1:0"
    "us.deepseek.r1-v1:0"
    "deepseek-reasoner"
    "kimi-k2-thinking"
    "MiniMax-M2.1"
    "doubao-seed-1-8-251228"
    "qwen3-30b-a3b-thinking-2507"
    "qwen3-235b-a22b"
    "openai/gpt-oss-120b"
    "qwen/qwen3-32b"
    "llama-3.3-70b-versatile"
    "open-mistral-nemo-2407"
    "gemini-2.5-flash"
    "gemini-3-pro-image-preview"
)

# 结果数组
NON_STREAM_WITH_REASONING=()
NON_STREAM_WITHOUT_REASONING=()
STREAM_WITH_REASONING=()
STREAM_WITHOUT_REASONING=()

echo "=========================================="
echo "开始测试推理字段格式支持情况"
echo "=========================================="
echo ""

# 测试非流式输出
test_non_stream() {
    local model=$1
    echo -n "测试非流式 [$model]... "

    response=$(curl -s "$API_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{
            \"model\": \"$model\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Hi, how are you?\"}],
            \"reasoning_effort\": \"low\"
        }")

    # 检查是否包含 reasoning_content 字段
    if echo "$response" | grep -q "reasoning_content"; then
        echo -e "${GREEN}✓ 包含 reasoning_content${NC}"
        NON_STREAM_WITH_REASONING+=("$model")
        return 0
    else
        echo -e "${RED}✗ 不包含 reasoning_content${NC}"
        NON_STREAM_WITHOUT_REASONING+=("$model")
        return 1
    fi
}

# 测试流式输出
test_stream() {
    local model=$1
    echo -n "测试流式   [$model]... "

    response=$(curl -s "$API_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d "{
            \"model\": \"$model\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Hi, how are you?\"}],
            \"stream\": true,
            \"reasoning_effort\": \"low\"
        }")

    # 检查是否包含 delta.reasoning 字段
    if echo "$response" | grep -q "\"reasoning\""; then
        echo -e "${GREEN}✓ 包含 delta.reasoning${NC}"
        STREAM_WITH_REASONING+=("$model")
        return 0
    else
        echo -e "${RED}✗ 不包含 delta.reasoning${NC}"
        STREAM_WITHOUT_REASONING+=("$model")
        return 1
    fi
}

# 遍历所有模型进行测试
for model in "${MODELS[@]}"; do
    echo "----------------------------------------"
    test_non_stream "$model"
    test_stream "$model"
    echo ""
done

# 输出汇总结果
echo "=========================================="
echo "测试结果汇总"
echo "=========================================="
echo ""

echo -e "${YELLOW}1. 非流式输出中包含 choices.message.reasoning_content 字段的模型:${NC}"
if [ ${#NON_STREAM_WITH_REASONING[@]} -eq 0 ]; then
    echo "   (无)"
else
    for model in "${NON_STREAM_WITH_REASONING[@]}"; do
        echo "   - $model"
    done
fi
echo ""

echo -e "${YELLOW}2. 非流式输出中不包含 choices.message.reasoning_content 字段的模型:${NC}"
if [ ${#NON_STREAM_WITHOUT_REASONING[@]} -eq 0 ]; then
    echo "   (无)"
else
    for model in "${NON_STREAM_WITHOUT_REASONING[@]}"; do
        echo "   - $model"
    done
fi
echo ""

echo -e "${YELLOW}3. 流式输出中包含 choices.delta.reasoning 字段的模型:${NC}"
if [ ${#STREAM_WITH_REASONING[@]} -eq 0 ]; then
    echo "   (无)"
else
    for model in "${STREAM_WITH_REASONING[@]}"; do
        echo "   - $model"
    done
fi
echo ""

echo -e "${YELLOW}4. 流式输出中不包含 choices.delta.reasoning 字段的模型:${NC}"
if [ ${#STREAM_WITHOUT_REASONING[@]} -eq 0 ]; then
    echo "   (无)"
else
    for model in "${STREAM_WITHOUT_REASONING[@]}"; do
        echo "   - $model"
    done
fi
echo ""

echo "=========================================="
echo "测试完成"
echo "=========================================="
