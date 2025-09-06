#!/bin/bash

# 创建无限循环脚本
while true; do
    echo "开始执行命令: USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh"
    echo "执行时间: $(date)"
    
    # 执行您的命令
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
    
    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "命令执行成功，等待5秒后重新执行..."
    else
        echo "命令执行失败，等待10秒后重新执行..."
        sleep 10
    fi
    
    sleep 5
done
