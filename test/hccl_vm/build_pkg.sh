#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -e

function usage() {
  echo "Usage:"
  echo "  sh build_pkg.sh [-h | --help]"
  echo "                  [--install <hccl|hcomm>]"
  echo "                  [--tool_path <PATH>]"
  echo ""
  echo "Options:"
  echo "    -h, --help     Print usage"
  echo "    --install <hccl|hcomm>"
  echo "                   Specify which component to build/install, default: both"
  echo "    --tool_path <PATH>"
  echo "                   Set HCCL_VM_PATH, default: current directory"
  echo ""
}

# ==========================================
# 第一步：解析脚本参数和环境变量
# ==========================================

# 1.默认值
BUILD_HCCL=true
BUILD_HCOMM=true
TOOL_PATH=""

# 2.解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        --install)
            if [ "$2" == "hccl" ]; then
                BUILD_HCCL=true
                BUILD_HCOMM=false
            elif [ "$2" == "hcomm" ]; then
                BUILD_HCCL=false
                BUILD_HCOMM=true
            else
                echo "错误: 无效的参数值，支持的参数值为 hccl 或 hcomm"
                exit 1
            fi
            shift 2
            ;;
        --tool_path)
            if [ -n "$2" ]; then
                TOOL_PATH="$2"
                shift 2
            else
                echo "错误: --tool_path 参数需要指定路径值"
                exit 1
            fi
            ;;
        *)
            echo "错误: 未知参数 $1"
            usage
            exit 1
            ;;
    esac
done

# 3.设置 HCCL_VM_PATH
if [ -n "$HCCL_VM_PATH" ]; then
    echo "使用环境变量中的 HCCL_VM_PATH: $HCCL_VM_PATH"
elif [ -n "$TOOL_PATH" ]; then
    export HCCL_VM_PATH="$TOOL_PATH"
    echo "使用 --tool_path 参数中的 HCCL_VM_PATH: $HCCL_VM_PATH"
else
    export HCCL_VM_PATH=$(pwd)
    echo "使用当前目录作为 HCCL_VM_PATH: $HCCL_VM_PATH"
fi

# 4.校验并获取 CANN 安装目录
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "错误: 环境变量 ASCEND_HOME_PATH 未设置，请参照source /home/workspace/Ascend/cann/set_env.sh设置"
    exit 1
fi

# 5.校验并获取 hccl 代码目录
if [ "$BUILD_HCCL" = true ] && [ -z "$HCCL_CODE_HOME" ]; then
    echo "错误: 环境变量 HCCL_CODE_HOME 未设置"
    exit 1
fi

# 6.校验并获取 hcomm 代码目录
if [ "$BUILD_HCOMM" = true ] && [ -z "$HCOMM_CODE_HOME" ]; then
    echo "错误: 环境变量 HCOMM_CODE_HOME 未设置"
    exit 1
fi

# 5.获得CANN安装目录: ASCEND_INSTALL_PATH
ASCEND_INSTALL_PATH=$(dirname "$ASCEND_HOME_PATH")

echo "--- 环境变量解析成功 ---"
echo "HCCL_VM_PATH: $HCCL_VM_PATH"
echo "ASCEND_INSTALL_PATH: $ASCEND_INSTALL_PATH"
echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
if [ "$BUILD_HCCL" = true ]; then
    echo "HCCL_CODE_HOME: $HCCL_CODE_HOME"
fi
if [ "$BUILD_HCOMM" = true ]; then
    echo "HCOMM_CODE_HOME: $HCOMM_CODE_HOME"
fi
echo "构建配置:"
echo "  - HCCL: $BUILD_HCCL"
echo "  - HCOMM: $BUILD_HCOMM"
echo "--------------------------"

# ==========================================
# 第二步：构建并安装 HCOMM
# ==========================================

# 关闭 Linux 系统对 Python 的保护锁，解决 pip 装包时报错
sudo rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

if [ "$BUILD_HCOMM" = true ]; then
    echo "正在开始构建 HCOMM..."
    cd "$HCOMM_CODE_HOME" || exit 1
    bash build.sh --full

    if [ $? -eq 0 ]; then
        echo "HCOMM 构建成功，准备安装..."
        # 自动匹配版本号run包
        CANN_HCOMM_PACKAGE=$(ls -t "$HCOMM_CODE_HOME"/build_out/cann-hcomm_*_linux-x86_64.run | head -n 1)
        
        # 检查是否找到安装包
        if [ ! -f "$CANN_HCOMM_PACKAGE" ]; then
            echo "错误: 未找到 HCOMM 安装包(.run文件)"
            exit 1
        fi
        
        echo "找到安装包: $CANN_HCOMM_PACKAGE"
        yes y | "$CANN_HCOMM_PACKAGE" --full --install-path="$ASCEND_INSTALL_PATH"
    else
        echo "错误: HCOMM 构建失败"
        exit 1
    fi
else
    echo "跳过 HCOMM 构建和安装..."
fi

# ==========================================
# 第三步：构建并安装 HCCL
# ==========================================
if [ "$BUILD_HCCL" = true ]; then
    echo "正在开始构建 HCCL..."
    cd "$HCCL_CODE_HOME" || exit 1
    bash build.sh --full

    if [ $? -eq 0 ]; then
        echo "HCCL 构建成功，准备安装..."
        # 自动匹配版本号run包
        CANN_HCCL_PACKAGE=$(ls -t "$HCCL_CODE_HOME"/build_out/cann-hccl_*_linux-x86_64.run | head -n 1)
        
        # 检查是否找到安装包
        if [ ! -f "$CANN_HCCL_PACKAGE" ]; then
            echo "错误: 未找到 HCCL 安装包(.run文件)"
            exit 1
        fi
        
        echo "找到安装包: $CANN_HCCL_PACKAGE"
        yes y | "$CANN_HCCL_PACKAGE" --full --install-path="$ASCEND_INSTALL_PATH"
    else
        echo "错误: HCCL 构建失败"
        exit 1
    fi
else
    echo "跳过 HCCL 构建和安装..."
fi

# ==========================================
# 第四步：拷贝驱动依赖库文件
# ==========================================
echo "正在拷贝驱动依赖库文件..."

# 1.定义源路径和目标路径
SOURCE_DIR="$ASCEND_INSTALL_PATH/cann/x86_64-linux/devlib/device"
TARGET_DIR="$HCCL_VM_PATH/hccl_vm_install/lib/aarch64"

# 2.检查目标目录是否存在，不存在则创建
if [ ! -d "$TARGET_DIR" ]; then
    echo "目录 $TARGET_DIR 不存在，正在创建..."
    mkdir -p "$TARGET_DIR"
fi

# 3.拷贝指定的动态库文件 (libascend_hal.so, libc_sec.so, libmmpa.so)
# 使用 -f 确保覆盖，-v 显示拷贝过程
if [ -d "$SOURCE_DIR" ]; then
    cp -fv "$SOURCE_DIR/libascend_hal.so" "$SOURCE_DIR/libc_sec.so" "$SOURCE_DIR/libmmpa.so" "$TARGET_DIR/"
else
    echo "错误: 源目录 $SOURCE_DIR 不存在，请检查 CANN 是否正确安装。"
    exit 1
fi

# ==========================================
# 第五步：拷贝并解压 aicpu 相关的 tar.gz 包
# ==========================================
echo "正在处理 aicpu 相关的构建产物..."

# 1. 定义源文件路径
HCCL_AICPU_TAR=$(find "$HCCL_CODE_HOME" -name "aicpu_hccl.tar.gz" -type f | head -n 1)
HCOMM_AICPU_TAR=$(find "$HCOMM_CODE_HOME" -name "aicpu_hcomm.tar.gz" -type f | head -n 1)

# 2.确保目标目录存在
mkdir -p "$TARGET_DIR"

# 3.拷贝并立即解压 - aicpu_hccl.tar.gz
if [ "$BUILD_HCCL" = true ]; then
    if [ -f "$HCCL_AICPU_TAR" ]; then
        echo "正在解压 aicpu_hccl.tar.gz..."
        tar -zxvf "$HCCL_AICPU_TAR" -C "$TARGET_DIR" --strip-components=1
    else
        echo "警告: aicpu_hccl.tar.gz 未找到 $HCCL_AICPU_TAR"
        exit 1
    fi
else
    echo "跳过 aicpu_hccl.tar.gz 处理..."
fi

# 4.拷贝并立即解压 - aicpu_hcomm.tar.gz
if [ "$BUILD_HCOMM" = true ]; then
    if [ -f "$HCOMM_AICPU_TAR" ]; then
        echo "正在解压 aicpu_hcomm.tar.gz..."
        tar -zxvf "$HCOMM_AICPU_TAR" -C "$TARGET_DIR" --strip-components=1
    else
        echo "警告: aicpu_hccl.tar.gz 未找到 $HCOMM_AICPU_TAR"
        exit 1
    fi
else
    echo "跳过 aicpu_hcomm.tar.gz 处理..."
fi

sudo chmod -R 755 "$TARGET_DIR"

echo "所有任务均已执行完成！"
