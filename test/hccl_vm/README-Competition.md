# HCCL-VM 指导手册

## 1. 概述

HCCL-VM是面向华为昇腾NPU卡的高性能集合通信的虚拟执行环境，该工具旨在无真实昇腾硬件的条件下，实现HCCL集合通信算子的开发和功能验证。

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. 前置依赖

|   依赖项   |     版本要求     |
| -------- | ------------- |
| 系统架构  | x86_64 Ubuntu22.04、Ubuntu24.04 |
| 规格约束  | Ascend950，其余参照[约束详情](#45-工具规格约束) |

### 2.1 CANN包安装

安装最新版本CANN Toolkit开发套件包和CANN ops算子包 [下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/20260701000328953/)

```bash
# 确保安装包具有可执行权限
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# 安装命令
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```

### 2.2 hccl_test编译

hccl_test是昇腾官方提供的HCCL性能测试工具，详见[HCCL性能测试工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html)，HCCL-VM支持在虚拟环境中运行hccl_test用例。请先参照[hccl_test用例构建](#42-hccl-test用例构建)章节进行用例二进制程序的编译。

备注：未来支持Pytorch用例。

---

## 3. 快速上手

### 3.1 工具构建&安装

#### 3.1.1 自动构建

一行完成依赖安装、源码拉取、CANN 检测与编译（默认 `campus-2026` 配套方案）：

```bash
# 1. 创建工作目录
mkdir -p /home/workspace
cd /home/workspace

# 2.下载自动构建&安装脚本
curl -fsSL https://raw.gitcode.com/cann/hcomm/raw/competition%2Fcampus-2026/test/hccl_vm/hccl_vm_installer | bash
```

也可下载后本地运行（便于先审阅或离线分发）：`bash hccl_vm_installer`；追加参数用 `... | bash -s -- --workspace /root/hvm`。

**前提**：x86_64 Linux；工具链需满足 hcomm build.md 要求——gcc/g++ 7.3.0–13.3.x、cmake ≥ 3.16.0（同时约束宿主与 aarch64 交叉编译器）。Ubuntu 22.04 / 24.04 开箱即用；更高版本默认 gcc（14/15）超范围，脚本会告警并继续尝试，建议在满足范围的环境编译。

**CANN**：脚本只在工作目录 `<workspace>/Ascend`（或 `--ascend-path` 指定的路径）探测 CANN；未找到即自动下载配套版本并安装到该处，root 与普通用户行为一致。`--offline` 只检测、从不下载；内网无公网时自动回退为打印自备 CANN 引导。

**hccl_test**：默认一并编译 OpenMPI 与 hccl_test 性能测试工具，`--skip-hccl-test` 可关闭。

**常用参数**：
- `--profile <名>`：配套方案（默认 `campus-2026`，`--list-profiles` 列全部）
- `--workspace <路径>`：工作目录，源码／编译／产物所在（默认当前目录）
- `--ascend-path <路径>`：指定 CANN 目录，有则复用、无则装到此处
- `--reinstall-cann`：重新下载覆盖现有 CANN（版本不匹配时用；默认保留）
- `--offline`：只用现有 CANN、从不下载
- `--skip-hccl-test`：跳过 hccl_test 编译
- `-h`：完整帮助

完成后工具位于 `<工作目录>/hcomm/test/hccl_vm/hccl_vm_install/bin/hccl-vm`。删除工作目录即可清理本工具产物（apt 安装的系统依赖如需卸载请自行 `apt remove`），本工具不改动 CANN。

#### 3.1.2 手动构建

```bash
# 1. 创建工作目录
mkdir -p /home/workspace
cd /home/workspace

# 2. 下载依赖源码
git clone -b competition/campus-2026 https://gitcode.com/cann/hccl.git
git clone -b competition/campus-2026 https://gitcode.com/cann/hcomm.git

# 3. 安装第三方依赖
sudo apt-get update
sudo apt install build-essential cmake libsqlite3-dev rdma-core libibverbs-dev pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user-static binfmt-support

# 4. 编译HCCL-VM工具，下载hcomm代码之后，工具源码所在路径：/home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full

# 5. 编译AI_CPU展开模式所需设备侧符号(不运行AI_CPU展开模式可跳过)
bash ./build_pkg.sh
```

### 3.2 alltoall示例算子编译和产物拷贝

```bash
# 1.进入alltoall示例算子目录
cd /home/workspace/hccl/hccl-campus-alltoall

# 2.编译alltoall示例算子
source /home/workspace/Ascend/cann/set_env.sh
bash build.sh

# 3.编译产物(hccl.h libhccl.so libhccl_device.so)拷贝
cp ./build/include/hccl.h /home/workspace/Ascend/cann/x86_64-linux/include/hccl/
cp ./build/lib64/libhccl.so /home/workspace/Ascend/cann/x86_64-linux/lib64/
cp ./build/lib64/libhccl_device.so /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/lib/aarch64/
```

### 3.3 使用示例

#### 3.3.1 环境配置

请参照[hccl_rootinfo文件内容](#47-hccl_rootinfojson文件)，创建并配置hccl_rootinfo.json文件。

#### 3.3.2 CCU模式

1. 环境变量配置。

```bash
# 进入工具安装目录
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
```

2. 执行

```bash
# 需要进入到新的bin文件目录下执行hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# 选择本次算子执行的通信域配置文件（详见4.3章节，在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/alltoall_test -b 64 -e 64 -d int32 -o sum -r 0 -w 0 -n 1 -c 0 > log.txt

# 执行checker校验
(hvm)$> hccl-vm plugin run @checker

# 退出工具终端
(hvm)$> exit
```

3. 验证hccl_test用例运行结果 [Checker结果查看](#492-checker插件结果)

#### 3.3.3 AICPU模式

AICPU展开模式需要将算法展开步骤放到设备侧执行，因此hccl-vm工具需要将HCCL的设备侧的符号编译并模拟执行。由于设备侧符号是ARM架构的，因此在X86环境上编译时需要借助交叉编译器，运行时需要借助QEMU实现AICPU模式的模拟运行。

设备侧符号使用hccl和hcomm的源码编译，为了保证Host与Device通信协议正确，需要同时编译Host侧的安装包并进行替换安装。

-备注：已在3.1章节通过执行`bash ./build_pkg.sh`完成。

1. 环境变量配置。

```bash
# 进入工具安装目录
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"
```

2. 执行

```bash
# 需要进入到新的bin文件目录下执行hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only

# 选择本次算子执行的通信域配置文件（详见4.3章节，在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/alltoall_test -b 64 -e 64 -d int32 -o sum -r 0 -w 0 -n 1 -c 0 > log.txt

# 执行checker校验
(hvm)$> hccl-vm plugin run @checker

# 退出工具终端
(hvm)$> exit
```

4. 验证hccl_test用例运行结果 [Checker结果查看](#492-checker插件结果)

#### 3.3.4 AIV模式

1. 环境变量配置。

```bash
# 进入工具安装目录
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AIV"
```

2. 执行

```bash
# 需要进入到新的bin文件目录下执行hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# 如需启用runner插件（可选）
(hvm)$> hccl-vm plugin install @runner

# 选择本次算子执行的通信域配置文件（详见4.3章节，在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/alltoall_test -b 64 -e 64 -d int32 -o sum -r 0 -w 0 -n 1 -c 1 > log.txt

# 执行checker校验
(hvm)$> hccl-vm plugin run @checker

# 退出工具终端
(hvm)$> exit
```

3. 验证hccl_test用例运行结果 [Runner结果查看](#491-runner插件结果) [Checker结果查看](#492-checker插件结果)

### 3.4 Pytorch用例示例

暂不支持。

### 3.5 hccl代码修改验证示例

修改alltoall示例算子代码后，需先执行[算子编译](#32-alltoall示例算子编译和产物拷贝)，然后参照[使用示例](#33-使用示例)中对应算子展开模式的指导执行用例。

---

## 4. 详细指导

### 4.1 工具环境变量配置

**HCCL-VM环境变量说明**：

| 环境变量                      | 用途                                                                                                        | 示例                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME`         | HCCL-VM编译指定HCCL源码路径。默认未配置。     | `export HCCL_CODE_HOME=/home/workspace/hccl`                     |
| `HCOMM_CODE_HOME`         | HCCL-VM编译指定HCOMM源码路径。默认未配置。     | `export HCOMM_CODE_HOME=/home/workspace/hcomm`                     |
| `HCCLVM_ENABLE_DUMP_DATA` | 使能Runner插件dump input\&output数据。若使能，则在测试用例执行过程中，会将每个算子的input\&output数据dump到all\_rank\_input\_output.txt文件中 | `export HCCLVM_ENABLE_DUMP_DATA=1 使能，export HCCLVM_ENABLE_DUMP_DATA=0 禁用` |

### 4.2 HCCL-Test用例构建

hccl_test用例源码在CANN包安装目录下，支持OpenMPI和MPICH两种环境编译、运行，运行时差异详见[OpenMPI和MPICH环境用例执行差异](#48-openmpi和mpich环境用例运行差异)，本用例指导中以OpenMPI环境为例。

#### 4.2.1 OpenMPI环境编译

1. 安装OpenMPI

```bash
sudo apt-get update
sudo apt install openmpi-bin libopenmpi-dev
```

2. 编译hccl_test

```bash
# 修改CANN安装目录权限
chmod -R 755 /home/workspace/Ascend

# 进入hccl_test用例源码目录
cd /home/workspace/Ascend/cann/tools/hccl_test

# 设置CANN环境变量
source /home/workspace/Ascend/cann/set_env.sh

# 临时修改Makefile脚本
if ! grep -q '\-lmpi_cxx' Makefile; then
    sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
fi

# 编译hccl_test用例(默认编译所有算子用例)
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}

# 编译hccl_test指定算子用例
# MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH} alltoall_test
```

#### 4.2.2 MPICH环境编译

假设mpich的路径为: `/usr/lib/mpich`。

```bash
# 进入hccl_test用例源码目录
cd /home/workspace/Ascend/cann/tools/hccl_test

# 设置CANN环境变量
source /home/workspace/Ascend/cann/set_env.sh

# 配置环境变量
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# 编译hccl_test用例
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 昇腾集群拓扑配置文件说明

工具提供了`hccl-vm mock-comm`命令读取和配置算子通信域配置文件。通信域配置文件格式为yaml，路径为hccl_vm_install/config/topo_meta。若目录中没有对应的通信域配置文件，则用户需要先创建一个。

- 对称拓扑场景：每个Server内包含相同数据的NPU设备

```yaml
# 124.yaml配置详情
# 1. 全局统计信息 podNum, serNum, rankNum 都小于 1024
meta:
  podNum: 1  # 总的超节点数
  serNum: 2  # 总的server数
  rankNum: 8 # 总的rank数

# 2. 详细拓扑结构 
topology:
  - podId: 0
    servers:
      - serId: 0
        # 每个server实际跑的rank的local id
        ranks: [0, 1, 2, 3]
      - serId: 1
        # 每个server实际跑的rank的local id
        ranks: [0, 1, 2, 3]
```

```bash
# 选择使用124.yaml代表两个Server，每个Server内4个NPU设备
(hvm)$> hccl-vm mock-comm 124
# mpirun启动参数-np需要对应上述总共8个NPU设备
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 8 
```

- 非对称拓扑场景：每个Server内包含不同数据的NPU设备

```yaml
# 12_2_4.yaml配置详情
# 1. 全局统计信息 podNum, serNum, rankNum 都小于 1024
meta:
  podNum: 1  # 总的超节点数
  serNum: 2  # 总的server数
  rankNum: 6 # 总的rank数

# 2. 详细拓扑结构 
topology:
  - podId: 0
    servers:
      - serId: 0
        # 每个server实际跑的rank的local id
        ranks: [0, 1]
      - serId: 1
        # 每个server实际跑的rank的local id
        ranks: [0, 1, 2, 3]
```

```bash
# 选择使用12_2_4.yaml代表两个Server，第一个Server内2个NPU设备，第二个Server内4个NPU设备
(hvm)$> hccl-vm mock-comm 12_2_4
# mpirun启动参数-np需要对应上述总共6个NPU设备
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 6 
```

### 4.4 hccl\_config.sh文件说明

hccl\_config.sh文件中包含了HCCL\_Test用例运行所需的环境变量配置。其中的环境变量与Hccl\_Test用例在真机环境中的环境变量一致。
用户需要根据自己的用例和需求，修改hccl\_config.sh脚本，配置HCCL用例运行环境变量。

```bash
#!/bin/bash
# hccl_config.sh - HCCL 环境变量配置

remove_files_by_prefix() {
  if [ "$#" -ne 1 ]; then
    echo "Usage: remove_files_by_prefix <prefix>" >&2
    return 2
  fi

  local prefix="$1"
  if [ -z "$prefix" ]; then
    return 0
  fi

  shopt -s nullglob
  local any_deleted=0
  for f in "${prefix}"*; do
    if [ -f "$f" ]; then
      rm -f -- "$f" && any_deleted=1
    fi
  done
  shopt -u nullglob

  # 无论是否删除了文件，均返回 0，确保脚本继续执行
  return 0
}

# 清理 data/ 目录的冗余文件（CCU 模式下生成的临时文件）
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# 设置CANN环境变量
source /home/workspace/Ascend/cann/set_env.sh

# 关闭hccl心跳功能
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# 设置 HCCL-VM 安装路径，基于脚本自身位置推断（兼容 bin/ 与 script/ 子目录）
_INSTALL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$(basename "${_INSTALL_SCRIPT_DIR}")" in
    bin|script)
        export HCCL_VM_INSTALL_DIR="$(dirname "${_INSTALL_SCRIPT_DIR}")"
        ;;
    *)
        export HCCL_VM_INSTALL_DIR="${_INSTALL_SCRIPT_DIR}"
        ;;
esac
unset _INSTALL_SCRIPT_DIR

# 配置LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH

# 设置 ranktable.json文件路径 (与 mock-comm 生成路径保持一致)
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# 设置日志级别
export ASCEND_GLOBAL_LOG_LEVEL=1

# 设置日志打屏输出
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# 设置 HCCL 运行模式（CCU、AI_CPU、AIV等展开模式）
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# 或设置 HCCL 运行参数(AI_CPU展开模式) AI_CPU模式环境变量与其他模式不可同时设置
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# 或设置 HCCL 运行参数(AIV展开模式) AIV模式环境变量与其他模式不可同时设置
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 工具规格约束

**支持算子类型**：

工具支持算子类型包含：allgather/allreduce/alltoall/reduce/reduce\_scatter/scatter/alltoallv.

**支持数据类型**：

HCCL-VM工具支持数据类型包含：int8/int16/int32/fp16/fp32/uint8/uint16/uint32/bfp16/hif8/fp8e4m3/fp8e5m2/fp8e8m0.

HCCL-VM Runner插件支持数据类如下：

| ReduceOp | DataType                           |
| -------- | ---------------------------------- |
| `ADD`    | `int8/int16/int32/uint8` |
| `MIN`    | `int8/int16/int32/uint8` |
| `MAX`    | `int8/int16/int32/uint8` |

**硬件规格**：

目前本工具仅适配支持Ascend950芯片。单server内最大支持8张卡；超过8张卡，需要跨server运行。

### 4.6 HCCL-VM插件功能

#### 4.6.1 Runner插件

Runner插件，即模拟执行Hccl业务编排生成的任务，输出output数据。
模拟运行器插件在hccl\_test用例执行过程中默认**关闭**。hccl\_test用例调用算子接口后，通过aclrtSynchronizeStream接口等待算子任务执行完成。模拟运行器工具会等待所有rank都处于等待状态时，开启模拟执行所有rank的task。执行完成后，通知各个rank的用例进行继续执行。
用例执行完成后，用户可以通过执行目录下的"all\_rank\_input\_output.txt"文件，查看每个rank的input buffer和output buffer数据。此功能默认不开启，用户可以在用例执行前通过对应命令开启。

**安装与卸载**：

Runner插件支持通过 `hccl-vm plugin install/uninstall` 命令进行安装和卸载。需在进入hccl-vm工具命令行后、执行算例前安装runner插件，后续的执行才会运行runner。

```bash
# 安装runner插件
(hvm)$> hccl-vm plugin install @runner

# 卸载runner插件
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker插件

Checker插件，即算法分析器插件：功能是将hccl生成的所有task形成一个DAG图，并且通过分析DAG图，判断是否存在内存冲突；通过模拟执行DAG图，判断是否存在语义错误等问题。
算法分析器插件，是由用户自行通过命令启动执行。

Checker插件正处于新旧交替阶段，Checker V3为原Checker的重构版，主要提高了校验性能，在默认情况下将会运行新Checker（Checker V3），可以通过修改Checker的`manifest.json`文件中的配置参数进行调整。

```bash

# 配置文件位于 /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  "name": "checker",        // Checker插件名
  "version": "1.0.0",       // Checker插件版本
  "entry": "./checker",     // Checker插件启动指令
  "dependency": {
      "min_core_version": "1.0.0"
  },
  "setting": {              // Checker插件配置项
      "enable_new_checker": true,           // 是否启用新Checker（Checker V3，默认开启）
      "enable_old_checker": false,          // 是否启用老Checker（默认关闭）
      "enable_insight_dump": false,         // 是否启用可视化数据输出（默认关闭，仅支持老Checker）
      "enable_memory_snapshot_dump": false  // 是否启用可视化内存快照数据输出（默认关闭，仅支持老Checker，需要先开启可视化数据输出"enable_insight_dump"）
  }
}
```

### 4.7 hccl_rootinfo.json文件

目前工具初始化通信域使用的是ranktable.json文件，因此hccl_rootinfo.json文件的作用仅限于提供topo.json文件的路径。
若/etc路径下没有hccl_rootinfo.json文件，则用户需自行创建该文件，内容如下：

```json
{
  "version": "2.0",
  "topo_file_path": "/home/workspace/hcomm/test/hccl_vm/hccl_vm_install/data/topo.json"
}
```

### 4.8 OpenMPI和MPICH环境用例运行差异

运行hccl_test用例前，用户可以通过which命令判断当前环境使用的是哪个mpirun。

#### 4.8.1 环境变量配置差异

环境中一般默认配置的是OpenMPI，若用户使用OpenMPI运行用例，一般不需要额外配置环境变量。
若用户使用MPICH环境运行用例，需要按照如下方式配置环境变量：

```bash
# 配置mpich环境变量
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.8.2 mpirun命令参数差异

OpenMPI环境，用户按照如下命令运行hccl_test用例：

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**参数说明**：

 - --allow-run-as-root：OpenMPI专属参数，允许以root用户运行MPI进程，用于在没有root权限的环境中运行。
 - --oversubscribe：OpenMPI专属参数，放开CPU槽位(slot)限制，允许单节点启动[进程数>CPU逻辑核数]，也就是超额超配跑进程。
 - -np 2：指定进程数为2，与节点数一致。

MPICH环境，用户按照如下命令运行hccl_test用例：

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**参数说明**：

 - -np 2：指定进程数为2，与节点数一致。

### 4.9 结果查看

#### 4.9.1 Runner插件结果

如果在hccl-vm终端执行hccl-vm plugin install @runner安装插件后，算子流程执行完成后会自动触发runner插件的执行，最终结果依赖hccl_test进行校验，用户需关注在重定向的日志文件中是否存在[error]级别日志和最终校验结果：

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.9.2 Checker插件结果

在hccl-vm终端内执行hccl-vm plugin run @checker后，Checker校验流程及结果会打印在终端内，用于需关注是否存在[error]级别日志和最终校验结果：

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---
### 4.10 大块内存复用（仅校验模式）

仅校验模式用于大规模集群仅运行 Checker 校验的场景。开启后，单块 200MB 到 4GB 的大内存申请复用同一块 4GB 共享区 `HcclCommPool`，各 rank 共享、允许互相覆盖，以此大幅降低 `/dev/shm` 占用。此时大块内容不保证正确，仅适用于不读取缓冲区数据的 Checker V3 校验链路，需要数值正确的结果时请勿开启。

仅校验模式是会话级开关，在 `start` 子命令后追加 `--check-only` 显式开启；不加时为默认的普通模式，大块走真实独立分配，正确性无损。小于 200MB 的申请始终走真实分配，单块大于 4GB 在仅校验模式下直接报错拒绝。仅校验模式与 Runner 不互斥，但在仅校验模式开启时安装 Runner，大块复用仍会生效、可能覆盖 Runner 数据，工具会打印告警。

```bash

# 启动工具时开启仅校验模式
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only
```

***

## 5 附录

### 开源第三方软件依赖

编译本项目时，依赖的第三方开源软件列表如下，离线编译场景可下载并重命名软件包后放置在本项目内的third_party目录下。

| 开源软件       | 版本          |下载地址 |
| ------------  | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11         | 2.2.0         | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json          | 3.11.3        | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog        | 1.11.0        | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp      | 0.8.0         | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| sqlite        | 3.51.0        | [sqlite-amalgamation-3510300.zip](https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip) |
| googletest    | 1.14.0        | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |

### 术语表

| 术语       | 说明                                                      |
| -------- | ------------------------------------------------------- |
| HCCL     | Huawei Collective Communication Library，华为集合通信库         |
| NPU      | Neural Processing Unit，神经网络处理器                          |
| CANN     | Compute Architecture for Neural Networks，华为昇腾 AI 处理器软件栈 |
| MPI      | Message Passing Interface，消息传递接口                        |
| CCU      | Collective Communication Unit，集合通信单元                    |
| Topology | 拓扑，设备连接关系                                               |
| Rank     | 进程编号，在 MPI 中的标识                                         |

---

**文档版本**：v1.1。
**最后更新**：2026-06-30。
