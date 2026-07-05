#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# HCCL-VM 一键安装：装依赖 -> 检测 CANN -> 拉取公开源码(hcomm/hccl) -> 编译。
#
# ============================ 依赖关系说明 ============================
# 组件角色：
#   CANN 包(用户提供)      : 终态发布包；内含编译好的 hcomm/hccl 库(.so)、对外导出头
#                           (如 hccl_res_expt.h 中的 ThreadType 等类型)、ACL runtime。
#   hcomm 源码(脚本 clone) : ① 承载 checker 本体(即 test/hccl_vm)；② 提供 hcomm 内部
#                           源码(src/platform、framework、base_comm…)供 checker 编译。
#   hccl 源码(脚本 clone)  : 提供 hccl 内部源码(src/ops、hcomm_dlsym…)供 checker 编译
#                           device/AICPU/AIV。上述内部源码 CANN 包里不包含。
#
# 官方(正常)配套方向：CANN 版本为锚 -> hcomm/hccl 源码用与该 CANN 版本一致的发布
#   tag(见 release-management 的 cann-hcomm/cann-hccl 行)。即“源码跟随 CANN”。
#
# 本工具的配套：checker 存在于 hcomm 的竞赛/主线分支（发布 tag 里没有 checker）。默认按
#   profile competition/campus-2026 取 hcomm/hccl 同一竞赛分支；CANN 版本由该分支 build.md
#   提取（见 --list-profiles）。源码与 CANN 需同代配套，不配套时 device 侧会因缺新接口
#   (如 ThreadType)编译失败，本脚本会在失败时提示更换配套 CANN。
# ====================================================================
set -euo pipefail

# 颜色（仅 stdout 为终端时启用；被管道/重定向时关闭，避免日志里出现乱码转义）
color_enabled() { [ -t 1 ]; }
if color_enabled; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; NC=''
fi

# 全局变量（可由环境变量或命令行覆盖）
WORKSPACE=${WORKSPACE:-}
ASCEND_PATH=${ASCEND_PATH:-}
HCOMM_PATH=${HCOMM_PATH:-}
HCCL_PATH=${HCCL_PATH:-}
CHECKER_PATH=${CHECKER_PATH:-}
# 分支引用：不在此写死默认，默认由 profile 落定（见 apply_defaults）。
# 记录“环境变量是否显式提供”，用于优先级链 CLI > env > profile。
ENV_HCOMM_REF="${HCOMM_REF:-}"
ENV_HCCL_REF="${HCCL_REF:-}"
CLI_HCOMM_REF=""
CLI_HCCL_REF=""
HCOMM_REF=""
HCCL_REF=""
HCOMM_REF_SRC=""
HCCL_REF_SRC=""
PROFILE=""
PROFILE_CUSTOM=false
VERBOSE=false
STEP_TOTAL=7
LOG_DIR=""
RUN_TS=""

# 版本标识（发布时更新；SCRIPT_COMMIT 可由同步流程回填，未知则保持 unknown）
SCRIPT_VERSION="1.0.0"
SCRIPT_COMMIT="unknown"

# 显示信息
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# 显示警告
warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

# 显示错误并退出
error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

# 步骤进度：[n/total] 消息
step() {
    echo -e "${GREEN}[$1/$2]${NC} $3"
}
# 版本串：commit 已回填才带 commit，否则只显示版本号（避免露 "commit unknown"）
version_line() {
    if [ "${SCRIPT_COMMIT}" = "unknown" ]; then
        echo "v${SCRIPT_VERSION}"
    else
        echo "v${SCRIPT_VERSION} (commit ${SCRIPT_COMMIT})"
    fi
}
# 起始信息块（非 ASCII art）
banner() {
    info "HCCL-VM 一键安装 $(version_line)"
    info "配套方案: ${PROFILE}"
    info "预计耗时约 5~15 分钟（拉源码 + 编译，视网络与机器），请保持网络畅通、勿中断。"
}
# 成功总结：产物 + 可复制的下一步
success_summary() {
    local bin="${CHECKER_PATH}/hccl_vm_install/bin/hccl-vm"
    info "HCCL-VM 安装完成！"
    info "产物: ${bin}"
    if [ -f "${CHECKER_PATH}/README.md" ]; then
        info "用法: 参考 ${CHECKER_PATH}/README.md（快速上手一节）"
    fi
}

# 检查 apt 包是否已安装（dpkg-query 精确查询，无管道，避免 dpkg -l|grep 在 pipefail 下被 SIGPIPE 误判）
check_package() {
    local status
    status="$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null)" || return 1
    [ "$status" = "install ok installed" ]
}

# 决定 apt 提权方式：已是 root 输出空前缀；非 root 但有 sudo 输出 "sudo"；两者皆无返回非零
apt_privilege_prefix() {
    if [ "$(id -u)" -eq 0 ]; then
        echo ""
    elif command -v sudo &> /dev/null; then
        echo "sudo"
    else
        return 1
    fi
}

# 判断操作系统是否受支持（仅 Linux）
is_supported_os() {
    [[ "$1" == "Linux" ]]
}

# 归一化 CPU 架构；未知架构返回非零
normalize_arch() {
    case "$1" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        *) return 1 ;;
    esac
}

# ── Profile 注册表：一个 profile 只映射 hcomm/hccl 分支；首行为默认；新增版本加一行即可 ──
# 每行格式：hcomm_ref|hccl_ref|说明
profile_all_names() {
    printf '%s\n' "campus-2026" "main"
}
profile_row() {
    case "$1" in
        campus-2026) echo "competition/campus-2026|competition/campus-2026|竞赛固定，CANN 从 build.md 提取" ;;
        main)        echo "master|master|主线最新，随官方更新" ;;
        *) return 1 ;;
    esac
}
profile_default()    { profile_all_names | head -n 1; }
profile_exists()     { profile_row "$1" >/dev/null 2>&1; }
profile_hcomm_ref()  { profile_row "$1" | cut -d'|' -f1; }
profile_hccl_ref()   { profile_row "$1" | cut -d'|' -f2; }
profile_desc()       { profile_row "$1" | cut -d'|' -f3; }

# 显示帮助信息
show_help() {
    echo "HCCL-VM 一键安装脚本 $(version_line)"
    echo ""
    echo "新手直接运行，无需任何参数（全程非交互，自动安装依赖）:"
    echo "  curl -fsSL https://raw.gitcode.com/cann/hcomm/raw/competition%2Fcampus-2026/test/hccl_vm/hccl_vm_install.sh | bash"
    echo "  （或下载后本地运行: bash hccl_vm_install.sh）"
    echo ""
    echo "用法:"
    echo "  bash hccl_vm_install.sh [选项]"
    echo ""
    echo "常用选项:"
    echo "  --profile <名称>       配套方案，默认: campus-2026（竞赛固定）"
    echo "  --list-profiles        查看所有可选配套方案"
    echo "  --workspace <路径>     工作目录 (默认: 当前目录)"
    echo "  --ascend-path <路径>   CANN 安装路径 (默认: 自动探测 /usr/local/Ascend 等)"
    echo "  --version              显示版本"
    echo "  -h, --help             显示此帮助"
    echo ""
    echo "高级选项（自定义分支组合，可能与 CANN 不配套）:"
    echo "  --hcomm-ref <分支/标签>  覆盖 profile 的 hcomm 分支（仅支持分支/标签名，非 commit）"
    echo "  --hccl-ref <分支/标签>   覆盖 profile 的 hccl 分支（仅支持分支/标签名，非 commit）"
    echo "  --hcomm-path/--hccl-path/--checker-path <路径>  自定义源码路径"
    echo "  --verbose              打印全过程日志（含 apt 与编译输出）"
}

# 打印可选 profile（供客户查看）
list_profiles() {
    echo "可用配套方案（--profile 可选值）:"
    echo ""
    local first=true name row hcomm hccl desc tag
    while IFS= read -r name; do
        row="$(profile_row "${name}")"
        hcomm="$(echo "${row}" | cut -d'|' -f1)"
        hccl="$(echo "${row}" | cut -d'|' -f2)"
        desc="$(echo "${row}" | cut -d'|' -f3)"
        if [ "${first}" = true ]; then tag=" (默认)"; first=false; else tag=""; fi
        echo "  ${name}${tag}"
        echo "    hcomm 分支: ${hcomm}"
        echo "    hccl  分支: ${hccl}"
        echo "    说明: ${desc}"
        echo ""
    done < <(profile_all_names)
    echo "注: profile 名 main 对应 git 分支 master。"
    echo "用法示例: --profile main"
}

# 校验带值选项确有参数值（避免 set -u 下末位缺值报晦涩的 $2 unbound）
require_val() {
    # $1=选项名 $2=剩余参数个数
    [[ "$2" -ge 2 ]] || error "选项 $1 需要一个参数值"
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --workspace)
                require_val "$1" "$#"
                WORKSPACE="$2"
                shift 2
                ;;
            --ascend-path)
                require_val "$1" "$#"
                ASCEND_PATH="$2"
                shift 2
                ;;
            --hcomm-path)
                require_val "$1" "$#"
                HCOMM_PATH="$2"
                shift 2
                ;;
            --hccl-path)
                require_val "$1" "$#"
                HCCL_PATH="$2"
                shift 2
                ;;
            --checker-path)
                require_val "$1" "$#"
                CHECKER_PATH="$2"
                shift 2
                ;;
            --hcomm-ref)
                require_val "$1" "$#"
                CLI_HCOMM_REF="$2"
                shift 2
                ;;
            --hccl-ref)
                require_val "$1" "$#"
                CLI_HCCL_REF="$2"
                shift 2
                ;;
            --profile)
                require_val "$1" "$#"
                PROFILE="$2"
                shift 2
                ;;
            --version)
                echo "HCCL-VM 一键安装脚本 $(version_line)"
                exit 0
                ;;
            --list-profiles)
                list_profiles
                exit 0
                ;;
            --verbose)
                VERBOSE=true
                shift 1
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "未知参数: $1。运行 bash hccl_vm_install.sh --help 查看可用选项。"
                ;;
        esac
    done
}

# 统一设置依赖其他变量的默认值（parse_args 之后调用）
apply_defaults() {
    WORKSPACE=${WORKSPACE:-$(pwd)}
    ASCEND_PATH=${ASCEND_PATH:-/usr/local/Ascend}
    HCOMM_PATH=${HCOMM_PATH:-${WORKSPACE}/hcomm}
    HCCL_PATH=${HCCL_PATH:-${WORKSPACE}/hccl}
    CHECKER_PATH=${CHECKER_PATH:-${HCOMM_PATH}/test/hccl_vm}

    # profile 落定（默认=注册表首行）
    PROFILE="${PROFILE:-$(profile_default)}"
    if ! profile_exists "${PROFILE}"; then
        error "未知 profile: ${PROFILE}。运行 --list-profiles 查看可选方案，或去掉 --profile 使用默认。"
    fi
    # 分支优先级：CLI --*-ref > 环境变量 HCOMM_REF/HCCL_REF > profile 映射
    if [ -n "${CLI_HCOMM_REF}" ]; then
        HCOMM_REF="${CLI_HCOMM_REF}"; HCOMM_REF_SRC="--hcomm-ref"; PROFILE_CUSTOM=true
    elif [ -n "${ENV_HCOMM_REF}" ]; then
        HCOMM_REF="${ENV_HCOMM_REF}"; HCOMM_REF_SRC="环境变量 HCOMM_REF"; PROFILE_CUSTOM=true
    else
        HCOMM_REF="$(profile_hcomm_ref "${PROFILE}")"; HCOMM_REF_SRC="profile ${PROFILE}"
    fi
    if [ -n "${CLI_HCCL_REF}" ]; then
        HCCL_REF="${CLI_HCCL_REF}"; HCCL_REF_SRC="--hccl-ref"; PROFILE_CUSTOM=true
    elif [ -n "${ENV_HCCL_REF}" ]; then
        HCCL_REF="${ENV_HCCL_REF}"; HCCL_REF_SRC="环境变量 HCCL_REF"; PROFILE_CUSTOM=true
    else
        HCCL_REF="$(profile_hccl_ref "${PROFILE}")"; HCCL_REF_SRC="profile ${PROFILE}"
    fi
    LOG_DIR="${WORKSPACE}/.hccl_vm_install_logs"
    RUN_TS="$(date +%Y%m%d-%H%M%S)"
}

# 自定义分支组合时提示一次（逐仓标注实际来源）；已定义 profile（含默认）不提示
notice_custom_refs() {
    [ "${PROFILE_CUSTOM}" = true ] || return 0
    warn "使用自定义分支组合，可能与 CANN 或彼此不配套："
    echo "  hcomm=${HCOMM_REF}（来自 ${HCOMM_REF_SRC}）" >&2
    echo "  hccl=${HCCL_REF}（来自 ${HCCL_REF_SRC}）" >&2
}

# 运行环境自检：必须 Linux + 具备 apt + 受支持架构
check_env() {
    if ! is_supported_os "$(uname -s)"; then
        error "本工具仅支持 Linux。Windows 请在 WSL2 或远端 Linux 中运行。"
    fi
    if ! command -v apt-get &> /dev/null; then
        error "未检测到 apt-get。本工具按 checker 官方要求支持 x86_64 Ubuntu 22.04 及以上（apt）。
  非 apt 系统请在 Ubuntu 22.04+ 上运行，或手动安装等价依赖后用 --hcomm-path/--hccl-path 复用已备源码：
  $(dependency_list | tr '\n' ' ')"
    fi
    if ! normalize_arch "$(uname -m)" &> /dev/null; then
        error "不支持的架构: $(uname -m)（仅 x86_64 / aarch64）。请换用 x86_64 环境。"
    fi
    # 非官方验证环境软警告（不阻断）：官方仅验证 x86_64 Ubuntu 22.04+
    if [ "$(uname -m)" != "x86_64" ]; then
        warn "当前架构 $(uname -m) 非官方验证环境（官方仅验证 x86_64）；可继续，但编译失败概率较高，出问题请优先换 x86_64。"
    fi
    if ! ubuntu_version_ok; then
        warn "当前系统非 Ubuntu 22.04+（官方仅验证该版本）；可继续，但遇到编译问题请优先换官方环境。"
    fi
    # 管道模式（curl|bash）下 stdin 非终端：明确非交互
    if [ ! -t 0 ]; then
        info "检测到管道模式（curl|bash），全程非交互，apt 依赖将自动安装。"
    fi
    # 轻量 CANN 预检：CANN 必须自备且无法自动下载，缺失就提前告知，避免白等 apt+clone
    if ! cann_maybe_present; then
        warn "未检测到 CANN。CANN 必须自备、无法自动下载；可继续拉源码，但编译前必须装好。"
        echo "  现在装 CANN 可 Ctrl-C，装好后重跑（已下载源码会自动复用）；详细引导在拉取源码后给出。" >&2
    fi
}

# 判断是否 Ubuntu 22.04 及以上（读 /etc/os-release，非 Ubuntu 或版本低返回非零）
ubuntu_version_ok() {
    local id ver_major
    [ -r /etc/os-release ] || return 1
    id="$(. /etc/os-release 2>/dev/null && echo "${ID:-}" || true)"
    ver_major="$(. /etc/os-release 2>/dev/null && echo "${VERSION_ID:-}" | cut -d. -f1 || true)"
    [ "${id}" = "ubuntu" ] || return 1
    [ -n "${ver_major}" ] && [ "${ver_major}" -ge 22 ] 2>/dev/null
}

# 轻量判断 CANN 是否可能就绪（只看环境变量与候选 set_env.sh 是否存在，不读 build.md）
cann_maybe_present() {
    [ -n "${ASCEND_HOME_PATH:-}" ] && return 0
    [ -n "${ASCEND_TOOLKIT_HOME:-}" ] && return 0
    local cand
    while IFS= read -r cand; do
        [ -f "${cand}" ] && return 0
    done < <(cann_setenv_candidates)
    return 1
}

# 磁盘空间预检：源码 + --full 编译需要一定空间，不足时提前告警（非阻断）
check_disk() {
    local min_gb=10 target avail_kb avail_gb
    target="${WORKSPACE}"
    while [ ! -d "${target}" ] && [ "${target}" != "/" ] && [ -n "${target}" ]; do
        target="$(dirname "${target}")"
    done
    avail_kb="$(df -Pk "${target}" 2>/dev/null | awk 'NR==2{print $4}' || true)"
    if [ -z "${avail_kb}" ]; then
        return 0    # 查不到就不拦
    fi
    avail_gb=$(( avail_kb / 1024 / 1024 ))
    if [ "${avail_gb}" -lt "${min_gb}" ]; then
        warn "工作目录所在磁盘可用空间约 ${avail_gb}G，低于建议的 ${min_gb}G；源码拉取 + 编译可能空间不足。"
        echo "  可用 --workspace 指向更大分区，或清理后重试。" >&2
    else
        info "磁盘可用空间约 ${avail_gb}G（建议 ≥ ${min_gb}G）"
    fi
}

# 依赖清单：编译基础包 + 运行时 python-yaml(hccl-vm start 生成组网拓扑需) + 仅 x86_64 宿主需要的 device(arm) 交叉编译工具链
dependency_list() {
    local deps=("git" "build-essential" "cmake" "libsqlite3-dev" "libboost-all-dev" "rdma-core" "libibverbs-dev" "pkg-config" "python3" "python3-yaml")
    if [ "$(uname -m)" = "x86_64" ]; then
        deps+=("gcc-aarch64-linux-gnu" "g++-aarch64-linux-gnu" "qemu-user-static" "binfmt-support")
    fi
    printf '%s\n' "${deps[@]}"
}

# 安装第三方依赖
install_deps() {
    info "检查第三方依赖..."
    local dependencies=() dep_line
    while IFS= read -r dep_line; do
        dependencies+=("${dep_line}")
    done < <(dependency_list)
    local missing_deps=()
    for dep in "${dependencies[@]}"; do
        if ! check_package "$dep"; then
            missing_deps+=("$dep")
        fi
    done
    if [ ${#missing_deps[@]} -eq 0 ]; then
        info "所有第三方依赖已安装，跳过安装步骤"
        return
    fi

    info "安装缺失的依赖: ${missing_deps[*]}"
    # 已是 root 直接装、非 root 用 sudo；无权限则给出手动安装指引后退出
    local apt_prefix
    if ! apt_prefix="$(apt_privilege_prefix)"; then
        warn "缺少系统依赖，但当前非 root 且无 sudo，无法自动安装："
        echo "  ${missing_deps[*]}" >&2
        echo "  请让管理员安装后重试，或手动执行：" >&2
        echo "  apt-get update && apt-get install -y ${missing_deps[*]}" >&2
        exit 1
    fi
    # 需 sudo 但当前无免密 sudo 且非交互(stdin 非终端，如 curl|bash)：提前明确提示，避免密码提示被吞而卡死
    if [ -n "${apt_prefix}" ] && ! sudo -n true 2>/dev/null && [ ! -t 0 ]; then
        warn "安装依赖需要 sudo，但当前无免密 sudo 且为非交互环境（stdin 非终端）。"
        echo "  请改用 root 运行，或先执行 sudo -v 预认证后重试；" >&2
        echo "  或手动安装：sudo apt-get update && sudo apt-get install -y ${missing_deps[*]}" >&2
        exit 1
    fi
    # 输出留存到日志文件（不再 >/dev/null 吞掉）；--verbose 时同时回显到终端，失败时回显便于定位
    mkdir -p "${LOG_DIR}"
    local apt_log="${LOG_DIR}/apt-install-${RUN_TS}.log"
    info "apt 安装日志: ${apt_log}"
    local rc=0
    if [ "${VERBOSE}" = true ]; then
        if { ${apt_prefix} apt-get update -y \
            && ${apt_prefix} env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing_deps[@]}"; } 2>&1 | tee "${apt_log}"; then
            rc=0
        else
            rc=${PIPESTATUS[0]}
        fi
    else
        { ${apt_prefix} apt-get update -y \
            && ${apt_prefix} env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing_deps[@]}"; } > "${apt_log}" 2>&1 &
        local apt_pid=$!
        spin_on_pid "${apt_pid}" "安装依赖中"
        wait "${apt_pid}" || rc=$?
    fi
    if [ "${rc}" -ne 0 ]; then
        warn "依赖安装失败，末尾日志："
        tail -n 20 "${apt_log}" >&2
        echo "  仍无法解决可带完整日志到 https://gitcode.com/cann/hcomm/issues 反馈。" >&2
        error "apt 安装依赖失败，请检查网络 / apt 源 / 磁盘 / apt 锁后重试（完整日志: ${apt_log}）。"
    fi
}

# source CANN 的 set_env.sh（其未按 nounset/errexit 编写，临时关闭再恢复）
source_cann_env() {
    info "检测到 CANN: $1，加载环境变量"
    set +u +e
    # shellcheck disable=SC1090
    # 部分 CANN 版本的 set_env.sh 会打 banner；输出收进日志，终端只留统一 [INFO]
    source "$1" > /dev/null 2>&1
    set -eu
}

# 从 build.md 提取 CANN 推荐快照号（cann-run-mirror/software/master/<10位以上数字>）。
# 命中 echo 数字 + rc0；未固定/无匹配/文件缺失 echo 空 + rc1。
extract_cann_snapshot() {
    local buildmd="$1" snap
    [ -f "${buildmd}" ] || return 1
    snap="$(grep -oE 'cann-run-mirror/software/master/[0-9]{10,}' "${buildmd}" 2>/dev/null | head -n 1 | grep -oE '[0-9]{10,}' || true)"
    [ -n "${snap}" ] || return 1
    echo "${snap}"
}

# 按序列出可能的 CANN set_env.sh 位置（--ascend-path 与常见默认安装位置，兼容两种布局）
cann_setenv_candidates() {
    printf '%s\n' \
        "${ASCEND_PATH}/ascend-toolkit/set_env.sh" \
        "${ASCEND_PATH}/cann/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/cann/set_env.sh" \
        "/home/workspace/Ascend/ascend-toolkit/set_env.sh" \
        "/home/workspace/Ascend/cann/set_env.sh" \
        "${HOME:-}/Ascend/ascend-toolkit/set_env.sh" \
        "${HOME:-}/Ascend/cann/set_env.sh"
}

# 缺 CANN 时的安装引导：优先原样打印 build.md 的“安装CANN软件包”章节；抠不到则回退精简步骤。
print_cann_guidance() {
    local buildmd="${HCOMM_PATH}/docs/zh/build/build.md" arch section dlpage
    arch="$(normalize_arch "$(uname -m)" 2>/dev/null || echo x86_64)"
    echo -e "${RED}[ERROR]${NC} 未检测到 CANN（编译必需）" >&2
    echo "  说明: 按下方官方步骤装好 CANN，再在“同一终端”重跑本脚本（换终端需重新 source）" >&2
    if [ -f "${buildmd}" ]; then
        # 抠“### 安装CANN软件包”到下一个 ## / ### 标题之前（先判后印，避免多打下一节标题；用 ###? 规避 mawk 区间语法）
        section="$(awk '/^###? /{if(f)exit} /^###? *安装.*CANN/{f=1} f{print}' "${buildmd}" 2>/dev/null)"
        # 提取下载页 URL，用于回退步骤拼真实命令
        dlpage="$(grep -oE 'https://[^ )]*cann-run-mirror/software/master/[0-9]*/?' "${buildmd}" 2>/dev/null | head -n 1 || true)"
    fi
    if [ -n "${section:-}" ]; then
        echo "  ── 摘自 build.md（本机架构 ${arch}）──" >&2
        printf '%s\n' "${section}" >&2
        echo "  ──" >&2
    else
        echo "  按 4 步安装（本机架构 ${arch}）：" >&2
        echo "   1) wget ${dlpage:-<build.md 中 cann-run-mirror/software/master/ 下载页>}Ascend-cann-toolkit_<版本>_linux-${arch}.run" >&2
        echo "      （下载页里认准带 weekly 日期、结尾 linux-${arch}.run 的 Toolkit 包）" >&2
        echo "   2) chmod +x 上述文件 && ./上述文件 --install" >&2
        echo "   3) source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 非 root 改 ~/Ascend/..." >&2
        echo "   4) 同一终端重跑本脚本" >&2
    fi
    echo "  提示: ① 只需 Toolkit 包（驱动/固件/ops 编译不用装） ② .run 自解压，无需手动解压" >&2
    echo "        ③ CANN 已装在别处 → 用 --ascend-path <含 set_env.sh 的目录> 指定，无需重装" >&2
    echo "  更多: https://www.hiascend.com/document/redirect/CannCommunityInstWizard" >&2
}

# 确保 CANN 就绪：优先已 source 的环境变量，其次遍历常见安装位置的 set_env.sh；缺失则引导退出。
ensure_cann() {
    if [ -n "${ASCEND_HOME_PATH:-}" ] || [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
        info "检测到已配置的 CANN 环境变量"
    else
        local cand found=""
        while IFS= read -r cand; do
            if [ -f "${cand}" ]; then found="${cand}"; break; fi
        done < <(cann_setenv_candidates)
        if [ -n "${found}" ]; then
            source_cann_env "${found}"
        else
            print_cann_guidance
            exit 1
        fi
    fi

    # build 阶段用 ASCEND_HOME_PATH；若仅有 ASCEND_TOOLKIT_HOME 则由它推导，避免后续 set -u 崩溃
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        if [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
            export ASCEND_HOME_PATH="${ASCEND_TOOLKIT_HOME}"
        else
            error "CANN 已识别但未导出 ASCEND_HOME_PATH。请先 source <CANN>/ascend-toolkit/set_env.sh 后重试。"
        fi
    fi
}

# 正在克隆的目标目录；中断时由 trap 清理半成品
CURRENT_CLONE_DIR=""

# 克隆或复用一个仓库：$1=名称 $2=git地址 $3=分支/标签 $4=目标目录
clone_repo() {
    local name="$1" url="$2" ref="$3" dest="$4"
    if [ -d "${dest}" ]; then
        if git -C "${dest}" rev-parse --is-inside-work-tree > /dev/null 2>&1; then
            local cur
            # 用 branch --show-current：对"无提交(unborn)分支"与正常仓库都可靠返回分支名（rev-parse --abbrev-ref 在 unborn 上返回 "HEAD"）
            cur="$(git -C "${dest}" branch --show-current 2>/dev/null || echo unknown)"
            [ -n "${cur}" ] || cur="detached@$(git -C "${dest}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
            if [ "${cur}" != "${ref}" ]; then
                error "目录 ${dest} 当前在分支 ${cur}，与本次期望 ${ref} 不符（可能上次用了别的 profile）。请删除后重跑：rm -rf '${dest}'，或改 --workspace。"
            fi
            info "${name} 已存在（分支 ${ref}，commit $(git -C "${dest}" rev-parse --short HEAD 2>/dev/null || echo unknown)），跳过下载"
            return
        fi
        error "目录已存在但不是完整 git 仓库（疑似上次下载中断）：${dest}。请删除后重跑：rm -rf '${dest}'"
    fi
    info "克隆 ${name}（${ref}）……"
    mkdir -p "$(dirname "${dest}")" "${LOG_DIR}"
    CURRENT_CLONE_DIR="${dest}"
    # 重试最多 3 次；用 http 低速阈值作为“卡住即中止”的超时（<1KB/s 持续 60s 判失败）。
    # git 原始输出收进日志，终端只显示统一风格的活动指示。
    local attempt=1 max_attempts=3 clone_log="${LOG_DIR}/clone-${name}-${RUN_TS}.log"
    while true; do
        git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 \
            clone --branch "${ref}" "${url}" "${dest}" > "${clone_log}" 2>&1 &
        local clone_pid=$!
        spin_on_pid "${clone_pid}" "拉取 ${name} 源码中"
        if wait "${clone_pid}"; then break; fi
        rm -rf "${dest}"          # 清理半成品，避免下次被“已存在”复用
        if [ "${attempt}" -ge "${max_attempts}" ]; then
            CURRENT_CLONE_DIR=""
            warn "克隆 ${name} 失败，末尾日志："
            tail -n 15 "${clone_log}" >&2
            error "克隆 ${name} 失败（分支/标签 ${ref}，已重试 ${max_attempts} 次）。请检查网络/代理、确认 ${ref} 存在；也可手动 git clone 到 ${dest} 后重跑（脚本会自动复用）。"
        fi
        warn "克隆 ${name} 第 ${attempt}/${max_attempts} 次尝试失败，3 秒后重试……"
        attempt=$((attempt + 1))
        sleep 3
    done
    CURRENT_CLONE_DIR=""
    info "${name} 就绪（commit $(git -C "${dest}" rev-parse --short HEAD 2>/dev/null || echo unknown)）"
}

# 读已装 CANN 的安装信息（用于 best-effort 版本比对）；找不到返回空
installed_cann_version_id() {
    local f
    # 拼接所有可能含版本/日期的 info（install.info 有 version/innerversion；version.info 常含快照日期）
    for f in \
        "${ASCEND_HOME_PATH}"/ascend-toolkit/latest/*/ascend_toolkit_install.info \
        "${ASCEND_HOME_PATH}"/ascend-toolkit/latest/*/version.info \
        "${ASCEND_HOME_PATH}"/*/ascend_toolkit_install.info \
        "${ASCEND_HOME_PATH}"/*/version.info \
        "${ASCEND_HOME_PATH}"/ascend_toolkit_install.info; do
        [ -f "${f}" ] && cat "${f}" 2>/dev/null || true
    done
    return 0
}

# best-effort 软校验：按建议快照前 8 位日期在已装信息里查；不阻断。
soft_verify_cann() {
    local snapshot="$1" date8 cann_info
    if [ -z "${snapshot}" ]; then
        info "本 profile 未固定具体 CANN 快照，跳过版本校验（如编译报类型未声明再换配套 CANN）。"
        return 0
    fi
    date8="${snapshot:0:8}"
    cann_info="$(installed_cann_version_id)"
    if [ -z "${cann_info}" ]; then
        info "建议 CANN 快照: master/${snapshot}（未能读取已装 CANN 版本，跳过比对）"
        return 0
    fi
    if grep -q "${date8}" <<<"${cann_info}"; then
        info "CANN 与建议快照匹配（${date8}）"
    elif [ "${PROFILE_CUSTOM}" = true ]; then
        warn "已装 CANN 未匹配到建议快照 master/${snapshot}（自定义分支组合）；若编译报类型未声明(如 ThreadType)请换配套 CANN。"
    else
        info "未能自动确认 CANN 与建议快照(${date8})一致；非竞赛环境常见，可忽略；仅当编译报类型未声明(如 ThreadType)再换配套 CANN。"
    fi
}

# 后台进程活动指示：stdout 为终端时单行动画（每 0.8s 转圈），否则每 10s 一行心跳；均附已用时长。
# 不 wait，调用方 wait 取退出码。
spin_on_pid() {
    local pid="$1" label="$2" start=${SECONDS} elapsed
    local tty=false; color_enabled && tty=true
    local frames='|/-\' i=0 tick=0
    while kill -0 "${pid}" 2>/dev/null; do
        elapsed=$(( SECONDS - start ))
        if ${tty}; then
            printf '\r%b[INFO]%b %s %s（已用 %d 分 %02d 秒）   ' \
                "${GREEN}" "${NC}" "${label}" "${frames:i%4:1}" $((elapsed / 60)) $((elapsed % 60))
            i=$((i + 1))
            sleep 0.8
        else
            tick=$((tick + 1))
            if [ "${tick}" -ge 10 ]; then
                tick=0
                info "${label}（已用 $((elapsed / 60)) 分 $((elapsed % 60)) 秒）"
            fi
            sleep 1
        fi
    done
    ${tty} && printf '\r\033[K'
    return 0
}

# 编译 HCCL-VM 工具（对应 README §3.1 的 bash ./build.sh --full）
build_hccl_vm() {
    info "编译 HCCL-VM 工具..."
    if [ ! -d "${CHECKER_PATH}" ]; then
        error "未找到 checker 源码目录: ${CHECKER_PATH}。checker 存在于 hcomm 的竞赛/主线分支，不在发布 tag；若用 --hcomm-ref 指向了发布 tag，请改回 competition/campus-2026 或 master，或用 --checker-path 指定。"
    fi
    if [ ! -f "${CHECKER_PATH}/build.sh" ]; then
        error "未找到 ${CHECKER_PATH}/build.sh，无法编译。请确认 checker 源码完整。"
    fi
    export HCOMM_CODE_HOME="${HCOMM_PATH}"
    export HCCL_CODE_HOME="${HCCL_PATH}"
    cd "${CHECKER_PATH}" || error "无法进入 checker 目录: ${CHECKER_PATH}"
    mkdir -p "${LOG_DIR}"
    local build_log="${LOG_DIR}/build-${RUN_TS}.log"
    info "编译日志: ${build_log}"
    local rc=0
    if [ "${VERBOSE}" = true ]; then
        if bash build.sh --package-path "${ASCEND_HOME_PATH}" --hcomm-path "${HCOMM_CODE_HOME}" --hccl-path "${HCCL_CODE_HOME}" --pkg --full 2>&1 | tee "${build_log}"; then
            rc=0
        else
            rc=${PIPESTATUS[0]}
        fi
    else
        info "开始编译，预计数分钟。实时日志: tail -f ${build_log}"
        bash build.sh --package-path "${ASCEND_HOME_PATH}" --hcomm-path "${HCOMM_CODE_HOME}" --hccl-path "${HCCL_CODE_HOME}" --pkg --full > "${build_log}" 2>&1 &
        local build_pid=$!
        spin_on_pid "${build_pid}" "编译中"
        wait "${build_pid}" || rc=$?
    fi
    if [ "${rc}" -ne 0 ]; then
        warn "编译失败（日志: ${build_log}）。若 device 侧报类型未声明(如 'ThreadType' has not been declared)，通常是 CANN 与 hccl 源码不配套（CANN 构建过旧、缺 hccl 新接口头）。"
        tail -n 20 "${build_log}" >&2
        echo "  请改用与 hccl 配套的更新 CANN：" >&2
        echo "  - 版本配套见 https://gitcode.com/cann/release-management/ 对应版本 release-notes 的 cann-hccl 行" >&2
        echo "  - 或用 devcloud master 通道较新的构建：https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/" >&2
        echo "  仍无法解决可带完整日志到 https://gitcode.com/cann/hcomm/issues 反馈。" >&2
        error "编译未通过，请更换配套 CANN 后重试（完整日志: ${build_log}）。"
    fi
    # 产物落地校验：build.sh 退 0 不代表二进制一定生成
    local bin="${CHECKER_PATH}/hccl_vm_install/bin/hccl-vm"
    if [ ! -x "${bin}" ]; then
        error "编译进程已结束但未找到可执行产物: ${bin}。请查看日志 ${build_log} 后重试。"
    fi
}

# 中断(Ctrl-C/kill)时清理正在克隆的半成品目录
cleanup_on_interrupt() {
    if [ -n "${CURRENT_CLONE_DIR:-}" ] && [ -d "${CURRENT_CLONE_DIR}" ]; then
        rm -rf "${CURRENT_CLONE_DIR}"
    fi
    echo "" >&2
    warn "已中断。已完整下载的源码会保留，重跑本脚本可自动复用。"
    exit 130
}

# 主函数
main() {
    trap cleanup_on_interrupt INT TERM

    parse_args "$@"
    apply_defaults

    banner
    notice_custom_refs
    step 1 "${STEP_TOTAL}" "检查运行环境"
    check_env
    step 2 "${STEP_TOTAL}" "检查磁盘空间"
    check_disk
    step 3 "${STEP_TOTAL}" "检查/安装第三方依赖"
    install_deps
    step 4 "${STEP_TOTAL}" "拉取 hcomm 源码（${HCOMM_REF}）"
    clone_repo "hcomm" "https://gitcode.com/cann/hcomm.git" "${HCOMM_REF}" "${HCOMM_PATH}"
    step 5 "${STEP_TOTAL}" "检测 CANN 并校验配套"
    local snapshot=""
    snapshot="$(extract_cann_snapshot "${HCOMM_PATH}/docs/zh/build/build.md" || true)"
    ensure_cann
    soft_verify_cann "${snapshot}"
    step 6 "${STEP_TOTAL}" "拉取 hccl 源码（${HCCL_REF}）"
    clone_repo "hccl" "https://gitcode.com/cann/hccl.git" "${HCCL_REF}" "${HCCL_PATH}"
    step 7 "${STEP_TOTAL}" "编译 HCCL-VM"
    build_hccl_vm

    success_summary
}

# 仅当脚本被直接执行时运行主函数；被 source（单测）时不执行
if [[ "${BASH_SOURCE[0]:-$0}" == "${0}" ]]; then
    main "$@"
fi
