/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_addr_info.h"
#include <stdint.h>
#include <stdlib.h>
#include <syslog.h>
#include <sys/stat.h>
#include "hal.h"
#include "topo.h"
#include "topo_addr_info_perf.h"
#include "product_card.h"
#include "product_server.h"
#include "product_pod.h"

#define MAX_DUMP_FILE_LEN (256)
#define DEFAULT_RANKINFO_FILE_PATH "/etc/hccl_rootinfo.json"
#define DEFAULT_RANKINFO_SIZE (4096)

typedef int (*GetSizeFuncType)(size_t* size);

typedef int (*GetRootinfoFuncType)(int npu_id, unsigned int mainboard_id, void* buf, size_t* len);

typedef struct {
    uint32_t mainboard_id;
    GetSizeFuncType get_size_func;
} GetSizeFuncTable;

typedef struct {
    uint32_t mainboard_id;
    GetRootinfoFuncType get_rootinfo_func;
} GetRootinfoFuncTable;

static GetSizeFuncTable g_get_size_func_table[] = {
    {MAIN_BOARD_ID_CARD_NOMESH, GetCardRankInfoLen},
    {MAIN_BOARD_ID_CARD_2PMESH, GetCardRankInfoLen},
    {MAIN_BOARD_ID_CARD_4PMESH, GetCardRankInfoLen},
    {MAIN_BOARD_ID_SERVER_8PMESH, ServerGetRootinfoLen},
    {MAIN_BOARD_ID_SERVER_TYPE1, ServerGetRootinfoLen},
    {MAIN_BOARD_ID_SERVER_8PMESH_UBOE, ServerGetRootinfoLen},
    {MAIN_BOARD_ID_SERVER_8PMESH_NOSP, ServerGetRootinfoLen},
    {MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE, ServerGetRootinfoLen},
    {MAIN_BOARD_ID_SERVER_UBX, ServerGetRootinfoLen},
    {MAIN_BOARD_ID_POD, PodGetRootinfoLen},
    {MAIN_BOARD_ID_POD_2D, PodGetRootinfoLen},
};

static GetRootinfoFuncTable g_get_rootinfo_func_table[] = {
    {MAIN_BOARD_ID_CARD_NOMESH, GetCardRankInfo},
    {MAIN_BOARD_ID_CARD_2PMESH, GetCardRankInfo},
    {MAIN_BOARD_ID_CARD_4PMESH, GetCardRankInfo},
    {MAIN_BOARD_ID_SERVER_8PMESH, ServerGetRootinfo},
    {MAIN_BOARD_ID_SERVER_TYPE1, ServerGetRootinfo},
    {MAIN_BOARD_ID_SERVER_8PMESH_UBOE, ServerGetRootinfo},
    {MAIN_BOARD_ID_SERVER_8PMESH_NOSP, ServerGetRootinfo},
    {MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE, ServerGetRootinfo},
    {MAIN_BOARD_ID_SERVER_UBX, ServerGetRootinfo},
    {MAIN_BOARD_ID_POD, PodGetRootinfo},
    {MAIN_BOARD_ID_POD_2D, PodGetRootinfo},
};

int TopoAddrInfoGetSize(int phyId, size_t* size)
{
    if (size == NULL) {
        return -1;
    }

    struct stat st;
    if (stat(DEFAULT_RANKINFO_FILE_PATH, &st) == 0) {
        (*size) = st.st_size;
        return 0;
    }

    uint32_t mainboard_id = 0;
    int ret = hal_get_mainboard_id(phyId, &mainboard_id);
    if (ret != 0) {
        return ret;
    }

    // 从g_get_size_func_table中查找对应的函数
    for (size_t i = 0; i < sizeof(g_get_size_func_table) / sizeof(g_get_size_func_table[0]); i++) {
        if (g_get_size_func_table[i].mainboard_id == mainboard_id) {
            return g_get_size_func_table[i].get_size_func(size);
        }
    }
    (*size) = DEFAULT_RANKINFO_SIZE;
    return 0;
}

static int PassThroughTopoFilePath(char* filePath, size_t bufSize)
{
    return GetTopoFilePathFromFile("/etc/hccl_rootinfo.json", filePath, bufSize);
}

/**
 *  获取拓扑文件路径
 */
int TopoAddrInfoGetTopoFilePath(int phyId, char* filePath, size_t bufSize)
{
    if (filePath == NULL) {
        return -1;
    }
    // 优先从/etc/hccl_rootinfo.json中读取
    int ret = PassThroughTopoFilePath(filePath, bufSize);
    if (ret == 0) {
        return ret;
    }
    uint32_t mainboard_id = 0;
    ret = hal_get_mainboard_id(phyId, &mainboard_id);
    if (ret != 0) {
        return ret;
    }
    struct dcmi_spod_info spod_info;
    ret = hal_get_spod_info(phyId, &spod_info);
    if (ret != 0) {
        return TopoGetFilePath(mainboard_id, TOPO_TYPE_IGNORE, filePath, bufSize);
    }
    return TopoGetFilePath(mainboard_id, spod_info.super_pod_type, filePath, bufSize);
}

static int PassThrough(char* rankInfo, size_t* bufSize)
{
    FILE* fp = fopen(DEFAULT_RANKINFO_FILE_PATH, "r");
    if (fp == NULL) {
        return -1;
    }
    struct stat stat;
    fstat(fileno(fp), &stat);
    if ((size_t)stat.st_size > (*bufSize)) {
        fclose(fp);
        return -1;
    }
    int ret = fread(rankInfo, 1, stat.st_size, fp);
    if (ret < 0) {
        fclose(fp);
        return -1;
    }
    *bufSize = (size_t)stat.st_size;
    return 0;
}

int TopoAddrInfoGet(int phyId, char* rankInfo, size_t* bufSize)
{
    TopoLogInit(); /* 懒初始化日志，全部下游共用 */
    TOPO_PERF_BEGIN(TopoAddrInfoGet);
    if (rankInfo == NULL || bufSize == NULL) {
        TOPO_PERF_END(TopoAddrInfoGet);
        return -1;
    }
    // 优先读取/etc/hccl_rootinfo.json中内容
    if (PassThrough(rankInfo, bufSize) == 0) {
        TOPO_PERF_END(TopoAddrInfoGet);
        return 0;
    }
    // 若/etc/hccl_rootinfo.json中无内容，根据mainboard_id生成rootinfo
    uint32_t mainboard_id = 0;
    int ret = hal_get_mainboard_id(phyId, &mainboard_id);
    if (ret != 0) {
        TOPO_PERF_END(TopoAddrInfoGet);
        return ret;
    }

    ret = -1;
    for (size_t i = 0; i < sizeof(g_get_rootinfo_func_table) / sizeof(GetRootinfoFuncTable); ++i) {
        if (g_get_rootinfo_func_table[i].mainboard_id == mainboard_id) {
            ret = g_get_rootinfo_func_table[i].get_rootinfo_func(phyId, mainboard_id, rankInfo, bufSize);
            break;
        }
    }
    TOPO_PERF_END(TopoAddrInfoGet);
    return ret;
}
