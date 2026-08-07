/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*
* The code snippet comes from Cann project.
*
* Copyright 2012-2019 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
 */

#ifndef HWTS_DRV_TASK_INFO_H
#define HWTS_DRV_TASK_INFO_H

struct hwts_ts_kernel {
    pid_t pid;
    unsigned short kernel_type : 8;
    unsigned short batchMode : 1; // default 0
    unsigned short satMode : 1;
    unsigned short rspMode : 1;
    unsigned short resv : 5;
    unsigned short streamID;
    unsigned long long kernelName;
    unsigned long long kernelSo;
    unsigned long long paramBase;
    unsigned long long l2VaddrBase;
    unsigned long long l2Ctrl;
    unsigned short blockId;
    unsigned short blockNum;
    unsigned int l2InMain;
    unsigned long long taskID;
};

/* Not allow hwts_ts_task to be parameter passing in aicpufw and drv */
struct hwts_ts_task {
    unsigned int mailbox_id;
    volatile unsigned long long serial_no;
    struct hwts_ts_kernel kernel_info;
};

typedef enum hwts_task_status {
    TASK_SUCC = 0,
    TASK_FAIL = 1,
    TASK_OVERFLOW = 2,
    TASK_STATUS_MAX,
} HWTS_TASK_STATUS;

#define HWTS_RESPONSE_RSV 3
struct hwts_response {
    unsigned int result;
    unsigned int mailbox_id;
    unsigned long long serial_no;
    unsigned int status;
    int rsv[HWTS_RESPONSE_RSV];
    char* msg;
    int len;
};

#endif
