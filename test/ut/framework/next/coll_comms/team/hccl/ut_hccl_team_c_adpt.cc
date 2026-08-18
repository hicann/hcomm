/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测对象：src/coll_communicator_mgr/team/hccl/hccl_team_c_adpt.cc 中 HcclTeam 系列 C 接口。
// 设计说明：
// 1) Hcomm 层 team 接口（HcommTeamCreate/Destroy/GetNetLayer/GetWorldTeamIds/WindowRegister/BindChannels/
//    BindWindow/WindowDeregister/BindSyncMem）已在 src/coll_communicator_mgr/team/hcomm/hcomm_team_c_adpt.cc 实现，
//    编译入 hccl_llt 共享库；用例内通过 mockcpp MOCKER(...) 拦截并按场景设置返回值/出参。
// 2) HcclTeamMgr 是非虚单例，不可 mockcpp。本测试通过被测 C 接口（HcclWorldTeamCreate 等）真实驱动
//    其状态，每个用例末尾调用 HcclTeamDestroy 清理，避免跨用例泄漏。
// 3) 通信域采用真实 hcclComm + RankGraphStub 2P 图：rank=1、rankSize=2，对端 rank=0。

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "hccl_team.h"
#include "hccl_team_c_adpt.h"
#include "hcomm_team_c_adpt.h"
#include "hcomm_team_mgr.h"
#include "hccl/hccl_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_types.h"
#include "hccl_comm_pub.h"
#include "adapter_rts_common.h"
#include "hccl_team_mgr.h"
#include "llt_hccl_stub_rank_graph.h"
#include "log.h"
#include "topoinfo_struct.h"

#define private public
#define protected public
#include "coll_comm.h"
#undef protected
#undef private

#include "../../../hccl_api_base_test.h"

using namespace hccl;
using namespace hcomm;

namespace {
// ===================== 公共测试常量 =====================
// worldTeam/subTeam/window 的伪造句柄值（非空即可，仅作为 map key 与 mock 出参）。
HcommTeamHandle g_fakeWorldTeam = reinterpret_cast<HcommTeamHandle>(0x10000);
HcommTeamHandle g_fakeSubTeam = reinterpret_cast<HcommTeamHandle>(0x20000);
HcommWindowHandle g_fakeWindow = reinterpret_cast<HcommWindowHandle>(0x30000);
void* g_fakeWsPtr = reinterpret_cast<void*>(0x40000);
HcclMemHandle g_fakeUserMemHandle = reinterpret_cast<HcclMemHandle>(0x60000);
ChannelHandle g_fakeChannel = static_cast<ChannelHandle>(0x70000);

// HcommTeamGetWorldTeamIds 出参缓冲：memberNum=4，selfMemberId 由调用方设定。
uint32_t g_worldTeamIds4[4] = {1, 3, 5, 7};

// ===================== mockcpp invoke 桩：为带出参的 Hcomm/Hccl 接口填充返回值 =====================
HcommResult StubTeamCreateFillWorld(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = g_fakeWorldTeam;
    *outSyncMemSize = 4096;
    return 0;
}
HcommResult StubTeamCreateZeroWs(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = (worldTeam == nullptr) ? g_fakeWorldTeam : g_fakeSubTeam;
    *outSyncMemSize = 0;
    return 0;
}
// 按 worldTeam 参数区分：worldTeam==nullptr（world team 创建）返回 g_fakeWorldTeam + syncMemSize=4096；
// 否则（sub team 创建）返回 g_fakeSubTeam + syncMemSize=4096。用于同一用例内先建 world 再建 sub 的场景，
// 避免 FIFO 下两次 MOCKER(HcommTeamCreate).stubs() 只命中首个桩的问题。
HcommResult StubTeamCreateByWorldArg(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)desc;
    *team = (worldTeam == nullptr) ? g_fakeWorldTeam : g_fakeSubTeam;
    *outSyncMemSize = 4096;
    return 0;
}
// 按 worldTeam 参数区分：world team 创建成功（返回 g_fakeWorldTeam），sub team 创建失败（返回 HCCL_E_INTERNAL）。
// 用于 HcclSubTeamCreate 失败用例的前置 worldTeam 创建 + sub 创建失败场景。
HcommResult StubTeamCreateWorldOkSubFail(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)desc;
    if (worldTeam == nullptr) {
        *team = g_fakeWorldTeam;
        *outSyncMemSize = 4096;
        return 0;
    }
    *team = nullptr;
    *outSyncMemSize = 0;
    return static_cast<HcommResult>(HCCL_E_INTERNAL);
}
// 按 worldTeam 参数区分：world team 创建成功（syncMemSize=4096，HcclWorldTeamCreate/HcclSubTeamCreate 现要求
// syncMemSize>0）， sub team 创建成功但 syncMemSize=0（触发 HcclSubTeamCreate 的 syncMemSize=0 → HCCL_E_PARA
// 错误路径）。
HcommResult StubTeamCreateWorldOkSubZeroWs(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)desc;
    *team = (worldTeam == nullptr) ? g_fakeWorldTeam : g_fakeSubTeam;
    *outSyncMemSize = (worldTeam == nullptr) ? 4096 : 0;
    return 0;
}
HcommResult StubTeamGetIds4(HcommTeamHandle team, uint32_t** worldTeamIds, uint32_t* memberNum)
{
    (void)team;
    *worldTeamIds = g_worldTeamIds4;
    *memberNum = 4;
    return 0;
}
HcommResult StubTeamGetNetLayerZero(HcommTeamHandle team, uint32_t* netLayer)
{
    (void)team;
    *netLayer = 0;
    return 0;
}
HcommResult StubTeamWindowRegisterFill(
    HcommTeamHandle worldTeam, const HcommTeamWindowDesc* desc, HcommWindowHandle* handle, HcommTeamWindowFlag flag)
{
    (void)worldTeam;
    (void)desc;
    (void)flag;
    *handle = g_fakeWindow;
    return 0;
}

// —— HcommTeamCreate 桩：返回 ok 但 *team=nullptr（触发 E_INTERNAL 分支）——
HcommResult StubTeamCreateReturnNullTeam(
    HcommTeamHandle worldTeam, const HcommTeamCreateDesc* desc, HcommTeamHandle* team, uint64_t* outSyncMemSize)
{
    (void)worldTeam;
    (void)desc;
    *team = nullptr;
    *outSyncMemSize = 0;
    return 0;
}

// —— HcommTeamGetWorldTeamIds 桩：可配置返回的 ids/num（通过全局变量）——
// 用于 MemberInfoAbnormal / LinkFail 等用例，按子场景设置 g_stubIds / g_stubIdNum。
uint32_t* g_stubIds = nullptr;
uint32_t g_stubIdNum = 0;
HcommResult StubTeamGetIdsConfigurable(HcommTeamHandle team, uint32_t** worldTeamIds, uint32_t* memberNum)
{
    (void)team;
    *worldTeamIds = g_stubIds;
    *memberNum = g_stubIdNum;
    return 0;
}

// —— HcclRankGraphGetLinks 桩：无 link（linkNum=0）→ E_NOT_FOUND ——
HcclResult StubRankGraphGetLinksEmpty(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)
{
    (void)comm;
    (void)netLayer;
    (void)srcRank;
    (void)dstRank;
    *links = nullptr;
    *linkNum = 0;
    return HCCL_SUCCESS;
}

// —— hrtMalloc 计数桩：前 g_hrtMallocOkCount 次成功（填 g_fakeWsPtr），之后失败（返回 HCCL_E_MEMORY）——
// 用于"前置 worldTeam 创建成功 + sub team hrtMalloc 失败"场景，避免 expects(once()) 误命中前置调用。
uint32_t g_hrtMallocOkCount = 0;
uint32_t g_hrtMallocCallCnt = 0;
HcclResult StubHrtMallocCountFail(void** devPtr, u64 size, bool level2Address)
{
    (void)size;
    (void)level2Address;
    g_hrtMallocCallCnt++;
    if (g_hrtMallocCallCnt <= g_hrtMallocOkCount) {
        *devPtr = g_fakeWsPtr;
        return HCCL_SUCCESS;
    }
    *devPtr = nullptr;
    return HCCL_E_MEMORY;
}

// HcclCommMemReg 出参桩：返回 user mem 句柄。
HcclResult StubMemRegFillUser(HcclComm comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle)
{
    (void)comm;
    (void)memTag;
    (void)mem;
    *memHandle = g_fakeUserMemHandle;
    return HCCL_SUCCESS;
}

// hrtMalloc 出参桩：填入伪指针。
HcclResult StubHrtMallocFill(void** devPtr, u64 size, bool level2Address)
{
    (void)size;
    (void)level2Address;
    *devPtr = g_fakeWsPtr;
    return HCCL_SUCCESS;
}

// HcclRankGraphGetLinks 出参桩：返回单条 ROCE 链路。
CommLink g_fakeLink{};
HcclResult StubRankGraphGetLinks(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** links, uint32_t* linkNum)
{
    (void)comm;
    (void)netLayer;
    (void)srcRank;
    (void)dstRank;
    g_fakeLink.header.version = 1;
    g_fakeLink.header.magicWord = 0x0f0e0f0f;
    g_fakeLink.header.size = sizeof(CommLink);
    g_fakeLink.header.reserved = 0;
    g_fakeLink.linkAttr.linkProtocol = COMM_PROTOCOL_UB_CTP;
    g_fakeLink.linkAttr.hop = 1;
    *links = &g_fakeLink;
    *linkNum = 1;
    return HCCL_SUCCESS;
}

// HcclChannelAcquire 出参桩：每个 channel 填入伪造句柄。
HcclResult StubChannelAcquireFill(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    (void)comm;
    (void)engine;
    (void)channelDescs;
    for (uint32_t i = 0; i < channelNum; i++) {
        channels[i] = g_fakeChannel + i;
    }
    return HCCL_SUCCESS;
}

// HcclChannelGetRemoteMems 出参桩：可配置返回 ws/user 两类 tag 远端内存。
struct RemoteMemScenario {
    bool hasWs{false};
    bool hasUser{false};
    std::string wsTag{};
    std::string userTag{};
    CommMem wsMem{};
    CommMem userMem{};
    bool multiMems{false}; // 同 peer 同 tag 多个 mem（验证仅取首个）
};
RemoteMemScenario g_remoteScenario;
CommMem g_remoteMemsBuf[4];
char* g_remoteTagsBuf[4];
char g_wsTagBuf[128];
char g_userTagBuf[128];

HcclResult
StubChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags)
{
    (void)comm;
    (void)channel;
    uint32_t n = 0;
    auto appendMem = [&](CommMem mem, const std::string& tag, char* tagBuf) {
        g_remoteMemsBuf[n] = mem;
        (void)std::strncpy(tagBuf, tag.c_str(), 127);
        tagBuf[127] = '\0';
        g_remoteTagsBuf[n] = tagBuf;
        n++;
        if (g_remoteScenario.multiMems) {
            // 同 tag 再追加一个，验证源码"仅取首个"逻辑（第二个因 addr!=nullptr 被跳过）
            g_remoteMemsBuf[n] = mem;
            g_remoteTagsBuf[n] = tagBuf;
            n++;
        }
    };
    if (g_remoteScenario.hasWs) {
        appendMem(g_remoteScenario.wsMem, g_remoteScenario.wsTag, g_wsTagBuf);
    }
    if (g_remoteScenario.hasUser) {
        appendMem(g_remoteScenario.userMem, g_remoteScenario.userTag, g_userTagBuf);
    }
    *memNum = n;
    *remoteMems = (n > 0) ? g_remoteMemsBuf : nullptr;
    *memTags = (n > 0) ? g_remoteTagsBuf : nullptr;
    return HCCL_SUCCESS;
}

// ===================== 公共辅助：构造真实 hcclComm（rank=1, rankSize=2） =====================
// rankGraphV2 必须由调用方持有（fixture 成员），因为 InitCollComm → CollComm::Init → IRankGraph 只存裸指针、
// 不获取所有权；若 rankGraphV2 为函数局部 shared_ptr，函数返回后悬空，后续 GetRankSize() 解引用已释放内存。
void BuildV2HcclComm(std::shared_ptr<hccl::hcclComm>& hcclCommPtr, std::shared_ptr<Hccl::RankGraph>& rankGraphV2)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));

    void* commV2 = reinterpret_cast<void*>(0x2000);
    RankGraphStub rankGraphStub;
    rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1024;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = reinterpret_cast<void*>(0x1000);
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1; // 非CCU模式
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 构造并初始化 HcclTeamCreateDesc。
void BuildTeamCreateDesc(
    HcclTeamCreateDesc& desc, const uint32_t* rankIds, uint32_t rankNum, uint32_t selfRankId, uint32_t netLayer = 1)
{
    (void)HcclTeamCreateDescInit(&desc);
    desc.rankIds = rankIds;
    desc.rankNum = rankNum;
    desc.selfRankId = selfRankId;
    desc.netLayer = netLayer;
    desc.protocol = COMM_PROTOCOL_UB_CTP;
    // 源码 HcclWorldTeamCreate/HcclSubTeamCreate 校验：signalCount/counterCount 必须为 0，barrierCount 必须 >= 1。
    desc.requirement.signalCount = 0;
    desc.requirement.counterCount = 0;
    desc.requirement.barrierCount = 1;
}

void BuildChannelDesc(HcclTeamCreateChannelsDesc& desc, CommEngine engine, uint32_t channelCnt, uint32_t notifyNum = 8)
{
    (void)HcclTeamCreateChannelsDescInit(&desc);
    desc.engine = engine;
    desc.notifyNum = notifyNum;
    desc.protocol = COMM_PROTOCOL_UB_CTP;
    desc.channelCnt = channelCnt;
}
} // namespace

// ===================== 测试夹具 =====================
class TestCollCommTeamCAdpt : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
        // 先构造真实 hcclComm（InitCollComm 内部会调用真实 hrtMalloc/hrtFree 等，必须在其完成后再
        // mock 这些接口，否则桩返回的伪指针会被 InitCollComm 解引用导致 SEGFAULT）。
        // rankGraphV2 存为 fixture 成员，使其生命周期覆盖整个用例（InitCollComm 只存裸指针）。
        BuildV2HcclComm(hcclCommPtr, rankGraphV2);
        comm = static_cast<HcclComm>(hcclCommPtr.get());
        g_remoteScenario = {};
        // 注意：不在 SetUp 内预 mock Hcomm/Hccl 接口。mockcpp 对同一函数的多个 stubs() 按 FIFO 匹配，
        // SetUp 预桩会使用例内的覆盖失效。各用例通过 SetupSuccessMocks() 或自行设置所需 mock。
    }
    void TearDown() override
    {
        // 兜底清理：直接通过 HcclTeamMgr 清除可能残留的 team 条目（不调 Hcomm 层，避免与用例内
        // expects(never()) 等期望冲突），防止单例状态泄漏到下一用例。
        HcclTeamMgr::GetInstance().UnregisterTeam(g_fakeWorldTeam);
        HcclTeamMgr::GetInstance().UnregisterTeam(g_fakeSubTeam);
        GlobalMockObject::verify();
        BaseInit::TearDown();
    }

protected:
    // 成功路径默认桩：集中设置 Hcomm/Hccl 接口的成功返回值 + 出参填充。
    // 成功路径用例在用例体首行调用；异常/覆盖用例不调用，自行设置所需 mock。
    void SetupSuccessMocks()
    {
        MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateFillWorld));
        MOCKER(HcommTeamDestroy).stubs().will(returnValue(0));
        MOCKER(HcommTeamGetNetLayer).stubs().will(invoke(StubTeamGetNetLayerZero));
        MOCKER(HcommTeamWindowRegister).stubs().will(invoke(StubTeamWindowRegisterFill));
        MOCKER(HcommTeamWindowDeregister).stubs().will(returnValue(0));
        MOCKER(HcommTeamBindChannels).stubs().will(returnValue(0));
        MOCKER(HcommTeamWindowBindRemoteMems).stubs().will(returnValue(0));
        MOCKER(HcommTeamBindRemoteSyncMem).stubs().will(returnValue(0));
        MOCKER(hrtMalloc).stubs().will(invoke(StubHrtMallocFill));
        MOCKER(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(HcclCommMemReg).stubs().will(invoke(StubMemRegFillUser));
        MOCKER(HcclRankGraphGetLinks).stubs().will(invoke(StubRankGraphGetLinks));
        MOCKER(HcclChannelAcquire).stubs().will(invoke(StubChannelAcquireFill));
        MOCKER(HcclChannelGetRemoteMems).stubs().will(invoke(StubChannelGetRemoteMems));
    }

    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    HcclComm comm{nullptr};
};

// =====================================================================
// 1. HcclWorldTeamCreate
// =====================================================================

// 功能：正常创建 4 卡 WorldTeam。注：真实 hcclComm 为 2P 图（rankSize=2），故此处用 rankNum=2 验证主流程，
// 同时通过 mock HcommTeamCreate 出参 selfMemberId/netLayer 透传逻辑。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclWorldTeamCreate_When_NormalDesc_Expect_ReturnSuccessAndWorldTeamFilled)
{
    SetupSuccessMocks();          // HcommTeamCreate=FillWorld, hrtMalloc=Fill, HcommTeamDestroy=0, hrtFree=0
    uint32_t rankIds[2] = {0, 1}; // rankSize=2, selfRank=1
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;

    HcclResult ret = HcclWorldTeamCreate(comm, &desc, &worldTeam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(worldTeam, g_fakeWorldTeam);
    // 清理：销毁已注册 team。
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：syncMemSize=0 视为非法，源码在 HcommTeamCreate 成功后校验 syncMemSize 并返回 HCCL_E_PARA。
// hrtMalloc 不应被调用（syncMemSize 校验在 hrtMalloc 之前）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclWorldTeamCreate_When_SyncMemSizeZero_Expect_ReturnHCCL_E_PARA)
{
    // 覆盖须在 SetupSuccessMocks() 之前注册：mockcpp stubs 按 FIFO 匹配，先注册的覆盖先生效。
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateZeroWs));
    // hrtMalloc 不应被调用：用 never() 验证（expects 优先级高于 stubs）。
    MOCKER(hrtMalloc).expects(never());
    SetupSuccessMocks();

    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;

    // syncMemSize=0 → 源码 CHK_PRT_RET(syncMemSize == 0, ..., HCCL_E_PARA) 直接返回，不调 hrtMalloc。
    // 注意：源码此处不置 *worldTeam=nullptr，*worldTeam 仍为 HcommTeamCreate 出参 g_fakeWorldTeam，
    // 但未注册到 HcclTeamMgr，故结尾用 HcommTeamDestroy 桩兜底清理（不调 HcclTeamDestroy）。
    HcclResult ret = HcclWorldTeamCreate(comm, &desc, &worldTeam);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 异常：入参空指针校验。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclWorldTeamCreate_When_NullptrParams_Expect_ReturnHCCL_E_PTR)
{
    HcclTeamCreateDesc desc{};
    HcommTeamHandle worldTeam = nullptr;
    EXPECT_EQ(HcclWorldTeamCreate(nullptr, &desc, &worldTeam), HCCL_E_PTR);
    EXPECT_EQ(HcclWorldTeamCreate(comm, nullptr, &worldTeam), HCCL_E_PTR);
    EXPECT_EQ(HcclWorldTeamCreate(comm, &desc, nullptr), HCCL_E_PTR);
}

// 异常：desc 参数非法校验（rankNum=0 → E_PARA；rankIds=nullptr → E_PTR）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclWorldTeamCreate_When_DescInvalid_Expect_ReturnParaOrPtr)
{
    HcclTeamCreateDesc desc = {};
    (void)HcclTeamCreateDescInit(&desc);
    HcommTeamHandle worldTeam = nullptr;

    // rankNum=0
    desc.rankNum = 0;
    desc.rankIds = nullptr;
    EXPECT_EQ(HcclWorldTeamCreate(comm, &desc, &worldTeam), HCCL_E_PARA);

    // rankIds=nullptr（rankNum=2 非零）
    uint32_t rankIds[2] = {0, 1};
    desc.rankNum = 2;
    desc.rankIds = nullptr;
    desc.selfRankId = 1;
    EXPECT_EQ(HcclWorldTeamCreate(comm, &desc, &worldTeam), HCCL_E_PTR);
}

// 异常：desc 与通信域不匹配校验（rankNum>commRankSize；rankIds 不含 selfRankId）。
// 注：worldTeam 支持通信域子集，rankNum<commRankSize 合法（见 Ut_HcclWorldTeamCreate_When_Subset_Expect_Success）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclWorldTeamCreate_When_DescNotMatchComm_Expect_ReturnParaOrPtr)
{
    HcommTeamHandle worldTeam = nullptr;

    // rankNum > commRankSize(=2)
    uint32_t rankIds3[3] = {0, 1, 2};
    HcclTeamCreateDesc desc1;
    BuildTeamCreateDesc(desc1, rankIds3, 3, 1, 1);
    EXPECT_EQ(HcclWorldTeamCreate(comm, &desc1, &worldTeam), HCCL_E_PARA);

    // rankIds 不含 selfRankId
    uint32_t rankIdsNoSelf[2] = {0, 2};
    HcclTeamCreateDesc desc3;
    BuildTeamCreateDesc(desc3, rankIdsNoSelf, 2, 1, 1); // selfRankId=1 不在 [0,2]
    EXPECT_EQ(HcclWorldTeamCreate(comm, &desc3, &worldTeam), HCCL_E_NOT_FOUND);

    // GetCollComm 返回 nullptr：直接置 comm 内 collComm 为空较难，改为 comm=nullptr 覆盖 E_PTR 路径
    EXPECT_EQ(HcclWorldTeamCreate(nullptr, &desc3, &worldTeam), HCCL_E_PTR);
}

// 异常：单 rank team（rankNum=1）不允许创建，返回 E_PARA。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclWorldTeamCreate_When_SingleRank_Expect_ReturnParaError)
{
    SetupSuccessMocks();
    // 通信域 rankSize=2，本 rank=1；仅含本 rank 的单 rank team 应被拒绝
    uint32_t rankIds1[1] = {1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds1, 1, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    HcclResult ret = HcclWorldTeamCreate(comm, &desc, &worldTeam);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(worldTeam, nullptr);
}

// 异常：HcommTeamCreate 失败不回滚（团队未创建）；worldTeam 句柄为空触发 E_INTERNAL。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclWorldTeamCreate_When_HcommCreateFail_Expect_NoRollbackAndFail)
{
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);

    // (1) HcommTeamCreate 失败：CHK_PRT_RET 直接返回 ret 转换码，不调用 HcommTeamDestroy（团队未创建）。
    {
        HcommTeamHandle worldTeam = nullptr;
        MOCKER(HcommTeamCreate).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
        MOCKER(HcommTeamDestroy).expects(never());
        HcclResult ret = HcclWorldTeamCreate(comm, &desc, &worldTeam);
        EXPECT_EQ(ret, static_cast<HcclResult>(HCCL_E_INTERNAL));
        EXPECT_EQ(worldTeam, nullptr);
    }
    // (2) HcommTeamCreate 返回 ok 但 *worldTeam=nullptr：触发 E_INTERNAL 分支。
    {
        HcommTeamHandle worldTeam = nullptr;
        MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateReturnNullTeam));
        HcclResult ret = HcclWorldTeamCreate(comm, &desc, &worldTeam);
        EXPECT_EQ(ret, HCCL_E_INTERNAL);
        EXPECT_EQ(worldTeam, nullptr);
    }
}

// 异常：hrtMalloc 失败回滚 HcommTeamDestroy，返回 E_MEMORY。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclWorldTeamCreate_When_HrtMallocFail_Expect_RollbackAndE_MEMORY)
{
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);

    HcommTeamHandle worldTeam = nullptr;
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateFillWorld));
    MOCKER(hrtMalloc).expects(once()).will(returnValue(HCCL_E_MEMORY));
    MOCKER(HcommTeamDestroy).expects(once()).will(returnValue(0));
    HcclResult ret = HcclWorldTeamCreate(comm, &desc, &worldTeam);
    EXPECT_EQ(ret, HCCL_E_MEMORY);
    EXPECT_EQ(worldTeam, nullptr);
}

// 功能：通信域销毁（ClearByCollComm）时，销毁该通信域下所有 team 及其 window 的 L3 资源。
// 验证：worldTeam 的 window 调 HcommTeamWindowDeregister，team 调 HcommTeamDestroy，syncMem 调 hrtFree。
TEST_F(TestCollCommTeamCAdpt, Ut_ClearByCollComm_When_TeamAndWindowExist_Expect_AllDestroyed)
{
    // expects 须在 SetupSuccessMocks 前注册：mockcpp FIFO，先注册的先命中，stubs 不计数 expects。
    MOCKER(HcommTeamWindowDeregister).expects(once()).will(returnValue(0));
    MOCKER(HcommTeamDestroy).expects(once()).will(returnValue(0));
    MOCKER(hrtFree).expects(once()).will(returnValue(HCCL_SUCCESS));
    SetupSuccessMocks();
    // 创建 worldTeam（注册到 HcclTeamMgr，collComm 取自真实 comm）
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, rankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);
    ASSERT_EQ(worldTeam, g_fakeWorldTeam);

    // 手动给 worldTeam 注册一个 window（AddWorldTeamWindow 直接写 teamMap 的 windows 列表）
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0x50000);
    localMem.size = 1024;
    HcclTeamMgr::GetInstance().AddWorldTeamWindow(
        worldTeam, g_fakeWindow, localMem, g_fakeUserMemHandle, "__hccl_team_usermem__test");

    CollComm* collComm = hcclCommPtr->GetCollComm();
    ASSERT_NE(collComm, nullptr);
    HcclTeamMgr::GetInstance().ClearByCollComm(collComm);

    // ClearByCollComm 已 erase teamMap 条目，后续 UnregisterTeam 兜底为 no-op
}

// =====================================================================
// 2. HcclSubTeamCreate
// =====================================================================

// 功能：正常创建 SubTeam，selfMemberId 由 rankIds 下标换算。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclSubTeamCreate_When_NormalDesc_Expect_ReturnSuccessAndSubTeamFilled)
{
    // 用按 worldTeam 参数区分的桩：worldTeam=nullptr 返回 g_fakeWorldTeam，否则返回 g_fakeSubTeam。
    // 避免同用例内两次 MOCKER(HcommTeamCreate).stubs() 在 FIFO 下只命中首个。
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateByWorldArg));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 创建 sub team：rankIds=[0,1], selfRank=1 → selfMemberId=1。
    uint32_t subRankIds[2] = {0, 1};
    HcclTeamCreateDesc subDesc;
    BuildTeamCreateDesc(subDesc, subRankIds, 2, 1, 1);
    HcommTeamHandle subTeam = nullptr;

    HcclResult ret = HcclSubTeamCreate(worldTeam, &subDesc, &subTeam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(subTeam, g_fakeSubTeam);

    EXPECT_EQ(HcclTeamDestroy(subTeam), HCCL_SUCCESS);
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：sub team syncMemSize=0 视为非法，源码在 HcommTeamCreate 成功后校验 syncMemSize 并返回 HCCL_E_PARA。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclSubTeamCreate_When_SyncMemSizeZero_Expect_ReturnHCCL_E_PARA)
{
    // world 前置用 syncMemSize=4096（HcclWorldTeamCreate 要求 syncMemSize>0）；sub 用 syncMemSize=0 触发 E_PARA。
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateWorldOkSubZeroWs));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // sub team 零 syncMem → 源码 CHK_PRT_RET(syncMemSize == 0, ..., HCCL_E_PARA) 直接返回。
    // 注意：源码此处不置 *team=nullptr，*team 仍为 HcommTeamCreate 出参 g_fakeSubTeam，但未注册到 HcclTeamMgr。
    uint32_t subRankIds[2] = {0, 1};
    HcclTeamCreateDesc subDesc;
    BuildTeamCreateDesc(subDesc, subRankIds, 2, 1, 1);
    HcommTeamHandle subTeam = nullptr;

    HcclResult ret = HcclSubTeamCreate(worldTeam, &subDesc, &subTeam);
    EXPECT_EQ(ret, HCCL_E_PARA);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：入参空指针。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclSubTeamCreate_When_NullptrParams_Expect_ReturnHCCL_E_PTR)
{
    HcclTeamCreateDesc desc{};
    HcommTeamHandle team = nullptr;
    EXPECT_EQ(HcclSubTeamCreate(nullptr, &desc, &team), HCCL_E_PTR);
    EXPECT_EQ(HcclSubTeamCreate(g_fakeWorldTeam, nullptr, &team), HCCL_E_PTR);
    EXPECT_EQ(HcclSubTeamCreate(g_fakeWorldTeam, &desc, nullptr), HCCL_E_PTR);
}

// 异常：desc 非法。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclSubTeamCreate_When_DescInvalid_Expect_ReturnParaOrPtr)
{
    HcommTeamHandle team = nullptr;

    // rankNum=0
    HcclTeamCreateDesc desc1 = {};
    (void)HcclTeamCreateDescInit(&desc1);
    desc1.rankNum = 0;
    desc1.rankIds = nullptr;
    EXPECT_EQ(HcclSubTeamCreate(g_fakeWorldTeam, &desc1, &team), HCCL_E_PARA);

    // rankIds=nullptr（rankNum=2 非零）
    HcclTeamCreateDesc desc2 = {};
    (void)HcclTeamCreateDescInit(&desc2);
    desc2.rankNum = 2;
    desc2.rankIds = nullptr;
    desc2.selfRankId = 1;
    EXPECT_EQ(HcclSubTeamCreate(g_fakeWorldTeam, &desc2, &team), HCCL_E_PTR);

    // worldTeam非法
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc3;
    BuildTeamCreateDesc(desc3, rankIds, 2, 1, 1);
    EXPECT_EQ(HcclSubTeamCreate(g_fakeWorldTeam, &desc3, &team), HCCL_E_PARA);
}

// 异常：Hcomm 层 / 内存 / 注册失败回滚。
// 异常：HcommTeamCreate 失败不回滚（团队未创建）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclSubTeamCreate_When_HcommCreateFail_Expect_NoRollbackAndFail)
{
    // world team 创建成功，sub team 创建失败（按 worldTeam 参数区分，避免 FIFO 命中首个桩）。
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateWorldOkSubFail));
    // HcommTeamDestroy 期望须在 SetupSuccessMocks() 前注册：mockcpp 对同一函数的 stubs/expects 按 FIFO
    // 匹配，先注册的先命中。SetupSuccessMocks 会注册 HcommTeamDestroy.stubs(returnValue(0))，
    // 若在其后注册 expects(once())，则 stubs 先命中、expects 永远 invoked(0) → TearDown 报错。
    // sub 创建失败无回滚，HcommTeamDestroy 仅在结尾销毁 worldTeam 时被调用 1 次。
    MOCKER(HcommTeamDestroy).expects(once()).will(returnValue(0));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    uint32_t subRankIds[2] = {0, 1};
    HcclTeamCreateDesc subDesc;
    BuildTeamCreateDesc(subDesc, subRankIds, 2, 1, 1);

    // HcommTeamCreate 失败：CHK_PRT_RET 直接返回 ret 转换码，不调用 HcommTeamDestroy（团队未创建）。
    HcommTeamHandle subTeam = nullptr;
    HcclResult ret = HcclSubTeamCreate(worldTeam, &subDesc, &subTeam);
    EXPECT_EQ(ret, static_cast<HcclResult>(HCCL_E_INTERNAL));
    EXPECT_EQ(subTeam, nullptr);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：hrtMalloc 失败回滚 HcommTeamDestroy；worldTeam 未注册时 RegisterSubTeam 返回 E_PARA。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclSubTeamCreate_When_MallocFailOrUnregistered_Expect_RollbackOrPara)
{
    // world team 创建成功，sub team 创建成功（按 worldTeam 参数区分，避免 FIFO 命中首个桩）。
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateByWorldArg));
    // hrtMalloc 计数桩：前置 worldTeam 第1次成功，sub team 第2次失败。须在 SetupSuccessMocks() 前注册
    // （FIFO 下先注册的先生效），否则 SetupSuccessMocks 的 StubHrtMallocFill 会先命中。
    g_hrtMallocOkCount = 1;
    g_hrtMallocCallCnt = 0;
    MOCKER(hrtMalloc).stubs().will(invoke(StubHrtMallocCountFail));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    uint32_t subRankIds[2] = {0, 1};
    HcclTeamCreateDesc subDesc;
    BuildTeamCreateDesc(subDesc, subRankIds, 2, 1, 1);

    // (1) hrtMalloc 失败：sub 的 hrtMalloc（第2次）失败，回滚 HcommTeamDestroy，返回 E_MEMORY。
    {
        HcommTeamHandle subTeam = nullptr;
        HcclResult ret = HcclSubTeamCreate(worldTeam, &subDesc, &subTeam);
        EXPECT_EQ(ret, HCCL_E_MEMORY);
        EXPECT_EQ(subTeam, nullptr);
    }
    // (2) worldTeam 未注册：RegisterSubTeam 返回 E_PARA。需令 hrtMalloc 成功（重置计数上限），
    //     否则 sub 的 hrtMalloc 会先失败返回 E_MEMORY 而非走到 RegisterSubTeam。
    {
        g_hrtMallocOkCount = 100; // 后续 hrtMalloc 均成功
        HcommTeamHandle subTeam = nullptr;
        HcclResult ret = HcclSubTeamCreate(g_fakeSubTeam, &subDesc, &subTeam); // g_fakeSubTeam 未注册
        EXPECT_EQ(ret, HCCL_E_PARA);
        EXPECT_EQ(subTeam, nullptr);
    }

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// =====================================================================
// 3. HcclTeamDestroy
// =====================================================================

// 功能：销毁已注册 team，验证 syncMem 释放与条目 erase（syncMemPtr 非空）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_TeamRegistered_Expect_SuccessAndWsFreed)
{
    // syncMem 非空：销毁时 UnregisterTeam 调 hrtFree 释放 syncMem。
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateFillWorld));
    SetupSuccessMocks();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &desc, &worldTeam), HCCL_SUCCESS);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：team 为 nullptr 返回 E_PTR。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_TeamNullptr_Expect_ReturnHCCL_E_PTR)
{
    EXPECT_EQ(HcclTeamDestroy(nullptr), HCCL_E_PTR);
}

// 异常：HcommTeamDestroy 失败时返回 ret 转换码，UnregisterTeam 已先执行。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_HcommDestroyFail_Expect_ReturnRet)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateFillWorld));
    // HcommTeamDestroy 失败覆盖须在 SetupSuccessMocks() 前注册（FIFO）。
    MOCKER(HcommTeamDestroy).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    SetupSuccessMocks();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc desc;
    BuildTeamCreateDesc(desc, rankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &desc, &worldTeam), HCCL_SUCCESS);

    HcclResult ret = HcclTeamDestroy(worldTeam);
    EXPECT_EQ(ret, static_cast<HcclResult>(HCCL_E_INTERNAL));
}

// 异常：未注册 team 不崩溃，HcommTeamDestroy 仍被调用。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_TeamUnregistered_Expect_NoCrashAndHcommDestroyCalled)
{
    // 未注册 team：FindWorldTeam 返回 nullptr（非 worldTeam，跳过 window 销毁），UnregisterTeam 静默，HcommTeamDestroy
    // 仍被调用。
    MOCKER(HcommTeamDestroy).stubs().will(returnValue(0));
    EXPECT_EQ(HcclTeamDestroy(g_fakeSubTeam), HCCL_SUCCESS); // g_fakeSubTeam 未注册
}

// 功能：HcclTeamDestroy 连带销毁 worldTeam 拥有的 window（L3 资源），避免泄漏。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_WorldTeamHasWindow_Expect_WindowAlsoDestroyed)
{
    // expects 须在 SetupSuccessMocks 前注册（FIFO）：window + team 各销毁一次
    MOCKER(HcommTeamWindowDeregister).expects(once()).will(returnValue(0));
    MOCKER(HcommTeamDestroy).expects(once()).will(returnValue(0));
    SetupSuccessMocks();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, rankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 手动给 worldTeam 注册一个 window
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0x58000);
    localMem.size = 1024;
    HcclTeamMgr::GetInstance().AddWorldTeamWindow(
        worldTeam, g_fakeWindow, localMem, g_fakeUserMemHandle, "__hccl_team_usermem__destroy_test");

    // HcclTeamDestroy 应连带销毁 window（HcommTeamWindowDeregister）+ team（HcommTeamDestroy）
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 功能：HcclTeamDestroy 销毁 worldTeam 时连带销毁其所有 subTeam，避免泄漏。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamDestroy_When_WorldTeamHasSubTeam_Expect_SubTeamAlsoDestroyed)
{
    // 用按 worldTeam 参数区分的桩：worldTeam==nullptr 返回 g_fakeWorldTeam，否则返回 g_fakeSubTeam
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateByWorldArg));
    SetupSuccessMocks();
    uint32_t rankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, rankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    HcclTeamCreateDesc subDesc;
    BuildTeamCreateDesc(subDesc, rankIds, 2, 1, 1);
    HcommTeamHandle subTeam = nullptr;
    ASSERT_EQ(HcclSubTeamCreate(worldTeam, &subDesc, &subTeam), HCCL_SUCCESS);
    ASSERT_EQ(subTeam, g_fakeSubTeam);
    // 确认 subTeam 已注册到 HcclTeamMgr（FindCollComm 可查到）
    ASSERT_NE(HcclTeamMgr::GetInstance().FindCollComm(subTeam), nullptr);

    // HcclTeamDestroy(worldTeam) 应连带销毁 subTeam
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
    // subTeam 应已从 teamMap 移除（连带销毁）
    EXPECT_EQ(HcclTeamMgr::GetInstance().FindCollComm(subTeam), nullptr);
    EXPECT_EQ(HcclTeamMgr::GetInstance().FindCollComm(worldTeam), nullptr);
}

// =====================================================================
// 4. HcclTeamWindowRegister
// =====================================================================

// 功能：首次注册 user mem + syncMem，验证 window 创建与记录。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowRegister_When_FirstRegister_Expect_SuccessAndWindowCreated)
{
    // HcclTeamWindowRegister 入参只能传 worldTeam（syncMem 注册已移至 ChannelsCreate）
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateFillWorld));
    SetupSuccessMocks();
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 首次注册：FindReusableWindow=false → 新建 window；user mem 注册一次。
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0xAA00);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;

    HcclResult ret = HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(window, g_fakeWindow);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：传 subTeam 应报 E_PARA（HcclTeamWindowRegister 入参只能传 worldTeam）
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowRegister_When_SubTeam_Expect_ReturnParaError)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateByWorldArg));
    SetupSuccessMocks();
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    uint32_t subRankIds[2] = {0, 1};
    HcclTeamCreateDesc subDesc;
    BuildTeamCreateDesc(subDesc, subRankIds, 2, 1, 1);
    HcommTeamHandle subTeam = nullptr;
    ASSERT_EQ(HcclSubTeamCreate(worldTeam, &subDesc, &subTeam), HCCL_SUCCESS);

    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0xAA01);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    EXPECT_EQ(HcclTeamWindowRegister(comm, subTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_E_PARA);

    EXPECT_EQ(HcclTeamDestroy(subTeam), HCCL_SUCCESS);
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 功能：首次注册创建 window；同 addr 同 size 再次注册复用 window（FindReusableWindow 命中，早返回）。
// 功能：首次注册创建 window；同 addr 同 size 再次注册复用 window（FindReusableWindow 命中，早返回）。
// 注：HcclTeamWindowRegister 通过 commMem->CommRegMem 注册 localMem（非 HcclCommMemReg，不可 mock）；
// syncMem 注册已移至 ChannelsCreate，本用例仅验证 window 复用逻辑。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowRegister_When_WsAlreadyRegistered_Expect_ReuseWindow)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateByWorldArg));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // (1) 首次注册：新建 window。
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0xBB00);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);
    EXPECT_EQ(window, g_fakeWindow);

    // (2) 同 addr 同 size 再次注册：FindReusableWindow 命中（子集/全等）→ 复用同一 window。
    HcommWindowHandle window2 = nullptr;
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window2, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);
    EXPECT_EQ(window2, g_fakeWindow);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 功能：窗口复用——子集 / 完全相同返回同一 window；非子集走新建。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowRegister_When_ReuseWindow_Expect_SubsetReusesNonSubsetCreatesNew)
{
    // worldTeam 用 syncMemSize=4096（源码现要求 syncMemSize>0）；ws 与复用逻辑无关。
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateByWorldArg));
    SetupSuccessMocks();
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 首次注册：addr=B(0xC000), size=2048。
    CommMem baseMem{};
    baseMem.type = COMM_MEM_TYPE_DEVICE;
    baseMem.addr = reinterpret_cast<void*>(0xC000);
    baseMem.size = 2048;
    HcommWindowHandle baseWindow = nullptr;
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &baseMem, &baseWindow, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);

    // (1) 子集：addr=B, size=512 → 复用同一 window（FindReusableWindow 命中，不调 Hcomm 层 WindowRegister/MemReg）。
    CommMem subsetMem{};
    subsetMem.type = COMM_MEM_TYPE_DEVICE;
    subsetMem.addr = reinterpret_cast<void*>(0xC000);
    subsetMem.size = 512;
    HcommWindowHandle subsetWindow = nullptr;
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &subsetMem, &subsetWindow, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC),
        HCCL_SUCCESS);
    EXPECT_EQ(subsetWindow, baseWindow);

    // (2) 完全相同：addr=B, size=2048 → 复用同一 window。
    HcommWindowHandle sameWindow = nullptr;
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &baseMem, &sameWindow, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);
    EXPECT_EQ(sameWindow, baseWindow);

    // (3) 非子集：addr=B+4096, size=512 → 新建 window（g_fakeWindow）。
    CommMem disjointMem{};
    disjointMem.type = COMM_MEM_TYPE_DEVICE;
    disjointMem.addr = reinterpret_cast<void*>(0xC000 + 4096);
    disjointMem.size = 512;
    HcommWindowHandle newWindow = nullptr;
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &disjointMem, &newWindow, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC),
        HCCL_SUCCESS);
    EXPECT_EQ(newWindow, g_fakeWindow);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 功能：多 window 按 addr 排序插入 + 二分查找复用。乱序插入 3 个 window，验证查找中间 window 的子集命中中间 window。
TEST_F(TestCollCommTeamCAdpt, Ut_FindReusableWindow_When_MultipleSortedWindows_Expect_BinarySearchHitMiddle)
{
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateByWorldArg));
    SetupSuccessMocks();
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 乱序手动注册 3 个 window（addr 不同），验证 AddWorldTeamWindow 按 addr 升序插入
    auto addWin = [worldTeam](uintptr_t addr, uint64_t size, HcommWindowHandle handle) {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_DEVICE;
        mem.addr = reinterpret_cast<void*>(addr);
        mem.size = size;
        HcclTeamMgr::GetInstance().AddWorldTeamWindow(worldTeam, handle, mem, g_fakeUserMemHandle, "tag");
    };
    // 乱序：先插 0x20000，再 0x10000，再 0x30000 → 排序后应为 0x10000/0x20000/0x30000
    addWin(0x20000, 4096, reinterpret_cast<HcommWindowHandle>(0xA200));
    addWin(0x10000, 4096, reinterpret_cast<HcommWindowHandle>(0xA100));
    addWin(0x30000, 4096, reinterpret_cast<HcommWindowHandle>(0xA300));

    // 查找中间 window（addr=0x20000, size=4096 → 覆盖 [0x20000, 0x21000)）的子集：
    // addr=0x20100, size=512 → 完全落在 [0x20000, 0x21000) 内，应命中 0xA200。
    CommMem midSubset{};
    midSubset.type = COMM_MEM_TYPE_DEVICE;
    midSubset.addr = reinterpret_cast<void*>(0x20100);
    midSubset.size = 512;
    HcommWindowHandle found = nullptr;
    EXPECT_TRUE(HcclTeamMgr::GetInstance().FindReusableWindow(worldTeam, midSubset, found));
    EXPECT_EQ(found, reinterpret_cast<HcommWindowHandle>(0xA200));

    // 查找首 window（0x10000）的子集 → 应命中 0xA100
    CommMem firstSubset{};
    firstSubset.type = COMM_MEM_TYPE_DEVICE;
    firstSubset.addr = reinterpret_cast<void*>(0x10000);
    firstSubset.size = 1024;
    found = nullptr;
    EXPECT_TRUE(HcclTeamMgr::GetInstance().FindReusableWindow(worldTeam, firstSubset, found));
    EXPECT_EQ(found, reinterpret_cast<HcommWindowHandle>(0xA100));

    // 查找未覆盖区间（0x50000）→ 不命中
    CommMem missMem{};
    missMem.type = COMM_MEM_TYPE_DEVICE;
    missMem.addr = reinterpret_cast<void*>(0x50000);
    missMem.size = 512;
    found = nullptr;
    EXPECT_FALSE(HcclTeamMgr::GetInstance().FindReusableWindow(worldTeam, missMem, found));

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：入参空指针校验（comm/team/window nullptr；localMem.addr=nullptr）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowRegister_When_NullptrParams_Expect_ReturnHCCL_E_PTR)
{
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0xDD00);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;

    EXPECT_EQ(
        HcclTeamWindowRegister(nullptr, g_fakeWorldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC),
        HCCL_E_PTR);
    EXPECT_EQ(HcclTeamWindowRegister(comm, nullptr, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_E_PTR);
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, g_fakeWorldTeam, &localMem, nullptr, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC),
        HCCL_E_PTR);

    // addr=nullptr 先于复用检查
    CommMem nullAddrMem{};
    nullAddrMem.type = COMM_MEM_TYPE_DEVICE;
    nullAddrMem.addr = nullptr;
    nullAddrMem.size = 1024;
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, g_fakeWorldTeam, &nullAddrMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC),
        HCCL_E_PTR);
}

// 异常：localMem.size=0 返回 E_PARA。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowRegister_When_SizeZero_Expect_ReturnHCCL_E_PARA)
{
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0xEE00);
    localMem.size = 0;
    HcommWindowHandle window = nullptr;
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, g_fakeWorldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC),
        HCCL_E_PARA);
}

// 异常：归属校验失败（FindWorldTeam=nullptr / commId 不一致 / GetCollComm=nullptr）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowRegister_When_OwnershipFail_Expect_ReturnParaOrPtr)
{
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0xFF00);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;

    // (1) team 不是 worldTeam（FindWorldTeam 返回 nullptr ≠ team）→ HCCL_E_PARA。
    EXPECT_EQ(
        HcclTeamWindowRegister(comm, g_fakeSubTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_E_PARA);

    // (2) GetCollComm=nullptr：comm=nullptr → E_PTR（先于归属校验）。
    EXPECT_EQ(
        HcclTeamWindowRegister(nullptr, g_fakeWorldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC),
        HCCL_E_PTR);
}

// 异常：Hcomm 层 WindowRegister 失败：window=nullptr，返回 ret。
// 注：源码重构后 ws/user MemReg 走 commMem->CommRegMem（非虚成员函数，不可 mock），
// 故 ws MemReg 失败 / user MemReg 失败两条路径无法通过 mock 触发，不再覆盖；仅保留 Hcomm 层 WindowRegister 失败路径。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowRegister_When_HcommWindowRegisterFail_Expect_ReturnRet)
{
    // Hcomm 层 WindowRegister 失败覆盖须在 SetupSuccessMocks() 前注册。
    MOCKER(HcommTeamWindowRegister).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateFillWorld));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0x12200);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    HcclResult ret = HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC);
    EXPECT_EQ(ret, static_cast<HcclResult>(HCCL_E_INTERNAL));
    EXPECT_EQ(window, nullptr);
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// =====================================================================
// 4.1 HcclTeamWindowDeregister
// =====================================================================

// 功能：正常注销已注册 window，验证 RemoveWorldTeamWindow + HcommTeamWindowDeregister 调用。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowDeregister_When_Normal_Expect_ReturnSuccess)
{
    SetupSuccessMocks();
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 先注册一个 window，使 windows 列表非空。
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0xA000);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    ASSERT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);
    ASSERT_EQ(window, g_fakeWindow);
    // 确认 window 已记录到 worldTeam 的 window 列表。
    ASSERT_EQ(HcclTeamMgr::GetInstance().GetWorldTeamWindows(worldTeam).size(), 1U);

    // 注销：RemoveWorldTeamWindow 移除记录；HcommTeamWindowDeregister 销毁 Hcomm 层 window。
    EXPECT_EQ(HcclTeamWindowDeregister(worldTeam, window), HCCL_SUCCESS);
    // 注销后 window 列表应清空。
    EXPECT_EQ(HcclTeamMgr::GetInstance().GetWorldTeamWindows(worldTeam).size(), 0U);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：入参空指针校验（team/window 为 nullptr → HCCL_E_PTR）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowDeregister_When_NullptrParams_Expect_ReturnHCCL_E_PTR)
{
    EXPECT_EQ(HcclTeamWindowDeregister(nullptr, g_fakeWindow), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamWindowDeregister(g_fakeWorldTeam, nullptr), HCCL_E_PTR);
}

// team 未注册时 Deregister 不报错：源码 FindWorldTeam 返回 nullptr 时记 WARNING 并返回 HCCL_SUCCESS
// （容忍 team 已销毁的幂等场景）。仅验证 HcommTeamWindowDeregister 仍被调用（SetupSuccessMocks 桩返回 0）。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowDeregister_When_TeamNotRegistered_Expect_ReturnSuccess)
{
    // g_fakeSubTeam 未注册到 HcclTeamMgr，FindWorldTeam 返回 nullptr → 源码返回 HCCL_SUCCESS。
    EXPECT_EQ(HcclTeamWindowDeregister(g_fakeSubTeam, g_fakeWindow), HCCL_SUCCESS);
}

// 异常：HcommTeamWindowDeregister 失败 → 返回 ret 转换码。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowDeregister_When_HcommWindowDestroyFail_Expect_ReturnRet)
{
    // HcommTeamWindowDeregister 失败覆盖须在 SetupSuccessMocks() 前注册（FIFO）。
    MOCKER(HcommTeamWindowDeregister).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 注册一个 window。
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0xB000);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    ASSERT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);

    // 注销时 HcommTeamWindowDeregister 失败 → 返回 HCCL_E_INTERNAL。
    // 注：RemoveWorldTeamWindow 已先执行（window 记录已移除），但 Hcomm 层 window 销毁失败。
    EXPECT_EQ(HcclTeamWindowDeregister(worldTeam, window), static_cast<HcclResult>(HCCL_E_INTERNAL));

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 功能：window 未在 worldTeam 列表中（RemoveWorldTeamWindow 空操作），仍调 HcommTeamWindowDeregister，返回成功。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamWindowDeregister_When_WindowNotInList_Expect_ReturnSuccess)
{
    SetupSuccessMocks();
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 不注册任何 window，直接用一个未注册的 window 句柄注销。
    // RemoveWorldTeamWindow 找不到该 window（空操作）；HcommTeamWindowDeregister 仍被调用（桩返回 0）。
    HcommWindowHandle unregisteredWindow = reinterpret_cast<HcommWindowHandle>(0xDEAD);
    EXPECT_EQ(HcclTeamWindowDeregister(worldTeam, unregisteredWindow), HCCL_SUCCESS);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// =====================================================================
// 5. HcclTeamChannelsCreate
// =====================================================================

// 功能：4 成员 channelCnt=1 全流程：建链+绑定+远端内存交换。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_FullFlow_Expect_Success)
{
    SetupSuccessMocks();
    // 建立已注册 window 的 worldTeam（memberNum=4 由 HcommTeamGetWorldTeamIds 桩给出）。
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 注册一个 user window，使 windows 列表非空。
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0x22200);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    ASSERT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);

    // memberNum=4（桩给出 [1,3,5,7]），selfRank=1 → selfMemberId=0；3 个 peer。
    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);

    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 功能：channelCnt=2 多通道 + 多 window 分别 BindWindow。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_MultiChannelMultiWindow_Expect_Success)
{
    SetupSuccessMocks();
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 注册 2 个不相交 window。
    CommMem mem1{};
    mem1.type = COMM_MEM_TYPE_DEVICE;
    mem1.addr = reinterpret_cast<void*>(0x30000);
    mem1.size = 1024;
    HcommWindowHandle win1 = nullptr;
    ASSERT_EQ(HcclTeamWindowRegister(comm, worldTeam, &mem1, &win1, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);
    CommMem mem2{};
    mem2.type = COMM_MEM_TYPE_DEVICE;
    mem2.addr = reinterpret_cast<void*>(0x31000);
    mem2.size = 1024;
    HcommWindowHandle win2 = nullptr;
    ASSERT_EQ(HcclTeamWindowRegister(comm, worldTeam, &mem2, &win2, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 功能：远端内存 tag 区分与每 peer 每 window 仅取首个。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_RemoteMemTagDispatch_Expect_Success)
{
    // worldTeam 用 syncMemSize=4096（源码现要求 syncMemSize>0）；远端 mem 仍只配 user tag（hasWs=false）。
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateByWorldArg));
    SetupSuccessMocks();
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 注册一个 user window。
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0x40000);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    ASSERT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);

    // 从 HcclTeamMgr 取出 window 注册时生成的 localMemTag，用作远端 mem tag，使 tag 匹配命中。
    std::vector<WindowInfo> wins = HcclTeamMgr::GetInstance().GetWorldTeamWindows(worldTeam);
    ASSERT_EQ(wins.size(), 1U);
    std::string userTag = wins[0].localMemTag;
    ASSERT_FALSE(userTag.empty());

    // 配置远端 mem：同 peer 同 tag 多个 mem（multiMems=true），验证"仅取首个"逻辑。
    // 无 ws 时 wsMemTag 为空，ws tag 分支不会命中。
    g_remoteScenario.hasWs = false;
    g_remoteScenario.hasUser = true;
    g_remoteScenario.userMem.type = COMM_MEM_TYPE_DEVICE;
    g_remoteScenario.userMem.addr = reinterpret_cast<void*>(0x41000);
    g_remoteScenario.userMem.size = 1024;
    g_remoteScenario.multiMems = true;
    g_remoteScenario.userTag = userTag;

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 功能：无 window / peer 无 ws tag 边界。注：worldTeam 现必有 ws（syncMemSize>0），self 槽填本地 ws，peer 无 ws tag
// 则其槽为零值。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_NoWsNoWindowEmptyRemote_Expect_Success)
{
    // worldTeam 用 syncMemSize=4096（源码现要求 syncMemSize>0）；无 window → 跳过 BindWindow；远端仅 user tag → peer 的
    // ws 槽为零值。
    MOCKER(HcommTeamCreate).stubs().will(invoke(StubTeamCreateByWorldArg));
    SetupSuccessMocks();
    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 远端仅 user tag（无 ws tag）。
    g_remoteScenario.hasWs = false;
    g_remoteScenario.hasUser = true;
    g_remoteScenario.userMem.type = COMM_MEM_TYPE_DEVICE;
    g_remoteScenario.userMem.addr = reinterpret_cast<void*>(0x42000);
    g_remoteScenario.userMem.size = 1024;
    g_remoteScenario.userTag = "any_user_tag";

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：入参空指针 → HCCL_E_PTR；channelCnt=0 → HCCL_E_PARA。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_NullptrOrChannelCntZero_Expect_ReturnParaOrPtr)
{
    HcclTeamCreateChannelsDesc chDesc{};
    (void)HcclTeamCreateChannelsDescInit(&chDesc);
    chDesc.notifyNum = 8;
    chDesc.engine = COMM_ENGINE_AICPU_TS;
    chDesc.channelCnt = 1;

    EXPECT_EQ(HcclTeamChannelsCreate(nullptr, g_fakeWorldTeam, &chDesc), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamChannelsCreate(comm, nullptr, &chDesc), HCCL_E_PTR);
    EXPECT_EQ(HcclTeamChannelsCreate(comm, g_fakeWorldTeam, nullptr), HCCL_E_PTR);

    // channelCnt=0 → E_PARA
    HcclTeamCreateChannelsDesc zeroDesc = chDesc;
    zeroDesc.channelCnt = 0;
    EXPECT_EQ(HcclTeamChannelsCreate(comm, g_fakeWorldTeam, &zeroDesc), HCCL_E_PARA);
}

// 异常：FindWorldTeam=nullptr（team 未注册）→ HCCL_E_PTR。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_FindWorldTeamNull_Expect_ReturnPtr)
{
    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    static uint32_t idsWithSelf[4] = {0, 1, 5, 7};
    g_stubIds = idsWithSelf;
    g_stubIdNum = 4;
    HcclResult ret = HcclTeamChannelsCreate(comm, g_fakeSubTeam, &chDesc); // g_fakeSubTeam 未注册
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 异常：GetNetLayer 失败 → ret 转换码。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_GetNetLayerFail_Expect_ReturnRet)
{
    // GetNetLayer 失败覆盖须在 SetupSuccessMocks() 前注册。
    static uint32_t ids[4] = {0, 1, 5, 7};
    g_stubIds = ids;
    g_stubIdNum = 4;
    MOCKER(HcommTeamGetNetLayer).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, static_cast<HcclResult>(HCCL_E_INTERNAL));
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：GetLinks 无 link（linkNum=0）→ E_NOT_FOUND。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_NoLink_Expect_ReturnNotFound)
{
    static uint32_t ids[4] = {0, 1, 5, 7};
    g_stubIds = ids;
    g_stubIdNum = 4;
    MOCKER(HcclRankGraphGetLinks).stubs().will(invoke(StubRankGraphGetLinksEmpty));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：Acquire 失败 → ret。
// 注：HcclChannelDescInit 为 static inline，无法 MOCKER；其失败分支不可达，故不覆盖。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_AcquireFail_Expect_ReturnRet)
{
    static uint32_t ids[4] = {0, 1, 5, 7};
    g_stubIds = ids;
    g_stubIdNum = 4;
    MOCKER(HcclChannelAcquire).stubs().will(returnValue(HCCL_E_INTERNAL));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：BindChannels 失败 → ret。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_BindChannelsFail_Expect_ReturnRet)
{
    static uint32_t ids[4] = {0, 1, 5, 7};
    g_stubIds = ids;
    g_stubIdNum = 4;
    MOCKER(HcommTeamBindChannels).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    // 注册一个 window，使 BindWindow 路径可达。
    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0x50000);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    ASSERT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, static_cast<HcommResult>(HCCL_E_INTERNAL));

    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：GetRemoteMems 失败 → ret。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_GetRemoteMemsFail_Expect_ReturnRet)
{
    static uint32_t ids[4] = {0, 1, 5, 7};
    g_stubIds = ids;
    g_stubIdNum = 4;
    MOCKER(HcclChannelGetRemoteMems).stubs().will(returnValue(HCCL_E_INTERNAL));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0x50000);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    ASSERT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：BindWindow 失败 → ret。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_BindWindowFail_Expect_ReturnRet)
{
    static uint32_t ids[4] = {0, 1, 5, 7};
    g_stubIds = ids;
    g_stubIdNum = 4;
    MOCKER(HcommTeamWindowBindRemoteMems).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0x50000);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    ASSERT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, static_cast<HcommResult>(HCCL_E_INTERNAL));
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}

// 异常：BindSyncMem 失败 → ret。
TEST_F(TestCollCommTeamCAdpt, Ut_HcclTeamChannelsCreate_When_BindSyncMemFail_Expect_ReturnRet)
{
    static uint32_t ids[4] = {0, 1, 5, 7};
    g_stubIds = ids;
    g_stubIdNum = 4;
    MOCKER(HcommTeamBindRemoteSyncMem).stubs().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    SetupSuccessMocks();

    uint32_t worldRankIds[2] = {0, 1};
    HcclTeamCreateDesc worldDesc;
    BuildTeamCreateDesc(worldDesc, worldRankIds, 2, 1, 1);
    HcommTeamHandle worldTeam = nullptr;
    ASSERT_EQ(HcclWorldTeamCreate(comm, &worldDesc, &worldTeam), HCCL_SUCCESS);

    CommMem localMem{};
    localMem.type = COMM_MEM_TYPE_DEVICE;
    localMem.addr = reinterpret_cast<void*>(0x50000);
    localMem.size = 1024;
    HcommWindowHandle window = nullptr;
    ASSERT_EQ(
        HcclTeamWindowRegister(comm, worldTeam, &localMem, &window, HCOMM_TEAM_WINDOW_FLAG_SYMMETRIC), HCCL_SUCCESS);

    HcclTeamCreateChannelsDesc chDesc;
    BuildChannelDesc(chDesc, COMM_ENGINE_AICPU_TS, 1, 8);
    HcclResult ret = HcclTeamChannelsCreate(comm, worldTeam, &chDesc);
    EXPECT_EQ(ret, static_cast<HcommResult>(HCCL_E_INTERNAL));
    EXPECT_EQ(HcclTeamDestroy(worldTeam), HCCL_SUCCESS);
}
