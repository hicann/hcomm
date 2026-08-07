/**
 * Stub file: provides no-op replacements for functions that mockcpp can't hook,
 * allowing test coverage of hccl_group.cc without real communicator setup.
 * -fno-access-control is enabled, so we can set private members directly.
 */
#include "hccl_comm_pub.h"
#include "op_base.h"
#include "adapter_rts_common.h"

namespace hccl {

HcclResult hcclComm::SetGroupMode(bool isGroup)
{
    isGroupMode_ = isGroup;
    return HCCL_SUCCESS;
}

} // namespace hccl

HcclResult HcclBatchSendRecvGroup(HcclSendRecvItem*, uint32_t, HcclComm, aclrtStream) { return HCCL_SUCCESS; }

HcclResult hcclStreamSynchronize(HcclRtStream, s32) { return HCCL_SUCCESS; }
