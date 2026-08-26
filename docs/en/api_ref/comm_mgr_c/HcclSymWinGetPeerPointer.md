# HcclSymWinGetPeerPointer

<!-- md-trans-meta sourceCommit=a6ab154081224a161b017e5ef386437c91108d23 translatedAt=2026-08-14T08:56:11.590Z pushedAt=2026-08-15T07:49:17.610Z -->

## Supported Products

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 training products/Atlas A3 inference products: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 training products/Atlas A2 inference products: Not supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Not supported
<!-- end id5 -->

## Description

Obtains the address pointer corresponding to the offset in the symmetric memory window of a specified rank ID based on the symmetric memory window resource handle and the offset.

For Atlas A3 training products/Atlas A3 inference products, this API supports the HCCS link communication scenario; for Ascend 950PR/Ascend 950DT, this API supports the URMA scenario.

## Function Prototype

```c
HcclResult HcclSymWinGetPeerPointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| winHandle | Input | Handle of the symmetric memory window resource. |
| offset | Input | Offset obtained by calling [HcclCommSymWinGet](HcclCommSymWinGet.md). |
| peerRank | Input | Rank ID, in the range [0, rankSize). |
| ptr | Output | Pointer to the corresponding address in the symmetric memory window. |

## Return Value

[HcclResult](./data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- For Atlas A3 training products/Atlas A3 inference products, only the HCCS link communication scenario is supported; for Ascend 950PR/Ascend 950DT, only the URMA scenario is supported.
- Only the scenario where the communication operator expansion mode is AI CPU is supported.
- This API can be called only on the device side.
- In the URMA scenario of Ascend 950PR/Ascend 950DT, before calling this API, ensure that the symmetric memory window has been registered and that the related UB/URMA communication channels have completed link establishment and remote memory information update.

  When using collective communication APIs, channel creation and remote memory information update are completed internally by the collective communication framework. When using independent communication channel resource APIs, call this API only after the channel is successfully created.

- In the URMA scenario of Ascend 950PR/Ascend 950DT, the winHandle passed in must be a valid registered symmetric memory window handle. If the window handle obtained through [HcclCommSymWinGet](HcclCommSymWinGet.md) is not hit, the returned winHandle is empty and cannot be passed to this API. If the remote memory information of the corresponding peerRank has not been updated, this API returns an error.

## Example

After the symmetric memory window is registered on the host side, pass the window as a parameter to the AI CPU kernel. This function must be compiled to run on the device AI CPU side. The following is the pseudocode description:

```c
AicpuKernelFunc(param):
// Obtain the symmetric window from param.
HcclCommSymWindow temp_win = param.win;
void *src_ptr;
void *dest_ptr;
int srcRankId = 0;
int destRankId = 1;
// Obtain the address corresponding to peerRank by using win + offset + peerRank.
HcclSymWinGetPeerPointer(temp_win, 0, srcRankId, &src_ptr);
HcclSymWinGetPeerPointer(temp_win, 0, destRankId, &dest_ptr);
// The obtained address can be directly read and written using data plane local copy. The thread and size must be prepared by the caller.
HcommLocalCopyOnThread(thread, dest_ptr, src_ptr, size);
```
