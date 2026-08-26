# Dispatching the Operator

<!-- md-trans-meta sourceCommit=4b3acc1183ff175f340b0421dbe1faf9a723a585 translatedAt=2026-08-11T07:16:16.419Z pushedAt=2026-08-20T11:39:14.571Z -->

After completing the communication operator kernel registration, developers need to dispatch the kernel function to a specific communication engine for execution.

For the CCU communication engine, developers need to call the HcommCcuKernelLaunch API to dispatch the kernel.

## Sample Code

Taking a custom AllGather operator as an example, when using the CCU communication engine, the code snippet for dispatching the CCU kernel function on the host is as follows:

```c
uint64_t currentRankSliceInputOffset = 0; // Input address offset between devices
uint64_t currentRankSliceOutputOffset = sliceSize* myRank; // Target address offset between devices.
std::vector<uint64_t> taskArgs = {
    inputAddr,
    outputAddr,
    token,
    currentRankSliceInputOffset,
    currentRankSliceOutputOffset,
    sliceSize
};
CcuResult launchRet = HcommCcuKernelLaunch(thread, ccuKernel, taskArgs.data(), taskArgs.size()); // Dispatch the CCU task.
if (launchRet != CCU_SUCCESS) {
    HCCL_ERROR("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] kernel launch failed, ccuRet -> %d", launchRet);
    return ConvertCcuToHccl(launchRet);
}
```
