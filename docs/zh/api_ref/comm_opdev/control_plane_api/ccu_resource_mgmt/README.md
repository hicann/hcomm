# 简介

CCU资源管理接口用于查询和描述CCU资源诉求、按需创建CCU实例、查询实例资源占用，以及销毁调用方持有的实例。

按需创建CCU实例的典型流程如下：

1. 调用[HcommCcuInsResDescCreate](HcommCcuInsResDescCreate.md)为目标IO Die创建资源描述符。
2. 调用[HcommCcuKernelQueryResReq](HcommCcuKernelQueryResReq.md)统计Kernel资源诉求，或调用[HcommCcuInsResDescSetNum](HcommCcuInsResDescSetNum.md)直接设置资源数量。
3. 调用[HcommCcuInsCreate](HcommCcuInsCreate.md)创建CCU实例。
4. 可调用[HcommCcuInsQueryResDesc](HcommCcuInsQueryResDesc.md)查询实例实际占用的资源数量。
5. 不再使用实例时，调用[HcommCcuInsDestroy](HcommCcuInsDestroy.md)销毁；若通过[HcclCommAssignCcuIns](../comms_domain_resource_mgmt/HcclCommAssignCcuIns.md)成功绑定到通信域，则由通信域管理实例生命周期。
