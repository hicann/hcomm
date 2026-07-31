# CcuEventHandle

## 功能说明

CCU Event（事件）的句柄类型，用于标识CCU kernel内通过[Event](../data_plane_api/ccu/resource_allocation_operation/Event.md)资源创建的事件资源。该句柄用于CCU数据面同步接口（如EventRecord、EventWait）中。

## 定义原型

```c
typedef uint64_t CcuEventHandle;
```
