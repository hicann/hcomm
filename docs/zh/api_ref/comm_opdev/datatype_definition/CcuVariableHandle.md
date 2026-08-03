# CcuVariableHandle

## 功能说明

CCU Variable（变量）的句柄类型，用于标识CCU kernel内通过[Variable](../data_plane_api/ccu/resource_allocation_operation/Variable.md)资源创建的变量资源。该句柄由CCU资源创建接口返回，在CCU数据面接口中使用。

## 定义原型

```c
typedef uint64_t CcuVariableHandle;
```
