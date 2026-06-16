# 通信模型

**图 1**  HCCL通信模型  
![HCCL通信模型](figures/hccl_communication_model.png)

上图描述了HCCL的通信模型，其中均为软件概念，下面分别对相应概念进行解释：

- 通信内存：通信成员上的一块内存可以注册给通信域，表示这块内存可以被通信域内的其他通信成员访问。
- Endpoint：表示与其他通信对象通信时使用的端口（如网卡的NetDevice等）。
  - Endpoint包含地址与协议等属性，一个Endpoint可能包含多个物理口（比如Bonding场景）。
  - 每个通信对象可以包含多个Endpoint。

- Channel：通信对象基于特定Endpoint与其他通信对象特定Endpoint之间建立的通信通道。
  - 一对Endpoint之间可以建立多个Channel。
  - 创建Channel时，本端与远端需要同步调用Channel创建接口。
  - 创建Channel时，本端注册的内存信息会与远端注册的内存信息进行交换，控制面也提供了基于Channel查询远端内存信息（地址与大小）的接口。

通信模型又分为网络语义通信模型和内存语义通信模型：

- 网络语义通信模型：通信算子开发者基于Channel访问远端通信对象内存，或与远端通信对象同步。
- 内存语义通信模型：远端对象的通信内存可以映射到本地内存地址空间中，通信算子开发者可以直接使用本地内存拷贝操作访问远端通信内存。

## 网络语义通信模型

在网络语义通信模型中，用户使用Channel读写远端通信对象内存或与远端通信对象进行同步，详见[通信操作](../../api_ref/comm_opdev/data_plan_api/cpu_ts/communication_operations/README.md)接口介绍。

**图 2**  网络语义通信模型  
![网络语义通信模型](figures/semantic_communication.png)

网络语义通信模型的关键对象是Channel，开发者可以在一对Endpoint之间创建多个Channel。下面打开Channel模型，详细介绍内部元素。

下图是RoCE协议下的Channel模型。

**图 3**  RoCE场景的Channel模型  
![RoCE场景的Channel模型](figures/roce_channel_model.png)

- Channel是本地与远端对象通信的入口，本端通信对象的Channel与远端通信对象的Channel具有一对一的关联关系。
- Channel关联一个或多个QP（Queue Pair）实例，与远端Channel关联的QP实例存在对应关系，Channel建立时对应QP会建链。
- Channel包含多个Notify实例，用于通信对象之间的同步操作。Notify是用于同步操作的抽象概念，在不同通信引擎下可能由不同实体实现。
  - 创建Channel时可以指定Channel包含的Notify实例数量。
  - 本端可以通过Channel向远端对象对应Channel中的某个Notify（使用序号指定）发送同步信号，详见[数据面接口](../../api_ref/comm_opdev/data_plan_api/README.md)章节。
  - 本端可以基于Channel的某个Notify，等待来自远端通信对象的同步信号，等到同步信号后才可以执行后续操作。

## 内存语义通信模型

在内存语义通信模型中，远端对象的通信内存可以映射到本地进程地址空间中，通信算子开发者可以使用本地操作接口实现节点间的数据搬运或同步，详见[本地操作](../../api_ref/comm_opdev/data_plan_api/cpu_ts/local_operations/README.md)接口。

下图展示了HCCL的内存语义通信模型：

**图 4**  内存语义通信模型  
![内存语义通信模型](figures/memory_semantic_model.png)

- Endpoint：表示与其他通信对象进行通信时的网络逻辑口。
  - 内存语义通信模型下，一个通信对象只有一个Endpoint。
  - Endpoint包含地址与协议等属性，一个Endpoint可能包含多个物理口（比如Bonding场景）。

- Channel：本端通信对象与远端通信对象建立的通信通道，在内存语义场景下Channel的建立表示两个通信对象之间的内存映射功能被开启，不用于通信对象间的通信。
  - 一对通信对象之间只能创建一个Channel。
  - 创建Channel时，本端注册的内存信息会与远端注册的内存信息进行交换并做内存映射。
  - 控制面提供了基于Channel查询远端内存在本端映射后的地址与大小。
