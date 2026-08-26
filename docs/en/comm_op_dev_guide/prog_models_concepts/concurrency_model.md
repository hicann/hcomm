# Concurrency Model

<!-- md-trans-meta sourceCommit=4b3acc1183ff175f340b0421dbe1faf9a723a585 translatedAt=2026-08-11T07:19:50.927Z pushedAt=2026-08-20T11:39:14.577Z -->

A communication operator consists of different communication tasks. If certain communication steps have no resource conflicts, they can be executed concurrently. HCCL provides a concurrency model in the operator programming model. The concurrency model requires defining concurrency units and the synchronization behavior between them, as shown in the following figure.

![](figures/concurrency_model.png)

- Concurrency unit: Provides a concurrency unit abstracted as a thread. Communication tasks are bound to threads, and operations across different threads can be executed concurrently.
- Synchronization between concurrency units:
  - A thread can contain multiple Notify instances. The number of Notify instances can be specified when creating a thread.
  - Within a communication entity, a thread can send a synchronization signal to another thread, and a thread can wait for a synchronization signal from another thread. For details about the APIs, see [Local Operations](../../api_ref/comm_opdev/data_plan_api/cpu-cpu_ts-aicpu_ts/local_operations/README.md).

Under different communication engines, a thread may correspond to different concurrency entities and synchronization methods:

- AI CPU+TS and Host CPU+TS: The concurrency entity is a stream. Synchronization between streams is implemented using the NPU's Notify registers. A thread is an encapsulation abstraction of a stream and Notify registers.
- AIV: The concurrency entity is a Vector Core. Synchronization between AIV cores is implemented using device memory. A thread is an encapsulation abstraction of an AIV core and device memory.
