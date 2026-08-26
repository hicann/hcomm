# Overall Process

<!-- md-trans-meta sourceCommit=4bce6591eb0a4898412343ee53a224437de00bfe translatedAt=2026-08-11T07:05:35.687Z pushedAt=2026-08-20T11:39:14.561Z -->

The development and execution of HCCL AI CPU communication operators are divided into two phases: preparation on the host and execution on the AI CPU. The host is responsible for parsing the topology, selecting algorithms, allocating resources, and delivering AI CPU kernels. After the kernel is launched, the AI CPU performs task orchestration based on the resource context. Therefore, **kernel delivery occurs first, and task orchestration follows**.

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "primaryColor": "#EAF2FF",
    "primaryBorderColor": "#4F7CAC",
    "primaryTextColor": "#1F2937",
    "actorBorder": "#4F7CAC",
    "actorBkg": "#F8FBFF",
    "activationBkgColor": "#DCEBFA",
    "activationBorderColor": "#4F7CAC",
    "sequenceNumberColor": "#6B7280",
    "noteBkgColor": "#FFF7D6",
    "noteBorderColor": "#D6A700"
  }
}}%%
sequenceDiagram
    title HCCL AI CPU Communication Operator Execution Process

    participant HCCL_H as Communication Operator API (Host)
    participant HCOMM_H as HCOMM (Host API)
    participant RTS as RTS
    participant HCCL_K as AICPU Kernel
    participant HCOMM_D as HCOMM (Device API)

    activate HCCL_H

    Note over HCCL_H: The host prepares operator execution information.
    HCCL_H->>HCCL_H: Construct operator parameters (I/O addresses, data size, data type, etc.)
    HCCL_H->>+HCOMM_H: Query rank, topology levels, and available links.
    HCOMM_H-->>-HCCL_H: Return communicator and topology information.

    opt Operator has multiple implementations
        HCCL_H->>HCCL_H: Select algorithm based on operator type, execution engine, topology, and data size.
    end

    Note over HCCL_H,HCOMM_H: Resources are reused at the granularity of operator and algorithm tag.
    HCCL_H->>+HCOMM_H: HcclEngineCtxGet(tag, engine)
    alt Resource exists
        HCOMM_H-->>HCCL_H: Return Device Context
    else First execution
        HCOMM_H-->>HCCL_H: Context does not exist.
        HCCL_H->>HCOMM_H: Create Context.
        HCCL_H->>HCOMM_H: Request and export host/AI CPU control thread.
        HCCL_H->>HCOMM_H: Request algorithm thread, Notify, channel, and communication memory.
        HCCL_H->>HCOMM_H: Serialize resource context and copy to device.
        HCOMM_H-->>HCCL_H: Return Device Context
    end
    deactivate HCOMM_H

    Note over HCCL_H,HCCL_K: Host dispatches kernel first, task orchestration occurs after kernel launch.
    HCCL_H->>+HCOMM_H: Host thread notifies AI CPU control thread.
    HCOMM_H-->>-HCCL_H: HcclResult
    HCCL_H->>+RTS: aclrtLaunchKernelWithConfig dispatches AI CPU kernel.
    RTS-->>HCCL_H: aclResult
    RTS->>HCCL_K: Launch kernel.
    deactivate RTS
    HCCL_H->>+HCOMM_H: Host thread waits for AI CPU completion notification.

    activate HCCL_K
    HCCL_K->>HCCL_K: Deserialize resource context.
    HCCL_K->>+HCOMM_D: HcommBatchModeStart(tag)
    HCOMM_D-->>-HCCL_K: HcclResult
    HCCL_K->>+HCOMM_D: AI CPU control thread waits for host startup notification.
    HCOMM_D-->>-HCCL_K: HcclResult

    Note over HCCL_K,HCOMM_D: Execute algorithm task orchestration within kernel.
    HCCL_K->>HCCL_K: ExecOp(param, resCtx)
    HCCL_K->>+HCOMM_D: Orchestrate thread sync, channel sync, and data movement tasks.
    HCOMM_D-->>-HCCL_K: HcclResult

    HCCL_K->>+HCOMM_D: AI CPU control thread notifies host thread.
    HCOMM_D-->>-HCCL_K: HcclResult
    HCCL_K->>+HCOMM_D: HcommBatchModeEnd(tag)
    HCOMM_D-->>-HCCL_K: HcclResult
    deactivate HCCL_K

    HCOMM_H-->>-HCCL_H: AI CPU execution completed
    deactivate HCCL_H
```

The main responsibilities of each phase are as follows:

1. **Define the operator API**: Specify the input and output, data size, data type, communicator, execution stream, and other information.
2. **Query topology information**: Obtain the number of ranks, topology levels, intra-level connections, and available links to provide a basis for algorithm selection and resource computation.
3. **Select algorithms**: Select a registered algorithm implementation based on conditions such as the operator type, execution engine, topology, data size, and data type. This step can be omitted for custom operators that have only one fixed implementation.
4. **Create resources**: Compute and allocate threads, Notify resources, channels, communication memory, and resource context, and copy the context required for AI CPU execution to the device.
5. **Dispatch kernels**: The host establishes a startup synchronization relationship with the AI CPU control thread, and then dispatches the AI CPU kernel to the execution stream.
6. **Orchestrate tasks**: After the AI CPU kernel starts and obtains the resource context, it calls the algorithm execution logic to orchestrate operations such as thread synchronization, channel synchronization, and data movement onto the corresponding threads.
7. **Complete synchronization**: After the AI CPU completes orchestration, it notifies the host. The host waits for this notification to ensure the execution order between the communication task and the service flow.
