# Communication Engine

<!-- md-trans-meta sourceCommit=2fbbaef94c56ddd7890f5afabcd087a13b03e44f translatedAt=2026-08-11T07:19:04.659Z pushedAt=2026-08-20T11:39:14.578Z -->

HCCL can use different communication engines to implement communication operators. Different communication engines are suitable for different scenarios. The comparison results are as follows.

**Table 1** Applicable scenarios of different communication engines

| Communication Engine | Advantage | Constraint | Applicable Scenario | Supported Product Model |
| --- | --- | --- | --- | --- |
| AI CPU+TS | Does not occupy compute cores, offers high communication efficiency, and is suitable for large-data, high-bandwidth scenarios. | Has high static communication overhead and is not suitable for small-data communication scenarios. | High-bandwidth communication scenarios | Ascend 950PR/Ascend 950DT<br>Atlas A3 training products/Atlas A3 inference products |
| Host CPU+TS | Does not occupy compute cores. | Has high dispatch overhead that increases linearly with the number of tasks. | NA | Atlas A2 training products/Atlas A2 inference products |
| AIV | Low latency. | Communication occupies Vector compute cores, and multiple Vector compute cores are required to fully utilize the communication bandwidth. Communication operators compete with compute operators for compute core resources, which may cause mutual interference. | Low-latency communication scenarios | Ascend 950PR/Ascend 950DT |
| CCU | Reduces memory access bandwidth and compute core usage.<br>High bandwidth and low latency. | Limited by on-chip resources, the number of supported communicators is limited. | High-bandwidth, low-latency communication scenarios | Ascend 950PR/Ascend 950DT |

The following describes the task execution process of each communication engine.

## AI CPU+TS

The AI CPU submits communication operation-related tasks to the Task Scheduler (TS).

1. The host submits an AI CPU kernel to the task queue.
2. After being scheduled by the task scheduler, the AI CPU kernel is dispatched to the AI CPU for execution.
3. The AI CPU submits communication tasks to the task queue.
4. The communication task submitted by the AI CPU is scheduled by the scheduler to the executor for execution.

**Figure 1** AI CPU+TS scheduling
![](figures/aicpu_ts_schedule.png "AI-CPU+TS-scheduling")

## Host CPU+TS

The Host CPU submits communication operation-related tasks to the Task Scheduler (TS) on the device.

1. The host submits various operations during the communication process (including memory copy and synchronization operations) to the task queue.
2. The scheduler dispatches the tasks in the task queue to the corresponding executors for execution.

**Figure 2**  Host CPU+TS scheduling
![](figures/hostcpu_ts_schedule.png "Host-CPU+TS-scheduling")

## AIV

The execution logic and operation steps of the communication operator are executed by the Vector Core.

1. The host submits an AIV kernel to the task queue.
2. The AIV kernel is scheduled by the scheduler and then sent to the Vector Core for execution.
3. The Vector Core can use different protocols to complete data movement.

**Figure 3**  AIV communication
![](figures/aiv_communication.png "AIV-communication")

## CCU

The communication operator is executed by the Collective Communication Unit (CCU), as shown in the figure.

1. The host dispatches a CCU instruction sequence (consisting of instructions recognizable by the CCU) to the CCU instruction space, and submits the corresponding CCU kernel task to the task queue.
2. The CCU kernel is dispatched by the scheduler and then sent to the CCU for execution.
3. The CCU executes the corresponding instruction stream and uses Unified Remote Memory Access (URMA) to complete data movement.

**Figure 4** CCU acceleration
![](figures/ccu_communication.png "CCU-acceleration")
