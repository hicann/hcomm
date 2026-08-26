# Before You Start

<!-- md-trans-meta sourceCommit=ffc5b1340e1ba3599f61afdda8ea78f73c8e36d1 translatedAt=2026-08-14T08:58:22.704Z pushedAt=2026-08-15T07:58:45.676Z -->

> [!NOTE] Note
> "Zero-copy" is a trial feature and may change in the future. It cannot be used in production environments.

## What Is Zero-Copy

In single-operator mode, the input and output buffers of an operator change dynamically. To avoid registering the memory of both processes for each communication, HCCL constructs a buffer inside the communicator as an intermediary to complete collective communication, which introduces additional memory copy overhead.

To reduce the memory copy overhead described above and allow HCCL to directly operate on the memory passed in by the service without using an intermediary HCCL buffer, HCCL provides the APIs described in this section to implement zero-copy.

## General Constraints

The zero-copy feature currently has the following usage restrictions and constraints:

- Only Atlas A3 training products/Atlas A3 inference products are supported.
- The related APIs can be called only from the TorchNPU plugin backend code. Other scenarios are not supported.
- Only the scenario where the communication operator expansion mode is AI CPU is supported.
- Only the collective communication operators AllGather, ReduceScatter, Broadcast, and AllReduce are supported. Among them, after the zero-copy feature is enabled for the ReduceScatter and AllReduce operators, the user's input memory will be modified.
- The input and output memory of the operators must be active memory.
- When enabling the zero-copy feature, developers are advised to use a larger communicator (that is, the communicator covering the maximum number of devices) for virtual address setup and activation. In this communicator, all processes can implement zero-copy when calling collective communication operators (communicators for which virtual addresses are not set up and activated cannot implement zero-copy).
