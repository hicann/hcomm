# HCOMM Header Files and Library Files

<!-- md-trans-meta sourceCommit=c3fa08793fb3a1a3c977d2c5e33ac694fe4df7fd translatedAt=2026-08-14T10:54:30.412Z pushedAt=2026-08-19T01:26:04.761Z -->

Huawei Communication (HCOMM) is the communication base library of HCCL. It provides communicator and communication resource management capabilities and serves as the underlying foundation of the CANN collective communication stack. HCCL dynamically loads HCOMM APIs through dlsym. HCCL and HCOMM are compiled and versioned independently.

This section describes the header files and library files of the external HCOMM APIs.

## API Categories

External HCOMM APIs are organized by API layers, as described in the following table.

**Table 1** API categories

| API Category | Target | Description |
| --- | --- | --- |
| Communicator management (L2-comm) | AI framework layer | APIs for communicator initialization and destruction. |
| Resources and topology (L2-res) | Operator developers | APIs for querying resources and topology such as channels, memory, topology, and endpoints. |
| Basic primitives (L3-prim) | Operator/Communication library developers | Primitive types for data transfer and synchronization. |
| Basic resources (L3-res) | Communication library developers | APIs for managing resources such as endpoints, channels, and memory registrations. |
| CCU operator development | CCU operator developers | Resource object encapsulation and kernel launch APIs for the Collective Communication Unit (CCU). |

## Header Files and Library Files Required for API Call

After the firmware, driver, and CANN software packages are installed, you can reference the HCOMM API header files and library files when compiling and running an application.

The external HCOMM header files are located in the `hccl/`, `hcomm/`, and `hcomm/ccu/` subdirectories under the `${INSTALL_DIR}/include/` directory, and the library files are located in the `${INSTALL_DIR}/lib64/` directory. Replace `${INSTALL_DIR}` with the path where the CANN software files are stored after installation. For example, if you install the software as the root user, the default storage path is `/usr/local/Ascend/cann`.

> [!CAUTION] Caution
> The header files in the `include/` directory are stable external header files. The `pkg_inc/` directory contains the inter-package APIs between HCOMM and HCCL, GE, and other components. These APIs are installed in a separate inter-package directory and their stability is not guaranteed. Do not directly reference them in external services. When compiling an API program, reference the library files corresponding to the included header files. Referencing unnecessary .so files may cause version function exceptions or compatibility issues during later version upgrades.

The purposes of the header files in the `include` directory are described in the following table.

**Table 2** Header file list

| API Header File | Purpose | Corresponding Library File |
| --- | --- | --- |
| hccl/hccl_comm.h | Defines C APIs such as communicator initialization and destruction (weak symbols). | libhcomm.so |
| hccl/hccl_types.h | Defines the HCCL return code enumeration and basic types. | libhcomm.so |
| hccl/hccn_rping.h | Defines HCCN RPing (Remote Ping) network connectivity detection C APIs, including types such as HccnRpingCtx, HccnResult, and HccnRpingMode, as well as detection APIs. | libhcomm.so |
| hccl/hccl_rank_graph.h | Defines the communication topology, endpoint attributes, and heterogeneous networking enumeration. | libhcomm.so |
| hccl/hccl_ccu_res.h | Defines C APIs for querying the CCU instance handle within a communicator. | libhcomm.so |
| hccl/hccl_res.h | Defines resource structures and constants such as HCCL channel descriptions and memory handles. | libhcomm.so |
| hccl/hccl_sym_win.h | Defines symmetric memory window (Symmetric Window) access APIs. | libhcomm.so |
| hccl/hccl_launch.h | Defines P2P operator descriptions and launch-related structures. | libhcomm.so |
| hcomm/hcomm_primitives.h | Defines basic primitive types such as channel/thread handles and reduction operators, and provides data transfer and synchronization primitives. | libhcomm.so |
| hcomm/hcomm_res.h | Defines resource management C APIs such as endpoint/channel/memory registration. | libhcomm.so |
| hcomm/hcomm_res_defs.h | Defines the HCOMM ABI version, handles, and resource description structures. | libhcomm.so |
| hcomm/ccu/ccu_primitives.hpp | CCU primitive aggregation header, including type aliases and resource creation entry points. | libhcomm.so |
| hcomm/ccu/ccu_launch.h | Defines C APIs for CCU kernel registration and launch (weak symbols). | libhcomm.so |
| hcomm/ccu/ccu_res.h | Defines the C API for querying the memory CCU access token (HcommCcuGetMemToken). | libhcomm.so |
| hcomm/ccu/ccu_types.h | Defines basic types such as CCU return codes and condition types. | libhcomm.so |
| hcomm/ccu/ccu_control_flow_macro.h | Defines control flow macros such as CCU while loops. | libhcomm.so |
| hcomm/ccu/ccu_address.hpp | CCU address object encapsulation (Address class). | libhcomm.so |
| hcomm/ccu/ccu_local_addr.hpp | CCU local address object encapsulation (LocalAddr class). | libhcomm.so |
| hcomm/ccu/ccu_remote_addr.hpp | CCU remote address object encapsulation (RemoteAddr class). | libhcomm.so |
| hcomm/ccu/ccu_buffer.hpp | CCU buffer resource object encapsulation (CcuBuffer class). | libhcomm.so |
| hcomm/ccu/ccu_array.hpp | CCU array template, a contiguous container for resources, used for batch allocation. | libhcomm.so |
| hcomm/ccu/ccu_event.hpp | CCU event resource object encapsulation (Event class). | libhcomm.so |
| hcomm/ccu/ccu_variable.hpp | CCU variable resource object encapsulation (Variable class). | libhcomm.so |
| hcomm/ccu/ccu_loop.hpp | CCU loop structure object encapsulation. | libhcomm.so |
| hcomm/ccu/ccu_func.hpp | Utility template that wraps a lambda as a CCU Func. | libhcomm.so |
| hcomm/ccu/ccu_utils.hpp | Internal auxiliary definitions such as CCU exception classes and operator utilities. | libhcomm.so |

- For details about the prototype definitions, parameter descriptions, and constraints of communicator management APIs, see [Communicator Creation and Management APIs (C Language)](./comm_mgr_c/README.md).

- For details about the prototype definitions, parameter descriptions, and constraints of communication operator development APIs, see [Communication Operator Development APIs](./comm_opdev/README.md).
