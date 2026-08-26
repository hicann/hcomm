# HCOMM API Overview

<!-- md-trans-meta sourceCommit=3af629f1371ba4d4f31764d28c389ab8615c7882 translatedAt=2026-08-14T10:54:30.755Z pushedAt=2026-08-19T01:06:31.886Z -->

- [Public Header Files and Library Files](./header_and_lib.md): Lists the header files and library files officially released by HCOMM for public use, serving as a reference for communication library/operator development and deployment.
- [Communicator Creation and Management APIs (C Language)](./comm_mgr_c/README.md): Used to implement framework adaptation in single-operator mode and implement distributed capabilities.
- [Communicator Creation and Management APIs (Python Language)](./comm_mgr_python/README.md): Used to implement framework adaptation in graph mode. Currently, they are used only for distributed optimization of TensorFlow networks on NPUs.
- [Communication Operator Development APIs](./comm_opdev/README.md): Provides control plane APIs and data plane APIs to allow developers to customize communication operators.
