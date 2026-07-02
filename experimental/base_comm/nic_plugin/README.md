# Experimental Base Comm NIC plugins

This directory contains experimental Base Comm NIC plugin sources for HOST RoCE and HOST UB generic-server use.

The plugins intentionally are not wired into the existing `libhcomm.so` build or public C API dispatch path. Build them as independent shared objects from this directory and load/deploy them through a future core loader once core changes are allowed.

Host-only adaptations in the copied sources:

- HOST resource id is fixed to `0`.
- HOST endpoint and channel paths avoid device runtime queries in the generic-server path.
- Device-scoped HCCP and TP manager initialization is not performed in these copied HOST endpoint paths.
- The plugin C ABI directly creates and calls the copied endpoint/channel objects instead of calling back through the existing hcomm C API.
