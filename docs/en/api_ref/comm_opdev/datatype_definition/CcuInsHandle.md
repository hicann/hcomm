# CcuInsHandle

<!-- md-trans-meta sourceCommit=0a8294da2b47d302f601034c095000b7c119376c translatedAt=2026-08-14T10:44:53.135Z pushedAt=2026-08-18T11:45:11.309Z -->

## Description

CCU instance handle, obtained from the HCCL communicator and used to identify a CCU instance. Subsequent kernel registration, translation, launch, and instance destruction operations are all performed through this handle.

## Prototype

```c
typedef uint64_t CcuInsHandle;
```
