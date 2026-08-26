# Defining Operator APIs

<!-- md-trans-meta sourceCommit=4b3acc1183ff175f340b0421dbe1faf9a723a585 translatedAt=2026-08-11T07:16:05.874Z pushedAt=2026-08-20T11:39:14.571Z -->

Developers need to first create an operator API definition header file based on the functionality of the communication operator. Subsequent calls to this communication operator depend on this header file.

Take the custom communication operator AllGather as an example. This type of operator API requires the source address, destination address, source data size, data type, as well as the communicator and stream information. Its API definition is as follows:

```c
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream);
```
