#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ncclComm *ncclComm_t;

typedef enum {
    ncclSuccess = 0,
} ncclResult_t;

typedef enum {
    ncclInt8 = 0,
    ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclHalf = 6,
    ncclFloat32 = 7,
    ncclFloat = 7,
    ncclFloat64 = 8,
    ncclDouble = 8,
    ncclBfloat16 = 9,
} ncclDataType_t;

typedef enum {
    ncclSum = 0,
} ncclRedOp_t;

const char *ncclGetErrorString(ncclResult_t result);
ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);
ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream);

#ifdef __cplusplus
}
#endif
