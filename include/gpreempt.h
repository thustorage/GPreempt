#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

#ifdef CUDA

typedef unsigned char      NvV8; /* "void": enumerated or multiple fields    */
typedef unsigned char      NvU8; /* 0 to 255                                 */
typedef   signed char      NvS8; /* -128 to 127                              */
typedef unsigned __INT16_TYPE__ NvV16; /* "void": enumerated or multiple fields */
typedef unsigned __INT16_TYPE__ NvU16; /* 0 to 65535                            */
typedef   signed __INT16_TYPE__ NvS16; /* -32768 to 32767                       */
typedef unsigned __INT32_TYPE__ NvV32; /* "void": enumerated or multiple fields */
typedef unsigned __INT32_TYPE__ NvU32; /* 0 to 4294967295                       */
typedef   signed __INT32_TYPE__ NvS32; /* -2147483648 to 2147483647             */
typedef unsigned long long NvU64; /* 0 to 18446744073709551615                      */
typedef          long long NvS64; /* -9223372036854775808 to 9223372036854775807    */
typedef void*              NvP64; /* 64 bit void pointer                     */
typedef NvU32 NvHandle;
typedef NvU8 NvBool;
#define NV_TRUE           ((NvBool)(0 == 0))
#define NV_FALSE          ((NvBool)(0 != 0))


typedef NvU32 NV_STATUS;
#define NV_OK          0x00000000
#define NV_ERR_GENERIC 0x0000FFFF

#define NV_DECLARE_ALIGNED(TYPE_VAR, ALIGN) TYPE_VAR __attribute__ ((aligned (ALIGN)))
#define NV_ALIGN_BYTES(size) __attribute__ ((aligned (size)))

typedef struct
{
    NvHandle hClient;
    NvHandle hObject;
    NvV32    cmd;
    NvU32    flags;
    NvP64    params NV_ALIGN_BYTES(8);
    NvU32    paramsSize;
    NvV32    status;
} NVOS54_PARAMETERS;

#define NVA06C_CTRL_CMD_SET_TIMESLICE (0xa06c0103)

typedef struct NVA06C_CTRL_TIMESLICE_PARAMS {
    NV_DECLARE_ALIGNED(NvU64 timesliceUs, 8);
} NVA06C_CTRL_TIMESLICE_PARAMS;

#define NVA06C_CTRL_CMD_PREEMPT (0xa06c0105)

typedef struct NVA06C_CTRL_PREEMPT_PARAMS {
    NvBool bWait;
    NvBool bManualTimeout;
    NvU32  timeoutUs;
} NVA06C_CTRL_PREEMPT_PARAMS;

#define NVA06F_CTRL_CMD_GPFIFO_SCHEDULE (0xa06f0103)

typedef struct NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS {
    NvBool bEnable;
    NvBool bSkipSubmit;
} NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS;

#define NVA06F_CTRL_CMD_RESTART_RUNLIST (0xa06f0111)

typedef struct NVA06F_CTRL_RESTART_RUNLIST_PARAMS {
    NvBool bForceRestart;
    NvBool bBypassWait;
} NVA06F_CTRL_RESTART_RUNLIST_PARAMS;

#define NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL (0xa06c0107)

typedef struct NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS {
    NvU32 tsgInterleaveLevel;
} NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;

#define NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS         (0x2080110b)
#define NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES (64)

typedef struct NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS {
    NvBool   bDisable;
    NvU32    numChannels;
    NvBool   bOnlyDisableScheduling;
    NvBool   bRewindGpPut;
    NV_DECLARE_ALIGNED(NvP64 pRunlistPreemptEvent, 8);
    // C form:  NvHandle hClientList[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES]
    NvHandle hClientList[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES];
    // C form:  NvHandle hChannelList[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES]
    NvHandle hChannelList[NV2080_CTRL_FIFO_DISABLE_CHANNELS_MAX_ENTRIES];
} NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS;

#define NV2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY                     (0x20801115)
#define NV2080_CTRL_FIFO_RUNLIST_SCHED_POLICY_CHANNEL_INTERLEAVED         0x1

typedef struct NV2080_CTRL_FIFO_RUNLIST_SET_SCHED_POLICY_PARAMS {
    NvU32 flags;
    NvU32 schedPolicy;
} NV2080_CTRL_FIFO_RUNLIST_SET_SCHED_POLICY_PARAMS;

#define OP_ALLOC   0xc020462b
#define OP_CONTROL 0xc020462a
#define OP_FREE    0xc0204629
#define OP_QUERY   0xc0204660

#define NV_HSUBDEVICE 0x5c000003

typedef NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS NvChannels;


struct NvContext {
    NvHandle hClient; 
    NvHandle hObject;
    NvChannels channels;
};

#define NVRMCHECK(cmd) do \
    {                      \
        NV_STATUS status = cmd;       \
        if (status != NV_OK)  \
            printf("%s fail with status 0x%x\n", #cmd, status);  \
    } while(0)

NV_STATUS NvRmControl(
    NvHandle hClient, 
    NvHandle hObject, 
    NvU32 cmd, 
    NvP64 params, 
    NvU32 paramsSize
);


NV_STATUS NvRmQuery(
    NvContext *pContext
);

NV_STATUS NvRmModifyTS(
    NvContext ctx,
    NvU64 timesliceUs
);

NV_STATUS NvRmPreempt(
    NvContext ctx
);

NV_STATUS NvRmGPFIFOSch(
    NvContext ctx, 
    NvBool bEnable
);

NV_STATUS NvRmRestartRunlist(
    NvContext ctx
);

NV_STATUS NvRmDisableCh(
    std::vector<NvContext> ctxs,
    NvBool bDisable
);

NV_STATUS NvRmSetPolicy(
    NvContext ctx
);

NV_STATUS NvRmSetLevel(
    NvContext ctx,
    NvU32 level
);

/*
- priority = 0: high priority
- priority = 1: low priority
*/ 
int set_priority(NvContext ctx, int priority);

#else

#include <linux/kfd_ioctl.h>

struct kfd_ioctl_wave_reset_args {
    __u32 gpu_id;
};

struct kfd_ioctl_dbg_trap_suspend_queues_args {
    __u64 exception_mask;
    __u64 queue_array_ptr;
    __u32 num_queues;
    __u32 grace_period;
    __u32 type;
};

int hipGetFd();
int hipResetWavefronts(int fd);
int hipSuspendStreams(int fd, std::vector<int> &stream_ids);
int hipResumeStreams(int fd, std::vector<int> &stream_ids);

#endif