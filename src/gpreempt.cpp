#include "gpreempt.h"
#include <stdio.h>

#ifdef CUDA

thread_local int fd = -1;

NV_STATUS NvRmControl(
    NvHandle hClient, 
    NvHandle hObject, 
    NvU32 cmd, 
    NvP64 params, 
    NvU32 paramsSize
) {
    if (fd < 0) {
        fd = open("/dev/nvidiactl", O_RDWR);
        if (fd < 0) {
            return NV_ERR_GENERIC;
        }
    }
    NVOS54_PARAMETERS controlArgs;
    controlArgs.hClient = hClient;
    controlArgs.hObject = hObject;
    controlArgs.cmd = cmd;
    controlArgs.params = params;
    controlArgs.paramsSize = paramsSize;
    controlArgs.flags = 0x0;
    controlArgs.status = 0x0;
    ioctl(fd, OP_CONTROL, &controlArgs);
    return controlArgs.status;
}

NV_STATUS NvRmQuery(
    NvContext *pContext
) {
    if (fd < 0) {
        fd = open("/dev/nvidiactl", O_RDWR);
        if (fd < 0) {
            return NV_ERR_GENERIC;
        }
    }
    NVOS54_PARAMETERS queryArgs;
    queryArgs.hClient = pContext->hClient;
    queryArgs.status = 0x0;
    queryArgs.params = (NvP64)&pContext->channels;
    ioctl(fd, OP_QUERY, &queryArgs);
    pContext->hClient = queryArgs.hClient;
    pContext->hObject = queryArgs.hObject;
    return queryArgs.status;
}

NV_STATUS NvRmModifyTS(
    NvContext ctx,
    NvU64 timesliceUs
) {
    NVA06C_CTRL_TIMESLICE_PARAMS timesliceParams0;
    timesliceParams0.timesliceUs = timesliceUs;
    return NvRmControl(ctx.hClient, ctx.hObject, NVA06C_CTRL_CMD_SET_TIMESLICE, (NvP64)&timesliceParams0, sizeof(timesliceParams0));
}

int set_priority(NvContext ctx, int priority) {
    NV_STATUS status;
    if (priority == 0){
        status = NvRmModifyTS(ctx, 1000000);
    } else {
        status = NvRmModifyTS(ctx, 1);
    }
    if (status != NV_OK) {
        return -1;
    }
    return 0;
}

NV_STATUS NvRmPreempt(
    NvContext ctx
) {
    NVA06C_CTRL_PREEMPT_PARAMS preemptParams;
    preemptParams.bWait = NV_FALSE;
    preemptParams.bManualTimeout = NV_FALSE;
    return NvRmControl(ctx.hClient, ctx.hObject, NVA06C_CTRL_CMD_PREEMPT, (NvP64)&preemptParams, sizeof(preemptParams));
}

// Not supported yet
NV_STATUS NvRmGPFIFOSch(
    NvContext ctx, 
    NvBool bEnable
) {
    NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS schParams;
    schParams.bEnable = bEnable;
    return NvRmControl(ctx.hClient, ctx.hObject, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, (NvP64)&schParams, sizeof(schParams));
    // return NvRmControl(ctx.hClient, ctx.channels.hChannelList[1], NVC56F_CTRL_CMD_GPFIFO_SCHEDULE, (NvP64)&schParams, sizeof(schParams));
}

NV_STATUS NvRmRestartRunlist(
    NvContext ctx
) {
    NVA06F_CTRL_RESTART_RUNLIST_PARAMS restartParams;
    restartParams.bForceRestart = NV_FALSE;
    restartParams.bBypassWait = NV_FALSE;
    return NvRmControl(ctx.hClient, ctx.channels.hChannelList[1], NVA06F_CTRL_CMD_RESTART_RUNLIST, (NvP64)&restartParams, sizeof(restartParams));
}

NV_STATUS NvRmDisableCh(
    std::vector<NvContext> ctxs,
    NvBool bDisable
) {
    if(!ctxs.size()) return NV_OK;
    NvChannels params;
    params.bDisable = bDisable;
    params.bOnlyDisableScheduling = NV_FALSE;
    params.pRunlistPreemptEvent = nullptr;
    params.bRewindGpPut = NV_FALSE;
    params.numChannels = 0;
    for(auto ctx : ctxs) {
        for(int i = 0; i < ctx.channels.numChannels; i++) {
            params.hClientList[params.numChannels] = ctx.channels.hClientList[i];
            params.hChannelList[params.numChannels] = ctx.channels.hChannelList[i];
            params.numChannels++;
        }
    }
    return NvRmControl(ctxs[0].hClient, NV_HSUBDEVICE, NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS, (NvP64)&params, sizeof(NvChannels));
}

NV_STATUS NvRmSetPolicy(
    NvContext ctx
) {
    NV2080_CTRL_FIFO_RUNLIST_SET_SCHED_POLICY_PARAMS policyParams = {0};
    policyParams.schedPolicy = NV2080_CTRL_FIFO_RUNLIST_SCHED_POLICY_CHANNEL_INTERLEAVED;
    return NvRmControl(ctx.hClient, NV_HSUBDEVICE, NV2080_CTRL_CMD_FIFO_RUNLIST_SET_SCHED_POLICY, (NvP64)&policyParams, sizeof(policyParams));
}

NV_STATUS NvRmSetLevel(
    NvContext ctx,
    NvU32 level
) {
    NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS levelParams;
    levelParams.tsgInterleaveLevel = level;
    return NvRmControl(ctx.hClient, ctx.hObject, NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL, (NvP64)&levelParams, sizeof(levelParams));
}


#else
int hipGetFd() {
    int fd = open("/dev/kfd", O_RDWR);
    if(fd < 0) {
        printf("open failed: %d\n", fd);
        return -1;
    }
    return fd;
}

int hipResetWavefronts(int fd) {
    kfd_ioctl_wave_reset_args reset_args;
    reset_args.gpu_id = 0x677d;
    int res = ioctl(fd, _IOWR('K', 0x86, kfd_ioctl_wave_reset_args), &reset_args);
    if(res < 0) {
        printf("ioctl failed: %d\n", res);
        perror("reset failed");
    }

    return 0;
}

int hipSuspendStreams(int fd, std::vector<int> &stream_ids) {
    kfd_ioctl_dbg_trap_suspend_queues_args args;
    args.num_queues = stream_ids.size();
    args.grace_period = 0;
    args.exception_mask = 63;
    args.queue_array_ptr = (__u64)stream_ids.data();
    args.type = 0;

    int res = ioctl(fd, _IOWR('K', 0x88, kfd_ioctl_dbg_trap_suspend_queues_args), &args);
    // printf("res = %d\n", res);
    if(res < 0) {
        printf("ioctl failed: %d\n", res);
        perror("suspend failed");
    }
    return 0;
}

int hipResumeStreams(int fd, std::vector<int> &stream_ids) {
    kfd_ioctl_dbg_trap_suspend_queues_args args;
    args.num_queues = stream_ids.size();
    args.grace_period = 0;
    args.exception_mask = 63;
    args.queue_array_ptr = (__u64)stream_ids.data();
    args.type = 1;

    int res = ioctl(fd, _IOWR('K', 0x88, kfd_ioctl_dbg_trap_suspend_queues_args), &args);
    // printf("res = %d\n", res);
    if(res < 0) {
        printf("ioctl failed: %d\n", res);
        perror("reset failed");
    }
    return 0;
}

#endif