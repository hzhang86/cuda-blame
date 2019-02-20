/*
 * Author: Hui Zhang
 * Modified based on Sangamesh Ragate's work
 * Date : 27th Sep 2017
 * UMD-CS
 * Description : This is the shared library that sets up the environent 
 * for the cuda application by creating the context and keeping it ready
 * to perform Exution Instrunction (sampling) of the cuda application as soon as it launces the kernel
 */



#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static CUpti_SubscriberHandle g_subscriber;
static FILE *samp_file;


#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                      \
do {                                                                          \
    CUptiResult _status = call;                                               \
    if (_status != CUPTI_SUCCESS) {                                           \
        const char *errstr;                                                   \
        cuptiGetResultString(_status, &errstr);                               \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
                __FILE__, __LINE__, #call, errstr);                           \
        exit(-1);                                                             \
    }                                                                         \
} while (0)

#define BUF_SIZE (32 * 16384)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer)) 

/*
static char* stall_name[12];
static int val[12]={0};
    
static const char *
getStallReasonString(CUpti_ActivityPCSamplingStallReason reason,unsigned int samples)
{
    switch (reason) {
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INVALID:
        stall_name[0]="Stall_invalid";
        val[0] += samples;
        return "Invalid";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NONE:
        stall_name[1]="Stall_none";
        val[1] += samples;
        return "Selected";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_INST_FETCH:
        stall_name[2]="Stall_inst_fetch";
        val[2] += samples;
        return "Instruction fetch";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_EXEC_DEPENDENCY:
        stall_name[3]="Stall_exec_dependency";
        val[3] += samples;
        return "Execution dependency";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_DEPENDENCY:
        stall_name[4]="Stall_mem_dependency";
        val[4] += samples;
        return "Memory dependency";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_TEXTURE:
        stall_name[5]="Stall_texture";
        val[5] += samples;
        return "Texture";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_SYNC:
        stall_name[6]="Stall_sync";
        val[6] += samples;
        return "Sync";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_CONSTANT_MEMORY_DEPENDENCY:
        stall_name[7]="Stall_const_mem_dependency";
        val[7] += samples;
        return "Constant memory dependency";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_PIPE_BUSY:
        stall_name[8]="Stall_pipe_busy";
        val[8] += samples;
        return "Pipe busy";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_MEMORY_THROTTLE:
        stall_name[9]="Stall_memory_throttle";
        val[9] += samples;
        return "Memory throttle";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_NOT_SELECTED:
        stall_name[10]="Stall_warp_not_selected";
        val[10] += samples;
        return "Warp Not selected";
    case CUPTI_ACTIVITY_PC_SAMPLING_STALL_OTHER:
        stall_name[11]="Stall_other";
        val[11] += samples;
        return "Other";
    default:
        break;
    }

    return NULL;
}
*/


static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind) {
    // The activity record for source locator contains the ID for the source path,
    // path for the file and the line number in the source.
    case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR:
    {
      CUpti_ActivitySourceLocator *sourceLocator = (CUpti_ActivitySourceLocator *)record;
      fprintf(samp_file, "SOURCE_LOCATOR SrcLctrId %d, File %s Line %d\n", sourceLocator->id, sourceLocator->fileName, sourceLocator->lineNumber);
      break;
    }
    // The activity record for instruction execution corresponds to a PC of the generated code, it contains the ID for source locator
    // the correlation ID of the kernel to which this record is associated, function ID and pc offset for the instruction.
    case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION:
    {
      CUpti_ActivityInstructionExecution *sourceRecord = (CUpti_ActivityInstructionExecution *)record;
      fprintf(samp_file, "INSTRUCTION_EXECUTION srcLctr %u, corr %u, functionId %u, pc %x\n",
        sourceRecord->sourceLocatorId, sourceRecord->correlationId, sourceRecord->functionId, 
        sourceRecord->pcOffset);
      // number of threads that executed this instruction and number of times the instruction was executed
      fprintf(samp_file, "notPredOffthread_inst_executed %llu, thread_inst_executed %llu, inst_executed %u\n\n",
        (unsigned long long)sourceRecord->notPredOffThreadsExecuted, 
        (unsigned long long)sourceRecord->threadsExecuted, sourceRecord->executed);
      break;
    }
    // function name and corresponding module information
    case CUPTI_ACTIVITY_KIND_FUNCTION:
    {
      CUpti_ActivityFunction *fResult = (CUpti_ActivityFunction *)record;
      fprintf(samp_file, "FUCTION functionId %u, moduleId %u, name %s\n",
        fResult->id,
        fResult->moduleId,
        fResult->name);
      break;
    }
    // Kernel activity records kernel information
    case CUPTI_ACTIVITY_KIND_KERNEL:
    {
      CUpti_ActivityKernel3 *kernel = (CUpti_ActivityKernel3 *)record;
      fprintf(samp_file, "\n\n************************************** KERNEL_RECORD_SUMMARY **********************************\n");
      fprintf(samp_file, "Kernel %s , device %d, context %d, correlation %d, stream %d,[start-end][%ld-%ld]\n\n",kernel->name, 
             kernel->deviceId,kernel->contextId,kernel->correlationId,kernel->streamId,kernel->start,kernel->end);
      break;
    }

    default:
      fprintf(samp_file, "  <unknown>\n");
      break;
  }
}

static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *b;

  *size = BUF_SIZE;
  b = (uint8_t *)malloc(*size + ALIGN_SIZE);
  if (*buffer == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *buffer = ALIGN_BUFFER(b, ALIGN_SIZE);
  *maxNumRecords = 0;
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  char host_name[128];
  gethostname(host_name, 127);
  samp_file = fopen(host_name, "a");

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
    if(status == CUPTI_SUCCESS) {
      printActivity(record);
    }
    else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }
    else {
      CUPTI_CALL(status);
    }
  } while (1);

  size_t dropped;
  CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
  if (dropped != 0) {
    printf("Dropped %u activity records\n", (unsigned int)dropped);
  }

  fclose(samp_file);
  free(buffer);
}

#define DUMP_CUBIN 1

void CUPTIAPI dumpCudaModule(CUpti_CallbackId cbid, void *resourceDescriptor)
{
#if DUMP_CUBIN
  const char *pCubin;
  size_t cubinSize;

  //dump the cubin at MODULE_LOADED_STARTING
  CUpti_ModuleResourceData *moduleResourceData = (CUpti_ModuleResourceData *)resourceDescriptor;
#endif
          
  if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
#if DUMP_CUBIN
    // You can use nvdisasm to dump the SASS from the cubin. 
    // Try nvdisasm -b -fun <function_id> sass_to_source.cubin
    pCubin = moduleResourceData->pCubin;
    cubinSize = moduleResourceData->cubinSize;
              
    FILE *cubin;
    cubin = fopen("sass_source_map.cubin", "wb");
    fwrite(pCubin, sizeof(uint8_t), cubinSize, cubin);
    fclose(cubin);
#endif
  }else if (cbid == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
    // You can dump the cubin either at MODULE_LOADED or MODULE_UNLOAD_STARTING
  }
}

static void
handleResource(CUpti_CallbackId cbid, const CUpti_ResourceData *resourceData)
{
  if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
    dumpCudaModule(cbid, resourceData->resourceDescriptor);
  }else if (cbid == CUPTI_CBID_RESOURCE_MODULE_UNLOAD_STARTING) {
    dumpCudaModule(cbid, resourceData->resourceDescriptor);
  }
}


static void CUPTIAPI
traceCallback(void *userdata, CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid, const void *cbdata)
{
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    handleResource(cbid, (CUpti_ResourceData *)cbdata);
  }
}


__attribute__((constructor)) void
initTrace()
{
  //get the arguments from the environment variables
  int deviceId;
  cudaDeviceProp g_deviceProp;
  CUcontext cuCtx;

  printf("In initTrace \n");
  if (getenv("GPU_DEVICE_ID") != NULL) 
    deviceId = atoi(getenv("GPU_DEVICE_ID"));
  else deviceId = 0;

  RUNTIME_API_CALL(cudaGetDeviceProperties(&g_deviceProp, deviceId));
  printf("Device Name: %s\n", g_deviceProp.name);
  if (g_deviceProp.major < 2) {
    printf("INSTRUCTION EXECUTION not supported on pre-Fermi devices\n");
    exit(-1);
  }
    
  cuInit(0);
  cuCtxCreate(&cuCtx,0,deviceId);
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION));
  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)traceCallback, NULL));
  CUPTI_CALL(cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE));
}

__attribute__((destructor)) void
finiTrace()
{
  CUPTI_CALL(cuptiActivityFlushAll(0));
  CUPTI_CALL(cuptiUnsubscribe(g_subscriber));
  CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION));
  //CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  printf("In finiTrace \n"); //wait for all cupti api calls end
}

