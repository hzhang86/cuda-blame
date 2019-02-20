/*
 * Copyright 2010-2017 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain profiler event values
 * using callbacks for CUDA runtime APIs
 *
 */

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#include "libunwind.h"
#include <sys/time.h>

#define UNW_LOCAL_ONLY

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

#define COMPUTE_N 50000
// Define some global variables
FILE *cpuFile, *gpuFile;
static uint64_t startTimestamp;

// Gpu wait
__device__ void gpu_sleep(clock_t cycles)
{ 
  clock_t start = clock64();
  clock_t elapsed = 0;
  while (elapsed < cycles) {
    elapsed = clock64() - start;
  }
}

// Kernels
__global__ void 
VecAdd(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
  gpu_sleep(N);
}

__global__ void 
VecSub(const int* A, const int* B, int* C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] - B[i];
  gpu_sleep(N);
}

static void
initVec(int *vec, int n)
{
  for (int i=0; i< n; i++)
    vec[i] = i;
}

static void
do_pass(cudaStream_t stream)
{
  int *h_A, *h_B, *h_C;
  int *d_A, *d_B, *d_C;
  size_t size = COMPUTE_N * sizeof(int);
  int threadsPerBlock = 256;
  int blocksPerGrid = 0;
  
  // Allocate input vectors h_A and h_B in host memory
  // don't bother to initialize
  h_A = (int*)malloc(size);
  h_B = (int*)malloc(size);
  h_C = (int*)malloc(size);
  
  // Initialize input vectors
  initVec(h_A, COMPUTE_N);
  initVec(h_B, COMPUTE_N);
  memset(h_C, 0, size);

  // Allocate vectors in device memory
  RUNTIME_API_CALL(cudaMalloc((void**)&d_A, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_B, size));
  RUNTIME_API_CALL(cudaMalloc((void**)&d_C, size));

  RUNTIME_API_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
  RUNTIME_API_CALL(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

  blocksPerGrid = (COMPUTE_N + threadsPerBlock - 1) / threadsPerBlock;
  VecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
  //VecAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
  VecSub<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, COMPUTE_N);
    
  RUNTIME_API_CALL(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

  if (stream == 0)
    RUNTIME_API_CALL(cudaDeviceSynchronize());
  else
    RUNTIME_API_CALL(cudaStreamSynchronize(stream));

  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

// Output CPU side stacktrace before each cudaLaunch
void CUPTIAPI
getStackTraceCallback(void *userdata, CUpti_CallbackDomain domain,
                      CUpti_CallbackId cbid, const void *cbdata)
{
  const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;
     
  // This callback is enabled only for launch so we shouldn't see anything else.
  if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
    printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
    exit(-1);
  }

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    // Get the CPU stacktrace to this point
    unw_word_t ip;
    unw_cursor_t cursor;
    unw_context_t uc;
    int count;
    char funcName[128];

    uint32_t correlationId = cbInfo->correlationId;
    fprintf(cpuFile, "<----START cpuStack %u\n", correlationId);
    
    unw_getcontext(&uc);
    if (unw_init_local(&cursor, &uc) <0)
      fprintf(stderr, "unw_init_local failed\n");
    count = 0;
    while (unw_step(&cursor) > 0) {
      unw_get_reg(&cursor, UNW_REG_IP, &ip);
      unw_get_proc_name(&cursor, funcName, sizeof(funcName), NULL);
      fprintf(cpuFile, "%d 0x%016lx %s\t", count, (unsigned long)ip, funcName);
      count++;
    }

    fprintf(cpuFile, "\n---->END\n");
  }
    
  else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    // Currently nothing needed to be done here
  }
}

// Output to GPU.txt
static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
    case CUPTI_ACTIVITY_KIND_DEVICE:
    {
      CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *) record;
      fprintf(gpuFile, "DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
             "multiprocessors %u, clock %u MHz\n",
             device->name, device->id,
             device->computeCapabilityMajor, device->computeCapabilityMinor,
             (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
             (unsigned int) (device->globalMemorySize / 1024 / 1024),
             device->numMultiprocessors, (unsigned int) (device->coreClockRate / 1000));
      break;
    }
    case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
    {
      CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;
      fprintf(gpuFile, "DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
             attribute->attribute.cupti, attribute->deviceId, (unsigned long long)attribute->value.vUint64);
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      //CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *) record; //cuda 8.0 doesn't have *Kernel4 yet, we use *Kernel3 struct
      CUpti_ActivityKernel3 *kernel = (CUpti_ActivityKernel3 *) record;
      fprintf(gpuFile, "%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             kindString,
             kernel->name,
             (unsigned long long) (kernel->start - startTimestamp),
             (unsigned long long) (kernel->end - startTimestamp),
             kernel->deviceId, kernel->contextId, kernel->streamId,
             kernel->correlationId);
      fprintf(gpuFile, "    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
             kernel->gridX, kernel->gridY, kernel->gridZ,
             kernel->blockX, kernel->blockY, kernel->blockZ,
             kernel->staticSharedMemory, kernel->dynamicSharedMemory);
      break;
    }
    case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      fprintf(gpuFile, "DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId);
      break;
    }
    case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      fprintf(gpuFile, "RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId);
      break;
    }
    case CUPTI_ACTIVITY_KIND_SOURCE_LOCATOR:
    {
      CUpti_ActivitySourceLocator *srcLocator = (CUpti_ActivitySourceLocator *)record;
      fprintf(gpuFile, "SOURCE_LOCATOR Id %d File %s Line %d\n", 
             srcLocator->id, srcLocator->fileName, srcLocator->lineNumber);
      break;
    }
    case CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION:
    {
      CUpti_ActivityInstructionExecution *inst_executed = (CUpti_ActivityInstructionExecution *)record;
      fprintf(gpuFile, "INSTRUCTION_EXECUTION srcLctr %u corrId %u funcId %u pc 0x%llx\n",
             inst_executed->sourceLocatorId, inst_executed->correlationId, 
             inst_executed->functionId, inst_executed->pcOffset);
      // number of threads that executed this instruction and number of times the instruction was executed
      //fprintf(gpuFile, "notPredOffthread_inst_executed %llu, thread_inst_executed %llu, inst_executed %u\n\n",
      //       (unsigned long long)inst_executed->notPredOffThreadsExecuted, 
      //       (unsigned long long)inst_executed->threadsExecuted, inst_executed->executed);
      break;
    }
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
    {
      CUpti_ActivityPCSampling *psRecord = (CUpti_ActivityPCSampling*)record;
      fprintf(gpuFile, "PC_SAMPLING srcLctr %u corrId %u funcId %u pc 0x%llx\n"/*, samples %u\n"*/,
             psRecord->sourceLocatorId, psRecord->correlationId, psRecord->functionId,
             (unsigned long long)psRecord->pcOffset/*, psRecord->samples*/);
      break;
    }
    case CUPTI_ACTIVITY_KIND_FUNCTION:
    {
      CUpti_ActivityFunction *fResult = (CUpti_ActivityFunction *)record;
      fprintf(gpuFile, "FUNCTION Id %u ctx %u moduleId %u functionIndex %u name %s\n",
             fResult->id, fResult->contextId, fResult->moduleId, fResult->functionIndex, fResult->name);
      break;
    }
    default:
      //fprintf(gpuFile, "  <unknown>\n");
      break;
  }
}

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
      }
      else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int) dropped);
    }
  }

  free(buffer);
}

void
initTrace()
{
  size_t attrValue = 0, attrValueSize = sizeof(size_t);
  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME)); 
  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));

  // Register callbacks for buffer requests and for buffers completed by CUPTI.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  // Get and set activity attributes.
  // Attributes can be set by the CUPTI client to change behavior of the activity API.
  // Some attributes require to be set before any CUDA context is created to be effective,
  // e.g. to be applied to all device buffer allocations (see documentation).
  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
  attrValue *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
  attrValue *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));

  //CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

int main(int argc, char *argv[])
{
  CUcontext context = 0;
  CUdevice dev = 0;
  int computeCapabilityMajor=0;
  int computeCapabilityMinor=0;
  int deviceNum;
  int deviceCount;
  char deviceName[32];
  CUpti_SubscriberHandle subscriber;
  CUpti_ActivityPCSamplingConfig configPC;

  initTrace(); // We need to initialize activityAPI before any CUDA calls
 
  cpuFile = fopen("CPU.txt", "w");
  gpuFile = fopen("GPU.txt", "w");

  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));

  if (deviceCount == 0) {
    printf("There is no device supporting CUDA.\n");
    return -2;
  }

  if (argc > 1)
    deviceNum = atoi(argv[1]);
  else
    deviceNum = 0;
  printf("CUDA Device Number: %d\n", deviceNum);

  DRIVER_API_CALL(cuDeviceGet(&dev, deviceNum));
  DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, dev));

  printf("CUDA Device Name: %s\n", deviceName);

  DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  DRIVER_API_CALL(cuCtxCreate(&context, 0, dev));

  CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getStackTraceCallback, NULL));
  CUPTI_CALL(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
  configPC.samplingPeriod=CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MIN; //..MIN->LOW->MID->HIGH->MAX
  //configPC.samplingPeriod2 = 29; //enable this could overwrite the above perid to "2^29"
  CUPTI_CALL(cuptiActivityConfigurePCSampling(context, &configPC));

  do_pass(0);
    
  CUPTI_CALL(cuptiActivityFlushAll(0));
  CUPTI_CALL(cuptiUnsubscribe(subscriber));

  fclose(cpuFile);
  fclose(gpuFile);
  
  return 0;
}

