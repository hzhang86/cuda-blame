/*
 * Author: Hui Zhang
 * Modified based on Sangamesh Ragate's work
 * Date : 27th Sep 2017
 * UMD-CS
 * Description : This is the shared library that sets up the environent 
 * for the cuda application by creating the context and keeping it ready
 * to perform Exution Instrunction (sampling) of the cuda application as soon as it launces the kernel
 */


#include "inst_sampling.h"
//libunwind MACRO for local unwind optimization
#define UNW_LOCAL_ONLY

// Define some global variables
static FILE *cpuFile, *gpuFile;
static uint64_t startTimestamp;
static CUpti_SubscriberHandle g_subscriber;
static CUpti_ActivityPCSamplingConfig configPC;
static CUcontext cuCtx;
static int deviceNum = 0;
static cudaDeviceProp prop;
static CUdevice dev = 0;
static bool pc_sampling = false;
static bool context_created = false;

// Output CPU side stacktrace before each cudaLaunch
void CUPTIAPI getStackTraceCallback(void *userdata, CUpti_CallbackDomain domain,
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
    //char funcName[128];

    uint32_t correlationId = cbInfo->correlationId;
    fprintf(cpuFile, "<----START cpuStack %u\n", correlationId);
    
    unw_getcontext(&uc);
    if (unw_init_local(&cursor, &uc) <0)
      fprintf(stderr, "unw_init_local failed\n");
    count = 0;
    while (unw_step(&cursor) > 0) {
      unw_get_reg(&cursor, UNW_REG_IP, &ip);
      //unw_get_proc_name(&cursor, funcName, sizeof(funcName), NULL);
      //fprintf(cpuFile, "%d 0x%016lx %s\t", count, (unsigned long)ip, funcName);
      fprintf(cpuFile, "%d 0x%016lx\t", count, (unsigned long)ip);
      count++;
    }

    fprintf(cpuFile, "\n---->END\n");
  }
    
  else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    // Currently nothing needed to be done here
  }
}

// Output to GPU.txt
void printActivity(CUpti_Activity *record)
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
      fprintf(gpuFile, "INST_EXEC srcLctr %u corrId %u funcId %u\n",
             inst_executed->sourceLocatorId, inst_executed->correlationId, inst_executed->functionId);
      // number of threads that executed this instruction and number of times the instruction was executed
      //fprintf(gpuFile, "notPredOffthread_inst_executed %llu, thread_inst_executed %llu, inst_executed %u\n\n",
      //       (unsigned long long)inst_executed->notPredOffThreadsExecuted, 
      //       (unsigned long long)inst_executed->threadsExecuted, inst_executed->executed);
      break;
    }
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
    {
      CUpti_ActivityPCSampling *psRecord = (CUpti_ActivityPCSampling*)record;
      fprintf(gpuFile, "PC_SAMPLING srcLctr %u corrId %u funcId %u\n"/*, samples %u\n"*/,
             psRecord->sourceLocatorId, psRecord->correlationId, psRecord->functionId);
      break;
    }
    case CUPTI_ACTIVITY_KIND_FUNCTION:
    {
      CUpti_ActivityFunction *fResult = (CUpti_ActivityFunction *)record;
      fprintf(gpuFile, "FUNCTION id %u ctx %u moduleId %u functionIndex %u name %s\n",
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

void initTrace()
{
  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGet(&dev, deviceNum));
  RUNTIME_API_CALL(cudaGetDeviceProperties(&prop, deviceNum));
  printf("Device compute capability: %d.%d\n", prop.major, prop.minor);
  if ((prop.major == 5 && prop.minor == 2) || prop.major >= 6) {
    printf("We use PC_SAMPLING in sampling.\n");
    pc_sampling = true;  
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));
    //configure pc_sampling rate
    configPC.samplingPeriod = CUPTI_ACTIVITY_PC_SAMPLING_PERIOD_MAX; //..MIN->LOW->MID->HIGH->MAX
    //configPC.samplingPeriod2 = 29; //enable this could overwrite the above perid to "2^29"
    //DRIVER_API_CALL(cuCtxGetCurrent(&cuCtx));
    //if (cuCtx == NULL) {
      printf("We called cuCtxCreate.\n");
      context_created = true;
      DRIVER_API_CALL(cuCtxCreate(&cuCtx, 0, dev));	
    //}   
    CUPTI_CALL(cuptiActivityConfigurePCSampling(cuCtx, &configPC));
  }
  else {
    printf("We use INSTRUCTION_EXECUTION in sampling.\n");
    pc_sampling = false;
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION));
  }

  //Register GPU tracer
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

  //Register CPU tracer
  CUPTI_CALL(cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)getStackTraceCallback, NULL));
  CUPTI_CALL(cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));

  //open output files
  cpuFile = fopen("CPU.txt", "w");
  gpuFile = fopen("GPU.txt", "w");
  printf("Done initTrace \n"); //wait for all cupti api calls end
}

void finiTrace()
{
  CUPTI_CALL(cuptiActivityFlushAll(0));
  CUPTI_CALL(cuptiUnsubscribe(g_subscriber));
  if (pc_sampling) 
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));
  else 
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_INSTRUCTION_EXECUTION));
  //CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  if (context_created)
    DRIVER_API_CALL(cuCtxDestroy(cuCtx));	
     
  fclose(cpuFile);
  fclose(gpuFile);
  printf("Done finiTrace \n"); //wait for all cupti api calls end
}

