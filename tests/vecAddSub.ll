; ModuleID = 'vecAddSub.bc'
source_filename = "vecAddSub.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.timeval = type { i64, i64 }
%struct.timezone = type { i32, i32 }
%struct.CUstream_st = type opaque
%struct.dim3 = type { i32, i32, i32 }

$_ZN4dim3C2Ejjj = comdat any

@.str = private unnamed_addr constant [36 x i8] c"\0AElapsed time         = %10.2f (s)\0A\00", align 1
@stderr = external global %struct._IO_FILE*, align 8
@.str.1 = private unnamed_addr constant [49 x i8] c"%s:%d: error: function %s failed with error %s.\0A\00", align 1
@.str.2 = private unnamed_addr constant [13 x i8] c"vecAddSub.cu\00", align 1
@.str.3 = private unnamed_addr constant [31 x i8] c"cudaMalloc((void**)&d_A, size)\00", align 1
@.str.4 = private unnamed_addr constant [31 x i8] c"cudaMalloc((void**)&d_B, size)\00", align 1
@.str.5 = private unnamed_addr constant [31 x i8] c"cudaMalloc((void**)&d_C, size)\00", align 1
@.str.6 = private unnamed_addr constant [64 x i8] c"cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream)\00", align 1
@.str.7 = private unnamed_addr constant [64 x i8] c"cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream)\00", align 1
@.str.8 = private unnamed_addr constant [64 x i8] c"cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream)\00", align 1
@.str.9 = private unnamed_addr constant [24 x i8] c"cudaDeviceSynchronize()\00", align 1
@.str.10 = private unnamed_addr constant [30 x i8] c"cudaStreamSynchronize(stream)\00", align 1

; Function Attrs: noinline uwtable
define void @_Z6VecAddPKiS0_Pii(i32* %A, i32* %B, i32* %C, i32 %N) #0 !dbg !656 {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  %C.addr = alloca i32*, align 8
  %N.addr = alloca i32, align 4
  store i32* %A, i32** %A.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %A.addr, metadata !662, metadata !663), !dbg !664
  store i32* %B, i32** %B.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %B.addr, metadata !665, metadata !663), !dbg !666
  store i32* %C, i32** %C.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %C.addr, metadata !667, metadata !663), !dbg !668
  store i32 %N, i32* %N.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %N.addr, metadata !669, metadata !663), !dbg !670
  %0 = bitcast i32** %A.addr to i8*, !dbg !671
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !671
  %2 = icmp eq i32 %1, 0, !dbg !671
  br i1 %2, label %setup.next, label %setup.end, !dbg !671

setup.next:                                       ; preds = %entry
  %3 = bitcast i32** %B.addr to i8*, !dbg !672
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !672
  %5 = icmp eq i32 %4, 0, !dbg !672
  br i1 %5, label %setup.next1, label %setup.end, !dbg !672

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast i32** %C.addr to i8*, !dbg !674
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !674
  %8 = icmp eq i32 %7, 0, !dbg !674
  br i1 %8, label %setup.next2, label %setup.end, !dbg !674

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast i32* %N.addr to i8*, !dbg !676
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 4, i64 24), !dbg !676
  %11 = icmp eq i32 %10, 0, !dbg !676
  br i1 %11, label %setup.next3, label %setup.end, !dbg !676

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (i32*, i32*, i32*, i32)* @_Z6VecAddPKiS0_Pii to i8*)), !dbg !678
  br label %setup.end, !dbg !678

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !680
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @cudaSetupArgument(i8*, i64, i64)

declare i32 @cudaLaunch(i8*)

; Function Attrs: noinline uwtable
define void @_Z6VecSubPKiS0_Pii(i32* %A, i32* %B, i32* %C, i32 %N) #0 !dbg !681 {
entry:
  %A.addr = alloca i32*, align 8
  %B.addr = alloca i32*, align 8
  %C.addr = alloca i32*, align 8
  %N.addr = alloca i32, align 4
  store i32* %A, i32** %A.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %A.addr, metadata !682, metadata !663), !dbg !683
  store i32* %B, i32** %B.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %B.addr, metadata !684, metadata !663), !dbg !685
  store i32* %C, i32** %C.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %C.addr, metadata !686, metadata !663), !dbg !687
  store i32 %N, i32* %N.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %N.addr, metadata !688, metadata !663), !dbg !689
  %0 = bitcast i32** %A.addr to i8*, !dbg !690
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !690
  %2 = icmp eq i32 %1, 0, !dbg !690
  br i1 %2, label %setup.next, label %setup.end, !dbg !690

setup.next:                                       ; preds = %entry
  %3 = bitcast i32** %B.addr to i8*, !dbg !691
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !691
  %5 = icmp eq i32 %4, 0, !dbg !691
  br i1 %5, label %setup.next1, label %setup.end, !dbg !691

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast i32** %C.addr to i8*, !dbg !693
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !693
  %8 = icmp eq i32 %7, 0, !dbg !693
  br i1 %8, label %setup.next2, label %setup.end, !dbg !693

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast i32* %N.addr to i8*, !dbg !695
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 4, i64 24), !dbg !695
  %11 = icmp eq i32 %10, 0, !dbg !695
  br i1 %11, label %setup.next3, label %setup.end, !dbg !695

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (i32*, i32*, i32*, i32)* @_Z6VecSubPKiS0_Pii to i8*)), !dbg !697
  br label %setup.end, !dbg !697

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !699
}

; Function Attrs: noinline norecurse uwtable
define i32 @main(i32 %argc, i8** %argv) #2 !dbg !700 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %start = alloca %struct.timeval, align 8
  %end = alloca %struct.timeval, align 8
  %elapsed_time = alloca double, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !703, metadata !663), !dbg !704
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !705, metadata !663), !dbg !706
  call void @llvm.dbg.declare(metadata %struct.timeval* %start, metadata !707, metadata !663), !dbg !716
  %call = call i32 @gettimeofday(%struct.timeval* %start, %struct.timezone* null) #8, !dbg !717
  call void @_Z9initTracev(), !dbg !718
  call void @_ZL7do_passP11CUstream_st(%struct.CUstream_st* null), !dbg !719
  call void @_Z9finiTracev(), !dbg !720
  call void @llvm.dbg.declare(metadata %struct.timeval* %end, metadata !721, metadata !663), !dbg !722
  %call1 = call i32 @gettimeofday(%struct.timeval* %end, %struct.timezone* null) #8, !dbg !723
  call void @llvm.dbg.declare(metadata double* %elapsed_time, metadata !724, metadata !663), !dbg !725
  %tv_sec = getelementptr inbounds %struct.timeval, %struct.timeval* %end, i32 0, i32 0, !dbg !726
  %0 = load i64, i64* %tv_sec, align 8, !dbg !726
  %tv_sec2 = getelementptr inbounds %struct.timeval, %struct.timeval* %start, i32 0, i32 0, !dbg !727
  %1 = load i64, i64* %tv_sec2, align 8, !dbg !727
  %sub = sub nsw i64 %0, %1, !dbg !728
  %conv = sitofp i64 %sub to double, !dbg !729
  %tv_usec = getelementptr inbounds %struct.timeval, %struct.timeval* %end, i32 0, i32 1, !dbg !730
  %2 = load i64, i64* %tv_usec, align 8, !dbg !730
  %tv_usec3 = getelementptr inbounds %struct.timeval, %struct.timeval* %start, i32 0, i32 1, !dbg !731
  %3 = load i64, i64* %tv_usec3, align 8, !dbg !731
  %sub4 = sub nsw i64 %2, %3, !dbg !732
  %conv5 = sitofp i64 %sub4 to double, !dbg !733
  %div = fdiv double %conv5, 1.000000e+06, !dbg !734
  %add = fadd double %conv, %div, !dbg !735
  store double %add, double* %elapsed_time, align 8, !dbg !725
  %4 = load double, double* %elapsed_time, align 8, !dbg !736
  %call6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str, i32 0, i32 0), double %4), !dbg !737
  ret i32 0, !dbg !738
}

; Function Attrs: nounwind
declare i32 @gettimeofday(%struct.timeval*, %struct.timezone*) #3

declare void @_Z9initTracev() #4

; Function Attrs: noinline uwtable
define internal void @_ZL7do_passP11CUstream_st(%struct.CUstream_st* %stream) #0 !dbg !739 {
entry:
  %stream.addr = alloca %struct.CUstream_st*, align 8
  %h_A = alloca i32*, align 8
  %h_B = alloca i32*, align 8
  %h_C = alloca i32*, align 8
  %d_A = alloca i32*, align 8
  %d_B = alloca i32*, align 8
  %d_C = alloca i32*, align 8
  %size = alloca i64, align 8
  %threadsPerBlock = alloca i32, align 4
  %blocksPerGrid = alloca i32, align 4
  %_status = alloca i32, align 4
  %_status7 = alloca i32, align 4
  %_status16 = alloca i32, align 4
  %_status25 = alloca i32, align 4
  %_status34 = alloca i32, align 4
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp42 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp42.coerce = alloca { i64, i32 }, align 4
  %agg.tmp44 = alloca %struct.dim3, align 4
  %agg.tmp45 = alloca %struct.dim3, align 4
  %agg.tmp44.coerce = alloca { i64, i32 }, align 4
  %agg.tmp45.coerce = alloca { i64, i32 }, align 4
  %_status51 = alloca i32, align 4
  %_status62 = alloca i32, align 4
  %_status71 = alloca i32, align 4
  store %struct.CUstream_st* %stream, %struct.CUstream_st** %stream.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.CUstream_st** %stream.addr, metadata !746, metadata !663), !dbg !747
  call void @llvm.dbg.declare(metadata i32** %h_A, metadata !748, metadata !663), !dbg !749
  call void @llvm.dbg.declare(metadata i32** %h_B, metadata !750, metadata !663), !dbg !751
  call void @llvm.dbg.declare(metadata i32** %h_C, metadata !752, metadata !663), !dbg !753
  call void @llvm.dbg.declare(metadata i32** %d_A, metadata !754, metadata !663), !dbg !755
  call void @llvm.dbg.declare(metadata i32** %d_B, metadata !756, metadata !663), !dbg !757
  call void @llvm.dbg.declare(metadata i32** %d_C, metadata !758, metadata !663), !dbg !759
  call void @llvm.dbg.declare(metadata i64* %size, metadata !760, metadata !663), !dbg !761
  store i64 200000, i64* %size, align 8, !dbg !761
  call void @llvm.dbg.declare(metadata i32* %threadsPerBlock, metadata !762, metadata !663), !dbg !763
  store i32 256, i32* %threadsPerBlock, align 4, !dbg !763
  call void @llvm.dbg.declare(metadata i32* %blocksPerGrid, metadata !764, metadata !663), !dbg !765
  store i32 0, i32* %blocksPerGrid, align 4, !dbg !765
  %0 = load i64, i64* %size, align 8, !dbg !766
  %call = call noalias i8* @malloc(i64 %0) #8, !dbg !767
  %1 = bitcast i8* %call to i32*, !dbg !768
  store i32* %1, i32** %h_A, align 8, !dbg !769
  %2 = load i64, i64* %size, align 8, !dbg !770
  %call1 = call noalias i8* @malloc(i64 %2) #8, !dbg !771
  %3 = bitcast i8* %call1 to i32*, !dbg !772
  store i32* %3, i32** %h_B, align 8, !dbg !773
  %4 = load i64, i64* %size, align 8, !dbg !774
  %call2 = call noalias i8* @malloc(i64 %4) #8, !dbg !775
  %5 = bitcast i8* %call2 to i32*, !dbg !776
  store i32* %5, i32** %h_C, align 8, !dbg !777
  %6 = load i32*, i32** %h_A, align 8, !dbg !778
  call void @_ZL7initVecPii(i32* %6, i32 50000), !dbg !779
  %7 = load i32*, i32** %h_B, align 8, !dbg !780
  call void @_ZL7initVecPii(i32* %7, i32 50000), !dbg !781
  %8 = load i32*, i32** %h_C, align 8, !dbg !782
  %9 = bitcast i32* %8 to i8*, !dbg !783
  %10 = load i64, i64* %size, align 8, !dbg !784
  call void @llvm.memset.p0i8.i64(i8* %9, i8 0, i64 %10, i32 4, i1 false), !dbg !783
  br label %do.body, !dbg !785, !llvm.loop !786

do.body:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %_status, metadata !787, metadata !663), !dbg !790
  %11 = bitcast i32** %d_A to i8**, !dbg !791
  %12 = load i64, i64* %size, align 8, !dbg !791
  %call3 = call i32 @cudaMalloc(i8** %11, i64 %12), !dbg !791
  store i32 %call3, i32* %_status, align 4, !dbg !791
  %13 = load i32, i32* %_status, align 4, !dbg !793
  %cmp = icmp ne i32 %13, 0, !dbg !793
  br i1 %cmp, label %if.then, label %if.end, !dbg !791

if.then:                                          ; preds = %do.body
  %14 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !796
  %15 = load i32, i32* %_status, align 4, !dbg !796
  %call4 = call i8* @cudaGetErrorString(i32 %15), !dbg !796
  %call5 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %14, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.1, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i32 76, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.3, i32 0, i32 0), i8* %call4), !dbg !799
  call void @exit(i32 -1) #9, !dbg !801
  unreachable, !dbg !796

if.end:                                           ; preds = %do.body
  br label %do.end, !dbg !803

do.end:                                           ; preds = %if.end
  br label %do.body6, !dbg !805, !llvm.loop !806

do.body6:                                         ; preds = %do.end
  call void @llvm.dbg.declare(metadata i32* %_status7, metadata !807, metadata !663), !dbg !809
  %16 = bitcast i32** %d_B to i8**, !dbg !810
  %17 = load i64, i64* %size, align 8, !dbg !810
  %call8 = call i32 @cudaMalloc(i8** %16, i64 %17), !dbg !810
  store i32 %call8, i32* %_status7, align 4, !dbg !810
  %18 = load i32, i32* %_status7, align 4, !dbg !812
  %cmp9 = icmp ne i32 %18, 0, !dbg !812
  br i1 %cmp9, label %if.then10, label %if.end13, !dbg !810

if.then10:                                        ; preds = %do.body6
  %19 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !815
  %20 = load i32, i32* %_status7, align 4, !dbg !815
  %call11 = call i8* @cudaGetErrorString(i32 %20), !dbg !815
  %call12 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %19, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.1, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i32 77, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.4, i32 0, i32 0), i8* %call11), !dbg !818
  call void @exit(i32 -1) #9, !dbg !820
  unreachable, !dbg !815

if.end13:                                         ; preds = %do.body6
  br label %do.end14, !dbg !822

do.end14:                                         ; preds = %if.end13
  br label %do.body15, !dbg !824, !llvm.loop !825

do.body15:                                        ; preds = %do.end14
  call void @llvm.dbg.declare(metadata i32* %_status16, metadata !826, metadata !663), !dbg !828
  %21 = bitcast i32** %d_C to i8**, !dbg !829
  %22 = load i64, i64* %size, align 8, !dbg !829
  %call17 = call i32 @cudaMalloc(i8** %21, i64 %22), !dbg !829
  store i32 %call17, i32* %_status16, align 4, !dbg !829
  %23 = load i32, i32* %_status16, align 4, !dbg !831
  %cmp18 = icmp ne i32 %23, 0, !dbg !831
  br i1 %cmp18, label %if.then19, label %if.end22, !dbg !829

if.then19:                                        ; preds = %do.body15
  %24 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !834
  %25 = load i32, i32* %_status16, align 4, !dbg !834
  %call20 = call i8* @cudaGetErrorString(i32 %25), !dbg !834
  %call21 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %24, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.1, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i32 78, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.5, i32 0, i32 0), i8* %call20), !dbg !837
  call void @exit(i32 -1) #9, !dbg !839
  unreachable, !dbg !834

if.end22:                                         ; preds = %do.body15
  br label %do.end23, !dbg !841

do.end23:                                         ; preds = %if.end22
  br label %do.body24, !dbg !843, !llvm.loop !844

do.body24:                                        ; preds = %do.end23
  call void @llvm.dbg.declare(metadata i32* %_status25, metadata !845, metadata !663), !dbg !847
  %26 = load i32*, i32** %d_A, align 8, !dbg !848
  %27 = bitcast i32* %26 to i8*, !dbg !848
  %28 = load i32*, i32** %h_A, align 8, !dbg !848
  %29 = bitcast i32* %28 to i8*, !dbg !848
  %30 = load i64, i64* %size, align 8, !dbg !848
  %31 = load %struct.CUstream_st*, %struct.CUstream_st** %stream.addr, align 8, !dbg !848
  %call26 = call i32 @cudaMemcpyAsync(i8* %27, i8* %29, i64 %30, i32 1, %struct.CUstream_st* %31), !dbg !848
  store i32 %call26, i32* %_status25, align 4, !dbg !848
  %32 = load i32, i32* %_status25, align 4, !dbg !850
  %cmp27 = icmp ne i32 %32, 0, !dbg !850
  br i1 %cmp27, label %if.then28, label %if.end31, !dbg !848

if.then28:                                        ; preds = %do.body24
  %33 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !853
  %34 = load i32, i32* %_status25, align 4, !dbg !853
  %call29 = call i8* @cudaGetErrorString(i32 %34), !dbg !853
  %call30 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %33, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.1, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i32 80, i8* getelementptr inbounds ([64 x i8], [64 x i8]* @.str.6, i32 0, i32 0), i8* %call29), !dbg !856
  call void @exit(i32 -1) #9, !dbg !858
  unreachable, !dbg !853

if.end31:                                         ; preds = %do.body24
  br label %do.end32, !dbg !860

do.end32:                                         ; preds = %if.end31
  br label %do.body33, !dbg !862, !llvm.loop !863

do.body33:                                        ; preds = %do.end32
  call void @llvm.dbg.declare(metadata i32* %_status34, metadata !864, metadata !663), !dbg !866
  %35 = load i32*, i32** %d_B, align 8, !dbg !867
  %36 = bitcast i32* %35 to i8*, !dbg !867
  %37 = load i32*, i32** %h_B, align 8, !dbg !867
  %38 = bitcast i32* %37 to i8*, !dbg !867
  %39 = load i64, i64* %size, align 8, !dbg !867
  %40 = load %struct.CUstream_st*, %struct.CUstream_st** %stream.addr, align 8, !dbg !867
  %call35 = call i32 @cudaMemcpyAsync(i8* %36, i8* %38, i64 %39, i32 1, %struct.CUstream_st* %40), !dbg !867
  store i32 %call35, i32* %_status34, align 4, !dbg !867
  %41 = load i32, i32* %_status34, align 4, !dbg !869
  %cmp36 = icmp ne i32 %41, 0, !dbg !869
  br i1 %cmp36, label %if.then37, label %if.end40, !dbg !867

if.then37:                                        ; preds = %do.body33
  %42 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !872
  %43 = load i32, i32* %_status34, align 4, !dbg !872
  %call38 = call i8* @cudaGetErrorString(i32 %43), !dbg !872
  %call39 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %42, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.1, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i32 81, i8* getelementptr inbounds ([64 x i8], [64 x i8]* @.str.7, i32 0, i32 0), i8* %call38), !dbg !875
  call void @exit(i32 -1) #9, !dbg !877
  unreachable, !dbg !872

if.end40:                                         ; preds = %do.body33
  br label %do.end41, !dbg !879

do.end41:                                         ; preds = %if.end40
  %44 = load i32, i32* %threadsPerBlock, align 4, !dbg !881
  %add = add nsw i32 50000, %44, !dbg !882
  %sub = sub nsw i32 %add, 1, !dbg !883
  %45 = load i32, i32* %threadsPerBlock, align 4, !dbg !884
  %div = sdiv i32 %sub, %45, !dbg !885
  store i32 %div, i32* %blocksPerGrid, align 4, !dbg !886
  %46 = load i32, i32* %blocksPerGrid, align 4, !dbg !887
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %46, i32 1, i32 1), !dbg !887
  %47 = load i32, i32* %threadsPerBlock, align 4, !dbg !888
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp42, i32 %47, i32 1, i32 1), !dbg !889
  %48 = load %struct.CUstream_st*, %struct.CUstream_st** %stream.addr, align 8, !dbg !891
  %49 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !892
  %50 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !892
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %49, i8* %50, i64 12, i32 4, i1 false), !dbg !892
  %51 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !892
  %52 = load i64, i64* %51, align 4, !dbg !892
  %53 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !892
  %54 = load i32, i32* %53, align 4, !dbg !892
  %55 = bitcast { i64, i32 }* %agg.tmp42.coerce to i8*, !dbg !892
  %56 = bitcast %struct.dim3* %agg.tmp42 to i8*, !dbg !892
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %55, i8* %56, i64 12, i32 4, i1 false), !dbg !892
  %57 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp42.coerce, i32 0, i32 0, !dbg !892
  %58 = load i64, i64* %57, align 4, !dbg !892
  %59 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp42.coerce, i32 0, i32 1, !dbg !892
  %60 = load i32, i32* %59, align 4, !dbg !892
  %call43 = call i32 @cudaConfigureCall(i64 %52, i32 %54, i64 %58, i32 %60, i64 0, %struct.CUstream_st* %48), !dbg !893
  %tobool = icmp ne i32 %call43, 0, !dbg !892
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !895

kcall.configok:                                   ; preds = %do.end41
  %61 = load i32*, i32** %d_A, align 8, !dbg !896
  %62 = load i32*, i32** %d_B, align 8, !dbg !898
  %63 = load i32*, i32** %d_C, align 8, !dbg !899
  call void @_Z6VecAddPKiS0_Pii(i32* %61, i32* %62, i32* %63, i32 50000), !dbg !900
  br label %kcall.end, !dbg !900

kcall.end:                                        ; preds = %kcall.configok, %do.end41
  %64 = load i32, i32* %blocksPerGrid, align 4, !dbg !901
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp44, i32 %64, i32 1, i32 1), !dbg !901
  %65 = load i32, i32* %threadsPerBlock, align 4, !dbg !902
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp45, i32 %65, i32 1, i32 1), !dbg !903
  %66 = load %struct.CUstream_st*, %struct.CUstream_st** %stream.addr, align 8, !dbg !904
  %67 = bitcast { i64, i32 }* %agg.tmp44.coerce to i8*, !dbg !905
  %68 = bitcast %struct.dim3* %agg.tmp44 to i8*, !dbg !905
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %67, i8* %68, i64 12, i32 4, i1 false), !dbg !905
  %69 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp44.coerce, i32 0, i32 0, !dbg !905
  %70 = load i64, i64* %69, align 4, !dbg !905
  %71 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp44.coerce, i32 0, i32 1, !dbg !905
  %72 = load i32, i32* %71, align 4, !dbg !905
  %73 = bitcast { i64, i32 }* %agg.tmp45.coerce to i8*, !dbg !905
  %74 = bitcast %struct.dim3* %agg.tmp45 to i8*, !dbg !905
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %73, i8* %74, i64 12, i32 4, i1 false), !dbg !905
  %75 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp45.coerce, i32 0, i32 0, !dbg !905
  %76 = load i64, i64* %75, align 4, !dbg !905
  %77 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp45.coerce, i32 0, i32 1, !dbg !905
  %78 = load i32, i32* %77, align 4, !dbg !905
  %call46 = call i32 @cudaConfigureCall(i64 %70, i32 %72, i64 %76, i32 %78, i64 0, %struct.CUstream_st* %66), !dbg !906
  %tobool47 = icmp ne i32 %call46, 0, !dbg !905
  br i1 %tobool47, label %kcall.end49, label %kcall.configok48, !dbg !907

kcall.configok48:                                 ; preds = %kcall.end
  %79 = load i32*, i32** %d_A, align 8, !dbg !908
  %80 = load i32*, i32** %d_B, align 8, !dbg !909
  %81 = load i32*, i32** %d_C, align 8, !dbg !910
  call void @_Z6VecSubPKiS0_Pii(i32* %79, i32* %80, i32* %81, i32 50000), !dbg !911
  br label %kcall.end49, !dbg !911

kcall.end49:                                      ; preds = %kcall.configok48, %kcall.end
  br label %do.body50, !dbg !912, !llvm.loop !913

do.body50:                                        ; preds = %kcall.end49
  call void @llvm.dbg.declare(metadata i32* %_status51, metadata !914, metadata !663), !dbg !916
  %82 = load i32*, i32** %h_C, align 8, !dbg !917
  %83 = bitcast i32* %82 to i8*, !dbg !917
  %84 = load i32*, i32** %d_C, align 8, !dbg !917
  %85 = bitcast i32* %84 to i8*, !dbg !917
  %86 = load i64, i64* %size, align 8, !dbg !917
  %87 = load %struct.CUstream_st*, %struct.CUstream_st** %stream.addr, align 8, !dbg !917
  %call52 = call i32 @cudaMemcpyAsync(i8* %83, i8* %85, i64 %86, i32 2, %struct.CUstream_st* %87), !dbg !917
  store i32 %call52, i32* %_status51, align 4, !dbg !917
  %88 = load i32, i32* %_status51, align 4, !dbg !919
  %cmp53 = icmp ne i32 %88, 0, !dbg !919
  br i1 %cmp53, label %if.then54, label %if.end57, !dbg !917

if.then54:                                        ; preds = %do.body50
  %89 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !922
  %90 = load i32, i32* %_status51, align 4, !dbg !922
  %call55 = call i8* @cudaGetErrorString(i32 %90), !dbg !922
  %call56 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %89, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.1, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i32 88, i8* getelementptr inbounds ([64 x i8], [64 x i8]* @.str.8, i32 0, i32 0), i8* %call55), !dbg !925
  call void @exit(i32 -1) #9, !dbg !927
  unreachable, !dbg !922

if.end57:                                         ; preds = %do.body50
  br label %do.end58, !dbg !929

do.end58:                                         ; preds = %if.end57
  %91 = load %struct.CUstream_st*, %struct.CUstream_st** %stream.addr, align 8, !dbg !931
  %cmp59 = icmp eq %struct.CUstream_st* %91, null, !dbg !933
  br i1 %cmp59, label %if.then60, label %if.else, !dbg !934

if.then60:                                        ; preds = %do.end58
  br label %do.body61, !dbg !935, !llvm.loop !936

do.body61:                                        ; preds = %if.then60
  call void @llvm.dbg.declare(metadata i32* %_status62, metadata !937, metadata !663), !dbg !939
  %call63 = call i32 @cudaDeviceSynchronize(), !dbg !940
  store i32 %call63, i32* %_status62, align 4, !dbg !940
  %92 = load i32, i32* %_status62, align 4, !dbg !942
  %cmp64 = icmp ne i32 %92, 0, !dbg !942
  br i1 %cmp64, label %if.then65, label %if.end68, !dbg !940

if.then65:                                        ; preds = %do.body61
  %93 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !945
  %94 = load i32, i32* %_status62, align 4, !dbg !945
  %call66 = call i8* @cudaGetErrorString(i32 %94), !dbg !945
  %call67 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %93, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.1, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i32 91, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.9, i32 0, i32 0), i8* %call66), !dbg !948
  call void @exit(i32 -1) #9, !dbg !950
  unreachable, !dbg !945

if.end68:                                         ; preds = %do.body61
  br label %do.end69, !dbg !952

do.end69:                                         ; preds = %if.end68
  br label %if.end79, !dbg !954

if.else:                                          ; preds = %do.end58
  br label %do.body70, !dbg !956, !llvm.loop !957

do.body70:                                        ; preds = %if.else
  call void @llvm.dbg.declare(metadata i32* %_status71, metadata !958, metadata !663), !dbg !960
  %95 = load %struct.CUstream_st*, %struct.CUstream_st** %stream.addr, align 8, !dbg !961
  %call72 = call i32 @cudaStreamSynchronize(%struct.CUstream_st* %95), !dbg !961
  store i32 %call72, i32* %_status71, align 4, !dbg !961
  %96 = load i32, i32* %_status71, align 4, !dbg !963
  %cmp73 = icmp ne i32 %96, 0, !dbg !963
  br i1 %cmp73, label %if.then74, label %if.end77, !dbg !961

if.then74:                                        ; preds = %do.body70
  %97 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !dbg !966
  %98 = load i32, i32* %_status71, align 4, !dbg !966
  %call75 = call i8* @cudaGetErrorString(i32 %98), !dbg !966
  %call76 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %97, i8* getelementptr inbounds ([49 x i8], [49 x i8]* @.str.1, i32 0, i32 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.2, i32 0, i32 0), i32 93, i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.10, i32 0, i32 0), i8* %call75), !dbg !969
  call void @exit(i32 -1) #9, !dbg !971
  unreachable, !dbg !966

if.end77:                                         ; preds = %do.body70
  br label %do.end78, !dbg !973

do.end78:                                         ; preds = %if.end77
  br label %if.end79

if.end79:                                         ; preds = %do.end78, %do.end69
  %99 = load i32*, i32** %h_A, align 8, !dbg !975
  %100 = bitcast i32* %99 to i8*, !dbg !975
  call void @free(i8* %100) #8, !dbg !976
  %101 = load i32*, i32** %h_B, align 8, !dbg !977
  %102 = bitcast i32* %101 to i8*, !dbg !977
  call void @free(i8* %102) #8, !dbg !978
  %103 = load i32*, i32** %h_C, align 8, !dbg !979
  %104 = bitcast i32* %103 to i8*, !dbg !979
  call void @free(i8* %104) #8, !dbg !980
  %105 = load i32*, i32** %d_A, align 8, !dbg !981
  %106 = bitcast i32* %105 to i8*, !dbg !981
  %call80 = call i32 @cudaFree(i8* %106), !dbg !982
  %107 = load i32*, i32** %d_B, align 8, !dbg !983
  %108 = bitcast i32* %107 to i8*, !dbg !983
  %call81 = call i32 @cudaFree(i8* %108), !dbg !984
  %109 = load i32*, i32** %d_C, align 8, !dbg !985
  %110 = bitcast i32* %109 to i8*, !dbg !985
  %call82 = call i32 @cudaFree(i8* %110), !dbg !986
  ret void, !dbg !987
}

declare void @_Z9finiTracev() #4

declare i32 @printf(i8*, ...) #4

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) #3

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL7initVecPii(i32* %vec, i32 %n) #5 !dbg !988 {
entry:
  %vec.addr = alloca i32*, align 8
  %n.addr = alloca i32, align 4
  %i = alloca i32, align 4
  store i32* %vec, i32** %vec.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %vec.addr, metadata !991, metadata !663), !dbg !992
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !993, metadata !663), !dbg !994
  call void @llvm.dbg.declare(metadata i32* %i, metadata !995, metadata !663), !dbg !997
  store i32 0, i32* %i, align 4, !dbg !997
  br label %for.cond, !dbg !998

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4, !dbg !999
  %1 = load i32, i32* %n.addr, align 4, !dbg !1002
  %cmp = icmp slt i32 %0, %1, !dbg !1003
  br i1 %cmp, label %for.body, label %for.end, !dbg !1004

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4, !dbg !1006
  %3 = load i32*, i32** %vec.addr, align 8, !dbg !1007
  %4 = load i32, i32* %i, align 4, !dbg !1008
  %idxprom = sext i32 %4 to i64, !dbg !1007
  %arrayidx = getelementptr inbounds i32, i32* %3, i64 %idxprom, !dbg !1007
  store i32 %2, i32* %arrayidx, align 4, !dbg !1009
  br label %for.inc, !dbg !1007

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4, !dbg !1010
  %inc = add nsw i32 %5, 1, !dbg !1010
  store i32 %inc, i32* %i, align 4, !dbg !1010
  br label %for.cond, !dbg !1012, !llvm.loop !1013

for.end:                                          ; preds = %for.cond
  ret void, !dbg !1016
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #6

declare i32 @cudaMalloc(i8**, i64) #4

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...) #4

declare i8* @cudaGetErrorString(i32) #4

; Function Attrs: noreturn nounwind
declare void @exit(i32) #7

declare i32 @cudaMemcpyAsync(i8*, i8*, i64, i32, %struct.CUstream_st*) #4

declare i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #4

; Function Attrs: noinline nounwind uwtable
define linkonce_odr void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #5 comdat align 2 !dbg !1017 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %this.addr, metadata !1040, metadata !663), !dbg !1042
  store i32 %vx, i32* %vx.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vx.addr, metadata !1043, metadata !663), !dbg !1044
  store i32 %vy, i32* %vy.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vy.addr, metadata !1045, metadata !663), !dbg !1046
  store i32 %vz, i32* %vz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vz.addr, metadata !1047, metadata !663), !dbg !1048
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0, !dbg !1049
  %0 = load i32, i32* %vx.addr, align 4, !dbg !1050
  store i32 %0, i32* %x, align 4, !dbg !1049
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1, !dbg !1051
  %1 = load i32, i32* %vy.addr, align 4, !dbg !1052
  store i32 %1, i32* %y, align 4, !dbg !1051
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2, !dbg !1053
  %2 = load i32, i32* %vz.addr, align 4, !dbg !1054
  store i32 %2, i32* %z, align 4, !dbg !1053
  ret void, !dbg !1055
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #6

declare i32 @cudaDeviceSynchronize() #4

declare i32 @cudaStreamSynchronize(%struct.CUstream_st*) #4

; Function Attrs: nounwind
declare void @free(i8*) #3

declare i32 @cudaFree(i8*) #4

attributes #0 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly nounwind }
attributes #7 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }
attributes #9 = { noreturn nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!653, !654}
!llvm.ident = !{!655}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.1 (tags/RELEASE_401/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !93, imports: !99)
!1 = !DIFile(filename: "vecAddSub.cu", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!2 = !{!3, !86}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaError", file: !4, line: 151, size: 32, elements: !5, identifier: "_ZTS9cudaError")
!4 = !DIFile(filename: "/opt/common/cuda/cuda-7.5.18/include/driver_types.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!5 = !{!6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85}
!6 = !DIEnumerator(name: "cudaSuccess", value: 0)
!7 = !DIEnumerator(name: "cudaErrorMissingConfiguration", value: 1)
!8 = !DIEnumerator(name: "cudaErrorMemoryAllocation", value: 2)
!9 = !DIEnumerator(name: "cudaErrorInitializationError", value: 3)
!10 = !DIEnumerator(name: "cudaErrorLaunchFailure", value: 4)
!11 = !DIEnumerator(name: "cudaErrorPriorLaunchFailure", value: 5)
!12 = !DIEnumerator(name: "cudaErrorLaunchTimeout", value: 6)
!13 = !DIEnumerator(name: "cudaErrorLaunchOutOfResources", value: 7)
!14 = !DIEnumerator(name: "cudaErrorInvalidDeviceFunction", value: 8)
!15 = !DIEnumerator(name: "cudaErrorInvalidConfiguration", value: 9)
!16 = !DIEnumerator(name: "cudaErrorInvalidDevice", value: 10)
!17 = !DIEnumerator(name: "cudaErrorInvalidValue", value: 11)
!18 = !DIEnumerator(name: "cudaErrorInvalidPitchValue", value: 12)
!19 = !DIEnumerator(name: "cudaErrorInvalidSymbol", value: 13)
!20 = !DIEnumerator(name: "cudaErrorMapBufferObjectFailed", value: 14)
!21 = !DIEnumerator(name: "cudaErrorUnmapBufferObjectFailed", value: 15)
!22 = !DIEnumerator(name: "cudaErrorInvalidHostPointer", value: 16)
!23 = !DIEnumerator(name: "cudaErrorInvalidDevicePointer", value: 17)
!24 = !DIEnumerator(name: "cudaErrorInvalidTexture", value: 18)
!25 = !DIEnumerator(name: "cudaErrorInvalidTextureBinding", value: 19)
!26 = !DIEnumerator(name: "cudaErrorInvalidChannelDescriptor", value: 20)
!27 = !DIEnumerator(name: "cudaErrorInvalidMemcpyDirection", value: 21)
!28 = !DIEnumerator(name: "cudaErrorAddressOfConstant", value: 22)
!29 = !DIEnumerator(name: "cudaErrorTextureFetchFailed", value: 23)
!30 = !DIEnumerator(name: "cudaErrorTextureNotBound", value: 24)
!31 = !DIEnumerator(name: "cudaErrorSynchronizationError", value: 25)
!32 = !DIEnumerator(name: "cudaErrorInvalidFilterSetting", value: 26)
!33 = !DIEnumerator(name: "cudaErrorInvalidNormSetting", value: 27)
!34 = !DIEnumerator(name: "cudaErrorMixedDeviceExecution", value: 28)
!35 = !DIEnumerator(name: "cudaErrorCudartUnloading", value: 29)
!36 = !DIEnumerator(name: "cudaErrorUnknown", value: 30)
!37 = !DIEnumerator(name: "cudaErrorNotYetImplemented", value: 31)
!38 = !DIEnumerator(name: "cudaErrorMemoryValueTooLarge", value: 32)
!39 = !DIEnumerator(name: "cudaErrorInvalidResourceHandle", value: 33)
!40 = !DIEnumerator(name: "cudaErrorNotReady", value: 34)
!41 = !DIEnumerator(name: "cudaErrorInsufficientDriver", value: 35)
!42 = !DIEnumerator(name: "cudaErrorSetOnActiveProcess", value: 36)
!43 = !DIEnumerator(name: "cudaErrorInvalidSurface", value: 37)
!44 = !DIEnumerator(name: "cudaErrorNoDevice", value: 38)
!45 = !DIEnumerator(name: "cudaErrorECCUncorrectable", value: 39)
!46 = !DIEnumerator(name: "cudaErrorSharedObjectSymbolNotFound", value: 40)
!47 = !DIEnumerator(name: "cudaErrorSharedObjectInitFailed", value: 41)
!48 = !DIEnumerator(name: "cudaErrorUnsupportedLimit", value: 42)
!49 = !DIEnumerator(name: "cudaErrorDuplicateVariableName", value: 43)
!50 = !DIEnumerator(name: "cudaErrorDuplicateTextureName", value: 44)
!51 = !DIEnumerator(name: "cudaErrorDuplicateSurfaceName", value: 45)
!52 = !DIEnumerator(name: "cudaErrorDevicesUnavailable", value: 46)
!53 = !DIEnumerator(name: "cudaErrorInvalidKernelImage", value: 47)
!54 = !DIEnumerator(name: "cudaErrorNoKernelImageForDevice", value: 48)
!55 = !DIEnumerator(name: "cudaErrorIncompatibleDriverContext", value: 49)
!56 = !DIEnumerator(name: "cudaErrorPeerAccessAlreadyEnabled", value: 50)
!57 = !DIEnumerator(name: "cudaErrorPeerAccessNotEnabled", value: 51)
!58 = !DIEnumerator(name: "cudaErrorDeviceAlreadyInUse", value: 54)
!59 = !DIEnumerator(name: "cudaErrorProfilerDisabled", value: 55)
!60 = !DIEnumerator(name: "cudaErrorProfilerNotInitialized", value: 56)
!61 = !DIEnumerator(name: "cudaErrorProfilerAlreadyStarted", value: 57)
!62 = !DIEnumerator(name: "cudaErrorProfilerAlreadyStopped", value: 58)
!63 = !DIEnumerator(name: "cudaErrorAssert", value: 59)
!64 = !DIEnumerator(name: "cudaErrorTooManyPeers", value: 60)
!65 = !DIEnumerator(name: "cudaErrorHostMemoryAlreadyRegistered", value: 61)
!66 = !DIEnumerator(name: "cudaErrorHostMemoryNotRegistered", value: 62)
!67 = !DIEnumerator(name: "cudaErrorOperatingSystem", value: 63)
!68 = !DIEnumerator(name: "cudaErrorPeerAccessUnsupported", value: 64)
!69 = !DIEnumerator(name: "cudaErrorLaunchMaxDepthExceeded", value: 65)
!70 = !DIEnumerator(name: "cudaErrorLaunchFileScopedTex", value: 66)
!71 = !DIEnumerator(name: "cudaErrorLaunchFileScopedSurf", value: 67)
!72 = !DIEnumerator(name: "cudaErrorSyncDepthExceeded", value: 68)
!73 = !DIEnumerator(name: "cudaErrorLaunchPendingCountExceeded", value: 69)
!74 = !DIEnumerator(name: "cudaErrorNotPermitted", value: 70)
!75 = !DIEnumerator(name: "cudaErrorNotSupported", value: 71)
!76 = !DIEnumerator(name: "cudaErrorHardwareStackError", value: 72)
!77 = !DIEnumerator(name: "cudaErrorIllegalInstruction", value: 73)
!78 = !DIEnumerator(name: "cudaErrorMisalignedAddress", value: 74)
!79 = !DIEnumerator(name: "cudaErrorInvalidAddressSpace", value: 75)
!80 = !DIEnumerator(name: "cudaErrorInvalidPc", value: 76)
!81 = !DIEnumerator(name: "cudaErrorIllegalAddress", value: 77)
!82 = !DIEnumerator(name: "cudaErrorInvalidPtx", value: 78)
!83 = !DIEnumerator(name: "cudaErrorInvalidGraphicsContext", value: 79)
!84 = !DIEnumerator(name: "cudaErrorStartupFailure", value: 127)
!85 = !DIEnumerator(name: "cudaErrorApiFailureBase", value: 10000)
!86 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaMemcpyKind", file: !4, line: 797, size: 32, elements: !87, identifier: "_ZTS14cudaMemcpyKind")
!87 = !{!88, !89, !90, !91, !92}
!88 = !DIEnumerator(name: "cudaMemcpyHostToHost", value: 0)
!89 = !DIEnumerator(name: "cudaMemcpyHostToDevice", value: 1)
!90 = !DIEnumerator(name: "cudaMemcpyDeviceToHost", value: 2)
!91 = !DIEnumerator(name: "cudaMemcpyDeviceToDevice", value: 3)
!92 = !DIEnumerator(name: "cudaMemcpyDefault", value: 4)
!93 = !{!94, !95, !97}
!94 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!95 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !96, size: 64)
!96 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!97 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !98, size: 64)
!98 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!99 = !{!100, !107, !112, !114, !116, !118, !120, !124, !126, !128, !130, !132, !134, !136, !138, !140, !142, !144, !146, !148, !150, !152, !156, !158, !160, !162, !166, !170, !172, !174, !179, !183, !185, !187, !189, !191, !193, !195, !197, !199, !204, !208, !210, !212, !216, !218, !220, !222, !224, !226, !230, !232, !234, !239, !246, !250, !252, !254, !258, !260, !262, !266, !268, !270, !274, !276, !278, !280, !282, !284, !286, !288, !290, !292, !297, !299, !301, !305, !307, !309, !311, !313, !315, !317, !319, !323, !327, !329, !331, !336, !338, !340, !342, !344, !346, !348, !352, !358, !362, !366, !371, !373, !377, !381, !394, !398, !402, !406, !410, !415, !417, !421, !425, !429, !437, !441, !445, !449, !453, !458, !464, !468, !472, !474, !482, !486, !494, !496, !498, !502, !506, !510, !515, !519, !524, !525, !526, !527, !530, !531, !532, !533, !534, !535, !536, !539, !541, !543, !545, !547, !549, !551, !553, !556, !558, !560, !562, !564, !566, !568, !570, !572, !574, !576, !578, !580, !582, !584, !586, !588, !590, !592, !594, !596, !598, !600, !602, !604, !606, !608, !610, !612, !614, !616, !618, !620, !624, !625, !627, !629, !631, !633, !635, !637, !639, !641, !643, !645, !647, !649, !651}
!100 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !103, line: 201)
!101 = !DINamespace(name: "std", scope: null, file: !102, line: 195)
!102 = !DIFile(filename: "/nfshomes/hzhang86/packages/llvm-4.0.1-install/bin/../lib/clang/4.0.1/include/__clang_cuda_math_forward_declares.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!103 = !DISubprogram(name: "abs", linkageName: "_ZL3absx", scope: !102, file: !102, line: 44, type: !104, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!104 = !DISubroutineType(types: !105)
!105 = !{!106, !106}
!106 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!107 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !108, line: 202)
!108 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !102, file: !102, line: 46, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!109 = !DISubroutineType(types: !110)
!110 = !{!111, !111}
!111 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!112 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !113, line: 203)
!113 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !102, file: !102, line: 48, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!114 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !115, line: 204)
!115 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !102, file: !102, line: 50, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!116 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !117, line: 205)
!117 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !102, file: !102, line: 52, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!118 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !119, line: 206)
!119 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !102, file: !102, line: 56, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!120 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !121, line: 207)
!121 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !102, file: !102, line: 54, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!122 = !DISubroutineType(types: !123)
!123 = !{!111, !111, !111}
!124 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !125, line: 208)
!125 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !102, file: !102, line: 58, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !127, line: 209)
!127 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !102, file: !102, line: 60, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!128 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !129, line: 210)
!129 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !102, file: !102, line: 62, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !131, line: 211)
!131 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !102, file: !102, line: 64, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!132 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !133, line: 212)
!133 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !102, file: !102, line: 66, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!134 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !135, line: 213)
!135 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !102, file: !102, line: 68, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!136 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !137, line: 214)
!137 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !102, file: !102, line: 72, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!138 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !139, line: 215)
!139 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !102, file: !102, line: 70, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!140 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !141, line: 216)
!141 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !102, file: !102, line: 76, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!142 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !143, line: 217)
!143 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !102, file: !102, line: 74, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!144 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !145, line: 218)
!145 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !102, file: !102, line: 78, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!146 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !147, line: 219)
!147 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !102, file: !102, line: 80, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!148 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !149, line: 220)
!149 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !102, file: !102, line: 82, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!150 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !151, line: 221)
!151 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !102, file: !102, line: 84, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!152 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !153, line: 222)
!153 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !102, file: !102, line: 86, type: !154, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!154 = !DISubroutineType(types: !155)
!155 = !{!111, !111, !111, !111}
!156 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !157, line: 223)
!157 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !102, file: !102, line: 88, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!158 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !159, line: 224)
!159 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !102, file: !102, line: 90, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!160 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !161, line: 225)
!161 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !102, file: !102, line: 92, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!162 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !163, line: 226)
!163 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !102, file: !102, line: 94, type: !164, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!164 = !DISubroutineType(types: !165)
!165 = !{!96, !111}
!166 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !167, line: 227)
!167 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !102, file: !102, line: 96, type: !168, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!168 = !DISubroutineType(types: !169)
!169 = !{!111, !111, !95}
!170 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !171, line: 228)
!171 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !102, file: !102, line: 98, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!172 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !173, line: 229)
!173 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !102, file: !102, line: 100, type: !164, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!174 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !175, line: 230)
!175 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !102, file: !102, line: 102, type: !176, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!176 = !DISubroutineType(types: !177)
!177 = !{!178, !111}
!178 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !180, line: 231)
!180 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !102, file: !102, line: 106, type: !181, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!181 = !DISubroutineType(types: !182)
!182 = !{!178, !111, !111}
!183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !184, line: 232)
!184 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !102, file: !102, line: 105, type: !181, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !186, line: 233)
!186 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !102, file: !102, line: 108, type: !176, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!187 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !188, line: 234)
!188 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !102, file: !102, line: 112, type: !181, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !190, line: 235)
!190 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !102, file: !102, line: 111, type: !181, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!191 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !192, line: 236)
!192 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !102, file: !102, line: 114, type: !181, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!193 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !194, line: 237)
!194 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !102, file: !102, line: 116, type: !176, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!195 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !196, line: 238)
!196 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !102, file: !102, line: 118, type: !176, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!197 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !198, line: 239)
!198 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !102, file: !102, line: 120, type: !181, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !200, line: 240)
!200 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !102, file: !102, line: 121, type: !201, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!201 = !DISubroutineType(types: !202)
!202 = !{!203, !203}
!203 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!204 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !205, line: 241)
!205 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !102, file: !102, line: 123, type: !206, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!206 = !DISubroutineType(types: !207)
!207 = !{!111, !111, !96}
!208 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !209, line: 242)
!209 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !102, file: !102, line: 125, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!210 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !211, line: 243)
!211 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !102, file: !102, line: 126, type: !104, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!212 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !213, line: 244)
!213 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !102, file: !102, line: 128, type: !214, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!214 = !DISubroutineType(types: !215)
!215 = !{!106, !111}
!216 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !217, line: 245)
!217 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !102, file: !102, line: 138, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!218 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !219, line: 246)
!219 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !102, file: !102, line: 130, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!220 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !221, line: 247)
!221 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !102, file: !102, line: 132, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!222 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !223, line: 248)
!223 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !102, file: !102, line: 134, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!224 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !225, line: 249)
!225 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !102, file: !102, line: 136, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!226 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !227, line: 250)
!227 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !102, file: !102, line: 140, type: !228, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!228 = !DISubroutineType(types: !229)
!229 = !{!203, !111}
!230 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !231, line: 251)
!231 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !102, file: !102, line: 142, type: !228, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!232 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !233, line: 252)
!233 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !102, file: !102, line: 143, type: !214, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!234 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !235, line: 253)
!235 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !102, file: !102, line: 145, type: !236, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!236 = !DISubroutineType(types: !237)
!237 = !{!111, !111, !238}
!238 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !111, size: 64)
!239 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !240, line: 254)
!240 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !102, file: !102, line: 146, type: !241, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!241 = !DISubroutineType(types: !242)
!242 = !{!94, !243}
!243 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !244, size: 64)
!244 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !245)
!245 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!246 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !247, line: 255)
!247 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !102, file: !102, line: 147, type: !248, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!248 = !DISubroutineType(types: !249)
!249 = !{!111, !243}
!250 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !251, line: 256)
!251 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !102, file: !102, line: 149, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!252 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !253, line: 257)
!253 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !102, file: !102, line: 151, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!254 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !255, line: 258)
!255 = !DISubprogram(name: "nexttoward", linkageName: "_ZL10nexttowardfd", scope: !102, file: !102, line: 153, type: !256, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!256 = !DISubroutineType(types: !257)
!257 = !{!111, !111, !94}
!258 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !259, line: 259)
!259 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !102, file: !102, line: 158, type: !206, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!260 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !261, line: 260)
!261 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !102, file: !102, line: 160, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!262 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !263, line: 261)
!263 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !102, file: !102, line: 162, type: !264, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!264 = !DISubroutineType(types: !265)
!265 = !{!111, !111, !111, !95}
!266 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !267, line: 262)
!267 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !102, file: !102, line: 164, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!268 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !269, line: 263)
!269 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !102, file: !102, line: 166, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!270 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !271, line: 264)
!271 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !102, file: !102, line: 168, type: !272, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!272 = !DISubroutineType(types: !273)
!273 = !{!111, !111, !203}
!274 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !275, line: 265)
!275 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !102, file: !102, line: 170, type: !206, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!276 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !277, line: 266)
!277 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !102, file: !102, line: 172, type: !176, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!278 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !279, line: 267)
!279 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !102, file: !102, line: 174, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!280 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !281, line: 268)
!281 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !102, file: !102, line: 176, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!282 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !283, line: 269)
!283 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !102, file: !102, line: 178, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!284 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !285, line: 270)
!285 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !102, file: !102, line: 180, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!286 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !287, line: 271)
!287 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !102, file: !102, line: 182, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!288 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !289, line: 272)
!289 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !102, file: !102, line: 184, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!290 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !291, line: 273)
!291 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !102, file: !102, line: 186, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!292 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !293, line: 102)
!293 = !DISubprogram(name: "acos", scope: !294, file: !294, line: 54, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!294 = !DIFile(filename: "/usr/include/bits/mathcalls.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!295 = !DISubroutineType(types: !296)
!296 = !{!94, !94}
!297 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !298, line: 121)
!298 = !DISubprogram(name: "asin", scope: !294, file: !294, line: 56, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!299 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !300, line: 140)
!300 = !DISubprogram(name: "atan", scope: !294, file: !294, line: 58, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!301 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !302, line: 159)
!302 = !DISubprogram(name: "atan2", scope: !294, file: !294, line: 60, type: !303, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!303 = !DISubroutineType(types: !304)
!304 = !{!94, !94, !94}
!305 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !306, line: 180)
!306 = !DISubprogram(name: "ceil", scope: !294, file: !294, line: 179, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!307 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !308, line: 199)
!308 = !DISubprogram(name: "cos", scope: !294, file: !294, line: 63, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!309 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !310, line: 218)
!310 = !DISubprogram(name: "cosh", scope: !294, file: !294, line: 72, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!311 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !312, line: 237)
!312 = !DISubprogram(name: "exp", scope: !294, file: !294, line: 100, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!313 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !314, line: 256)
!314 = !DISubprogram(name: "fabs", scope: !294, file: !294, line: 182, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!315 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !316, line: 275)
!316 = !DISubprogram(name: "floor", scope: !294, file: !294, line: 185, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!317 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !318, line: 294)
!318 = !DISubprogram(name: "fmod", scope: !294, file: !294, line: 188, type: !303, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!319 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !320, line: 315)
!320 = !DISubprogram(name: "frexp", scope: !294, file: !294, line: 103, type: !321, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!321 = !DISubroutineType(types: !322)
!322 = !{!94, !94, !95}
!323 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !324, line: 334)
!324 = !DISubprogram(name: "ldexp", scope: !294, file: !294, line: 106, type: !325, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!325 = !DISubroutineType(types: !326)
!326 = !{!94, !94, !96}
!327 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !328, line: 353)
!328 = !DISubprogram(name: "log", scope: !294, file: !294, line: 109, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!329 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !330, line: 372)
!330 = !DISubprogram(name: "log10", scope: !294, file: !294, line: 112, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!331 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !332, line: 391)
!332 = !DISubprogram(name: "modf", scope: !294, file: !294, line: 115, type: !333, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!333 = !DISubroutineType(types: !334)
!334 = !{!94, !94, !335}
!335 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !94, size: 64)
!336 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !337, line: 403)
!337 = !DISubprogram(name: "pow", scope: !294, file: !294, line: 154, type: !303, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!338 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !339, line: 440)
!339 = !DISubprogram(name: "sin", scope: !294, file: !294, line: 65, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!340 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !341, line: 459)
!341 = !DISubprogram(name: "sinh", scope: !294, file: !294, line: 74, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!342 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !343, line: 478)
!343 = !DISubprogram(name: "sqrt", scope: !294, file: !294, line: 157, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!344 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !345, line: 497)
!345 = !DISubprogram(name: "tan", scope: !294, file: !294, line: 67, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!346 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !347, line: 516)
!347 = !DISubprogram(name: "tanh", scope: !294, file: !294, line: 76, type: !295, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!348 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !349, line: 118)
!349 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !350, line: 101, baseType: !351)
!350 = !DIFile(filename: "/usr/include/stdlib.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!351 = !DICompositeType(tag: DW_TAG_structure_type, file: !350, line: 97, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!352 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !353, line: 119)
!353 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !350, line: 109, baseType: !354)
!354 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !350, line: 105, size: 128, elements: !355, identifier: "_ZTS6ldiv_t")
!355 = !{!356, !357}
!356 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !354, file: !350, line: 107, baseType: !203, size: 64)
!357 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !354, file: !350, line: 108, baseType: !203, size: 64, offset: 64)
!358 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !359, line: 121)
!359 = !DISubprogram(name: "abort", scope: !350, file: !350, line: 514, type: !360, isLocal: false, isDefinition: false, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: false)
!360 = !DISubroutineType(types: !361)
!361 = !{null}
!362 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !363, line: 122)
!363 = !DISubprogram(name: "abs", scope: !350, file: !350, line: 770, type: !364, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!364 = !DISubroutineType(types: !365)
!365 = !{!96, !96}
!366 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !367, line: 123)
!367 = !DISubprogram(name: "atexit", scope: !350, file: !350, line: 518, type: !368, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!368 = !DISubroutineType(types: !369)
!369 = !{!96, !370}
!370 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !360, size: 64)
!371 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !372, line: 129)
!372 = !DISubprogram(name: "atof", scope: !350, file: !350, line: 144, type: !241, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!373 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !374, line: 130)
!374 = !DISubprogram(name: "atoi", scope: !350, file: !350, line: 147, type: !375, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!375 = !DISubroutineType(types: !376)
!376 = !{!96, !243}
!377 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !378, line: 131)
!378 = !DISubprogram(name: "atol", scope: !350, file: !350, line: 150, type: !379, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!379 = !DISubroutineType(types: !380)
!380 = !{!203, !243}
!381 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !382, line: 132)
!382 = !DISubprogram(name: "bsearch", scope: !350, file: !350, line: 754, type: !383, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!383 = !DISubroutineType(types: !384)
!384 = !{!98, !385, !385, !387, !387, !390}
!385 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !386, size: 64)
!386 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!387 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !388, line: 62, baseType: !389)
!388 = !DIFile(filename: "/nfshomes/hzhang86/packages/llvm-4.0.1-install/bin/../lib/clang/4.0.1/include/stddef.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!389 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!390 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !350, line: 741, baseType: !391)
!391 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !392, size: 64)
!392 = !DISubroutineType(types: !393)
!393 = !{!96, !385, !385}
!394 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !395, line: 133)
!395 = !DISubprogram(name: "calloc", scope: !350, file: !350, line: 467, type: !396, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!396 = !DISubroutineType(types: !397)
!397 = !{!98, !387, !387}
!398 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !399, line: 134)
!399 = !DISubprogram(name: "div", scope: !350, file: !350, line: 784, type: !400, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!400 = !DISubroutineType(types: !401)
!401 = !{!349, !96, !96}
!402 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !403, line: 135)
!403 = !DISubprogram(name: "exit", scope: !350, file: !350, line: 542, type: !404, isLocal: false, isDefinition: false, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: false)
!404 = !DISubroutineType(types: !405)
!405 = !{null, !96}
!406 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !407, line: 136)
!407 = !DISubprogram(name: "free", scope: !350, file: !350, line: 482, type: !408, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!408 = !DISubroutineType(types: !409)
!409 = !{null, !98}
!410 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !411, line: 137)
!411 = !DISubprogram(name: "getenv", scope: !350, file: !350, line: 563, type: !412, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!412 = !DISubroutineType(types: !413)
!413 = !{!414, !243}
!414 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !245, size: 64)
!415 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !416, line: 138)
!416 = !DISubprogram(name: "labs", scope: !350, file: !350, line: 771, type: !201, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!417 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !418, line: 139)
!418 = !DISubprogram(name: "ldiv", scope: !350, file: !350, line: 786, type: !419, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!419 = !DISubroutineType(types: !420)
!420 = !{!353, !203, !203}
!421 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !422, line: 140)
!422 = !DISubprogram(name: "malloc", scope: !350, file: !350, line: 465, type: !423, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!423 = !DISubroutineType(types: !424)
!424 = !{!98, !387}
!425 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !426, line: 142)
!426 = !DISubprogram(name: "mblen", scope: !350, file: !350, line: 859, type: !427, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!427 = !DISubroutineType(types: !428)
!428 = !{!96, !243, !387}
!429 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !430, line: 143)
!430 = !DISubprogram(name: "mbstowcs", scope: !350, file: !350, line: 870, type: !431, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!431 = !DISubroutineType(types: !432)
!432 = !{!387, !433, !436, !387}
!433 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !434)
!434 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !435, size: 64)
!435 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!436 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !243)
!437 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !438, line: 144)
!438 = !DISubprogram(name: "mbtowc", scope: !350, file: !350, line: 862, type: !439, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!439 = !DISubroutineType(types: !440)
!440 = !{!96, !433, !436, !387}
!441 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !442, line: 146)
!442 = !DISubprogram(name: "qsort", scope: !350, file: !350, line: 760, type: !443, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!443 = !DISubroutineType(types: !444)
!444 = !{null, !98, !387, !387, !390}
!445 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !446, line: 152)
!446 = !DISubprogram(name: "rand", scope: !350, file: !350, line: 374, type: !447, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!447 = !DISubroutineType(types: !448)
!448 = !{!96}
!449 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !450, line: 153)
!450 = !DISubprogram(name: "realloc", scope: !350, file: !350, line: 479, type: !451, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!451 = !DISubroutineType(types: !452)
!452 = !{!98, !98, !387}
!453 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !454, line: 154)
!454 = !DISubprogram(name: "srand", scope: !350, file: !350, line: 376, type: !455, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!455 = !DISubroutineType(types: !456)
!456 = !{null, !457}
!457 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!458 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !459, line: 155)
!459 = !DISubprogram(name: "strtod", scope: !350, file: !350, line: 164, type: !460, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!460 = !DISubroutineType(types: !461)
!461 = !{!94, !436, !462}
!462 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !463)
!463 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !414, size: 64)
!464 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !465, line: 156)
!465 = !DISubprogram(name: "strtol", scope: !350, file: !350, line: 183, type: !466, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!466 = !DISubroutineType(types: !467)
!467 = !{!203, !436, !462, !96}
!468 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !469, line: 157)
!469 = !DISubprogram(name: "strtoul", scope: !350, file: !350, line: 187, type: !470, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!470 = !DISubroutineType(types: !471)
!471 = !{!389, !436, !462, !96}
!472 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !473, line: 158)
!473 = !DISubprogram(name: "system", scope: !350, file: !350, line: 716, type: !375, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!474 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !475, line: 160)
!475 = !DISubprogram(name: "wcstombs", scope: !350, file: !350, line: 873, type: !476, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!476 = !DISubroutineType(types: !477)
!477 = !{!387, !478, !479, !387}
!478 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !414)
!479 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !480)
!480 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !481, size: 64)
!481 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !435)
!482 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !483, line: 161)
!483 = !DISubprogram(name: "wctomb", scope: !350, file: !350, line: 866, type: !484, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!484 = !DISubroutineType(types: !485)
!485 = !{!96, !414, !435}
!486 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !487, entity: !489, line: 201)
!487 = !DINamespace(name: "__gnu_cxx", scope: null, file: !488, line: 68)
!488 = !DIFile(filename: "/usr/lib/gcc/x86_64-redhat-linux/4.8.5/../../../../include/c++/4.8.5/bits/cpp_type_traits.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!489 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !350, line: 121, baseType: !490)
!490 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !350, line: 117, size: 128, elements: !491, identifier: "_ZTS7lldiv_t")
!491 = !{!492, !493}
!492 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !490, file: !350, line: 119, baseType: !106, size: 64)
!493 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !490, file: !350, line: 120, baseType: !106, size: 64, offset: 64)
!494 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !487, entity: !495, line: 207)
!495 = !DISubprogram(name: "_Exit", scope: !350, file: !350, line: 556, type: !404, isLocal: false, isDefinition: false, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: false)
!496 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !487, entity: !497, line: 211)
!497 = !DISubprogram(name: "llabs", scope: !350, file: !350, line: 775, type: !104, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!498 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !487, entity: !499, line: 217)
!499 = !DISubprogram(name: "lldiv", scope: !350, file: !350, line: 792, type: !500, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!500 = !DISubroutineType(types: !501)
!501 = !{!489, !106, !106}
!502 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !487, entity: !503, line: 228)
!503 = !DISubprogram(name: "atoll", scope: !350, file: !350, line: 157, type: !504, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!504 = !DISubroutineType(types: !505)
!505 = !{!106, !243}
!506 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !487, entity: !507, line: 229)
!507 = !DISubprogram(name: "strtoll", scope: !350, file: !350, line: 209, type: !508, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!508 = !DISubroutineType(types: !509)
!509 = !{!106, !436, !462, !96}
!510 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !487, entity: !511, line: 230)
!511 = !DISubprogram(name: "strtoull", scope: !350, file: !350, line: 214, type: !512, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!512 = !DISubroutineType(types: !513)
!513 = !{!514, !436, !462, !96}
!514 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!515 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !487, entity: !516, line: 232)
!516 = !DISubprogram(name: "strtof", scope: !350, file: !350, line: 172, type: !517, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!517 = !DISubroutineType(types: !518)
!518 = !{!111, !436, !462}
!519 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !487, entity: !520, line: 233)
!520 = !DISubprogram(name: "strtold", scope: !350, file: !350, line: 175, type: !521, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!521 = !DISubroutineType(types: !522)
!522 = !{!523, !436, !462}
!523 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!524 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !489, line: 241)
!525 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !495, line: 243)
!526 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !497, line: 245)
!527 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !528, line: 246)
!528 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !487, file: !529, line: 214, type: !500, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!529 = !DIFile(filename: "/usr/lib/gcc/x86_64-redhat-linux/4.8.5/../../../../include/c++/4.8.5/cstdlib", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!530 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !499, line: 247)
!531 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !503, line: 249)
!532 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !516, line: 250)
!533 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !507, line: 251)
!534 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !511, line: 252)
!535 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !520, line: 253)
!536 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !537, line: 418)
!537 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !538, file: !538, line: 1300, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!538 = !DIFile(filename: "/opt/common/cuda/cuda-7.5.18/include/math_functions.hpp", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!539 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !540, line: 419)
!540 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !538, file: !538, line: 1328, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!541 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !542, line: 420)
!542 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !538, file: !538, line: 1295, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!543 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !544, line: 421)
!544 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !538, file: !538, line: 1333, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!545 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !546, line: 422)
!546 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !538, file: !538, line: 1285, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!547 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !548, line: 423)
!548 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !538, file: !538, line: 1290, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!549 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !550, line: 424)
!550 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !538, file: !538, line: 1338, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!551 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !552, line: 425)
!552 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !538, file: !538, line: 1388, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!553 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !554, line: 426)
!554 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !555, file: !555, line: 667, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!555 = !DIFile(filename: "/opt/common/cuda/cuda-7.5.18/include/device_functions.hpp", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!556 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !557, line: 427)
!557 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !538, file: !538, line: 1147, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!558 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !559, line: 428)
!559 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !538, file: !538, line: 1201, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!560 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !561, line: 429)
!561 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !538, file: !538, line: 1270, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!562 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !563, line: 430)
!563 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !538, file: !538, line: 1448, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!564 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !565, line: 431)
!565 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !538, file: !538, line: 1438, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!566 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !567, line: 432)
!567 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !555, file: !555, line: 657, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!568 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !569, line: 433)
!569 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !538, file: !538, line: 1252, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!570 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !571, line: 434)
!571 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !538, file: !538, line: 1343, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!572 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !573, line: 435)
!573 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !555, file: !555, line: 607, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!574 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !575, line: 436)
!575 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !538, file: !538, line: 1574, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!576 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !577, line: 437)
!577 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !555, file: !555, line: 597, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!578 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !579, line: 438)
!579 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !538, file: !538, line: 1526, type: !154, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!580 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !581, line: 439)
!581 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !555, file: !555, line: 622, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!582 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !583, line: 440)
!583 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !555, file: !555, line: 617, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!584 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !585, line: 441)
!585 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !538, file: !538, line: 1511, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!586 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !587, line: 442)
!587 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !538, file: !538, line: 1501, type: !168, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!588 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !589, line: 443)
!589 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !538, file: !538, line: 1348, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!590 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !591, line: 444)
!591 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !538, file: !538, line: 1579, type: !164, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!592 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !593, line: 445)
!593 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !538, file: !538, line: 1478, type: !206, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!594 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !595, line: 446)
!595 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !538, file: !538, line: 1473, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!596 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !597, line: 447)
!597 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !538, file: !538, line: 1107, type: !214, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!598 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !599, line: 448)
!599 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !538, file: !538, line: 1560, type: !214, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!600 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !601, line: 449)
!601 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !538, file: !538, line: 1314, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!602 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !603, line: 450)
!603 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !538, file: !538, line: 1323, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!604 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !605, line: 451)
!605 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !538, file: !538, line: 1243, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!606 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !607, line: 452)
!607 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !538, file: !538, line: 1584, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!608 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !609, line: 453)
!609 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !538, file: !538, line: 1305, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!610 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !611, line: 454)
!611 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !538, file: !538, line: 1098, type: !228, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!612 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !613, line: 455)
!613 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !538, file: !538, line: 1565, type: !228, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!614 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !615, line: 456)
!615 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !538, file: !538, line: 1506, type: !236, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!616 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !617, line: 457)
!617 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !538, file: !538, line: 1112, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!618 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !619, line: 458)
!619 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !538, file: !538, line: 1176, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!620 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !621, line: 459)
!621 = !DISubprogram(name: "nexttowardf", scope: !294, file: !294, line: 285, type: !622, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!622 = !DISubroutineType(types: !623)
!623 = !{!111, !111, !523}
!624 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !621, line: 460)
!625 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !626, line: 461)
!626 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !538, file: !538, line: 1541, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!627 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !628, line: 462)
!628 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !538, file: !538, line: 1516, type: !122, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!629 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !630, line: 463)
!630 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !538, file: !538, line: 1521, type: !264, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!631 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !632, line: 464)
!632 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !538, file: !538, line: 1093, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!633 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !634, line: 465)
!634 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !538, file: !538, line: 1555, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!635 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !636, line: 466)
!636 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !538, file: !538, line: 1488, type: !272, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!637 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !638, line: 467)
!638 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !538, file: !538, line: 1483, type: !206, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!639 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !640, line: 468)
!640 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !538, file: !538, line: 1192, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!641 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !642, line: 469)
!642 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !538, file: !538, line: 1275, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!643 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !644, line: 470)
!644 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !555, file: !555, line: 907, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!645 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !646, line: 471)
!646 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !538, file: !538, line: 1234, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!647 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !648, line: 472)
!648 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !538, file: !538, line: 1280, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!649 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !650, line: 473)
!650 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !538, file: !538, line: 1550, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!651 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !101, entity: !652, line: 474)
!652 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !555, file: !555, line: 662, type: !109, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!653 = !{i32 2, !"Dwarf Version", i32 4}
!654 = !{i32 2, !"Debug Info Version", i32 3}
!655 = !{!"clang version 4.0.1 (tags/RELEASE_401/final)"}
!656 = distinct !DISubprogram(name: "VecAdd", linkageName: "_Z6VecAddPKiS0_Pii", scope: !1, file: !1, line: 31, type: !657, isLocal: false, isDefinition: true, scopeLine: 32, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !661)
!657 = !DISubroutineType(types: !658)
!658 = !{null, !659, !659, !95, !96}
!659 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !660, size: 64)
!660 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !96)
!661 = !{}
!662 = !DILocalVariable(name: "A", arg: 1, scope: !656, file: !1, line: 31, type: !659)
!663 = !DIExpression()
!664 = !DILocation(line: 31, column: 19, scope: !656)
!665 = !DILocalVariable(name: "B", arg: 2, scope: !656, file: !1, line: 31, type: !659)
!666 = !DILocation(line: 31, column: 33, scope: !656)
!667 = !DILocalVariable(name: "C", arg: 3, scope: !656, file: !1, line: 31, type: !95)
!668 = !DILocation(line: 31, column: 41, scope: !656)
!669 = !DILocalVariable(name: "N", arg: 4, scope: !656, file: !1, line: 31, type: !96)
!670 = !DILocation(line: 31, column: 48, scope: !656)
!671 = !DILocation(line: 32, column: 1, scope: !656)
!672 = !DILocation(line: 32, column: 1, scope: !673)
!673 = !DILexicalBlockFile(scope: !656, file: !1, discriminator: 1)
!674 = !DILocation(line: 32, column: 1, scope: !675)
!675 = !DILexicalBlockFile(scope: !656, file: !1, discriminator: 2)
!676 = !DILocation(line: 32, column: 1, scope: !677)
!677 = !DILexicalBlockFile(scope: !656, file: !1, discriminator: 3)
!678 = !DILocation(line: 32, column: 1, scope: !679)
!679 = !DILexicalBlockFile(scope: !656, file: !1, discriminator: 4)
!680 = !DILocation(line: 37, column: 1, scope: !656)
!681 = distinct !DISubprogram(name: "VecSub", linkageName: "_Z6VecSubPKiS0_Pii", scope: !1, file: !1, line: 40, type: !657, isLocal: false, isDefinition: true, scopeLine: 41, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !661)
!682 = !DILocalVariable(name: "A", arg: 1, scope: !681, file: !1, line: 40, type: !659)
!683 = !DILocation(line: 40, column: 19, scope: !681)
!684 = !DILocalVariable(name: "B", arg: 2, scope: !681, file: !1, line: 40, type: !659)
!685 = !DILocation(line: 40, column: 33, scope: !681)
!686 = !DILocalVariable(name: "C", arg: 3, scope: !681, file: !1, line: 40, type: !95)
!687 = !DILocation(line: 40, column: 41, scope: !681)
!688 = !DILocalVariable(name: "N", arg: 4, scope: !681, file: !1, line: 40, type: !96)
!689 = !DILocation(line: 40, column: 48, scope: !681)
!690 = !DILocation(line: 41, column: 1, scope: !681)
!691 = !DILocation(line: 41, column: 1, scope: !692)
!692 = !DILexicalBlockFile(scope: !681, file: !1, discriminator: 1)
!693 = !DILocation(line: 41, column: 1, scope: !694)
!694 = !DILexicalBlockFile(scope: !681, file: !1, discriminator: 2)
!695 = !DILocation(line: 41, column: 1, scope: !696)
!696 = !DILexicalBlockFile(scope: !681, file: !1, discriminator: 3)
!697 = !DILocation(line: 41, column: 1, scope: !698)
!698 = !DILexicalBlockFile(scope: !681, file: !1, discriminator: 4)
!699 = !DILocation(line: 46, column: 1, scope: !681)
!700 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 104, type: !701, isLocal: false, isDefinition: true, scopeLine: 105, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !661)
!701 = !DISubroutineType(types: !702)
!702 = !{!96, !96, !463}
!703 = !DILocalVariable(name: "argc", arg: 1, scope: !700, file: !1, line: 104, type: !96)
!704 = !DILocation(line: 104, column: 14, scope: !700)
!705 = !DILocalVariable(name: "argv", arg: 2, scope: !700, file: !1, line: 104, type: !463)
!706 = !DILocation(line: 104, column: 26, scope: !700)
!707 = !DILocalVariable(name: "start", scope: !700, file: !1, line: 106, type: !708)
!708 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "timeval", file: !709, line: 30, size: 128, elements: !710, identifier: "_ZTS7timeval")
!709 = !DIFile(filename: "/usr/include/bits/time.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!710 = !{!711, !714}
!711 = !DIDerivedType(tag: DW_TAG_member, name: "tv_sec", scope: !708, file: !709, line: 32, baseType: !712, size: 64)
!712 = !DIDerivedType(tag: DW_TAG_typedef, name: "__time_t", file: !713, line: 148, baseType: !203)
!713 = !DIFile(filename: "/usr/include/bits/types.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!714 = !DIDerivedType(tag: DW_TAG_member, name: "tv_usec", scope: !708, file: !709, line: 33, baseType: !715, size: 64, offset: 64)
!715 = !DIDerivedType(tag: DW_TAG_typedef, name: "__suseconds_t", file: !713, line: 150, baseType: !203)
!716 = !DILocation(line: 106, column: 11, scope: !700)
!717 = !DILocation(line: 107, column: 3, scope: !700)
!718 = !DILocation(line: 109, column: 3, scope: !700)
!719 = !DILocation(line: 111, column: 3, scope: !700)
!720 = !DILocation(line: 113, column: 3, scope: !700)
!721 = !DILocalVariable(name: "end", scope: !700, file: !1, line: 115, type: !708)
!722 = !DILocation(line: 115, column: 11, scope: !700)
!723 = !DILocation(line: 116, column: 3, scope: !700)
!724 = !DILocalVariable(name: "elapsed_time", scope: !700, file: !1, line: 117, type: !94)
!725 = !DILocation(line: 117, column: 10, scope: !700)
!726 = !DILocation(line: 117, column: 38, scope: !700)
!727 = !DILocation(line: 117, column: 53, scope: !700)
!728 = !DILocation(line: 117, column: 45, scope: !700)
!729 = !DILocation(line: 117, column: 33, scope: !700)
!730 = !DILocation(line: 118, column: 39, scope: !700)
!731 = !DILocation(line: 118, column: 55, scope: !700)
!732 = !DILocation(line: 118, column: 47, scope: !700)
!733 = !DILocation(line: 118, column: 34, scope: !700)
!734 = !DILocation(line: 118, column: 64, scope: !700)
!735 = !DILocation(line: 117, column: 61, scope: !700)
!736 = !DILocation(line: 119, column: 51, scope: !700)
!737 = !DILocation(line: 119, column: 3, scope: !700)
!738 = !DILocation(line: 120, column: 3, scope: !700)
!739 = distinct !DISubprogram(name: "do_pass", linkageName: "_ZL7do_passP11CUstream_st", scope: !1, file: !1, line: 56, type: !740, isLocal: true, isDefinition: true, scopeLine: 57, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !661)
!740 = !DISubroutineType(types: !741)
!741 = !{null, !742}
!742 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaStream_t", file: !4, line: 1425, baseType: !743)
!743 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !744, size: 64)
!744 = !DICompositeType(tag: DW_TAG_structure_type, name: "CUstream_st", file: !745, line: 224, flags: DIFlagFwdDecl, identifier: "_ZTS11CUstream_st")
!745 = !DIFile(filename: "/opt/common/cuda/cuda-7.5.18/include/cuda.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!746 = !DILocalVariable(name: "stream", arg: 1, scope: !739, file: !1, line: 56, type: !742)
!747 = !DILocation(line: 56, column: 22, scope: !739)
!748 = !DILocalVariable(name: "h_A", scope: !739, file: !1, line: 58, type: !95)
!749 = !DILocation(line: 58, column: 8, scope: !739)
!750 = !DILocalVariable(name: "h_B", scope: !739, file: !1, line: 58, type: !95)
!751 = !DILocation(line: 58, column: 14, scope: !739)
!752 = !DILocalVariable(name: "h_C", scope: !739, file: !1, line: 58, type: !95)
!753 = !DILocation(line: 58, column: 20, scope: !739)
!754 = !DILocalVariable(name: "d_A", scope: !739, file: !1, line: 59, type: !95)
!755 = !DILocation(line: 59, column: 8, scope: !739)
!756 = !DILocalVariable(name: "d_B", scope: !739, file: !1, line: 59, type: !95)
!757 = !DILocation(line: 59, column: 14, scope: !739)
!758 = !DILocalVariable(name: "d_C", scope: !739, file: !1, line: 59, type: !95)
!759 = !DILocation(line: 59, column: 20, scope: !739)
!760 = !DILocalVariable(name: "size", scope: !739, file: !1, line: 60, type: !387)
!761 = !DILocation(line: 60, column: 10, scope: !739)
!762 = !DILocalVariable(name: "threadsPerBlock", scope: !739, file: !1, line: 61, type: !96)
!763 = !DILocation(line: 61, column: 7, scope: !739)
!764 = !DILocalVariable(name: "blocksPerGrid", scope: !739, file: !1, line: 62, type: !96)
!765 = !DILocation(line: 62, column: 7, scope: !739)
!766 = !DILocation(line: 66, column: 22, scope: !739)
!767 = !DILocation(line: 66, column: 15, scope: !739)
!768 = !DILocation(line: 66, column: 9, scope: !739)
!769 = !DILocation(line: 66, column: 7, scope: !739)
!770 = !DILocation(line: 67, column: 22, scope: !739)
!771 = !DILocation(line: 67, column: 15, scope: !739)
!772 = !DILocation(line: 67, column: 9, scope: !739)
!773 = !DILocation(line: 67, column: 7, scope: !739)
!774 = !DILocation(line: 68, column: 22, scope: !739)
!775 = !DILocation(line: 68, column: 15, scope: !739)
!776 = !DILocation(line: 68, column: 9, scope: !739)
!777 = !DILocation(line: 68, column: 7, scope: !739)
!778 = !DILocation(line: 71, column: 11, scope: !739)
!779 = !DILocation(line: 71, column: 3, scope: !739)
!780 = !DILocation(line: 72, column: 11, scope: !739)
!781 = !DILocation(line: 72, column: 3, scope: !739)
!782 = !DILocation(line: 73, column: 10, scope: !739)
!783 = !DILocation(line: 73, column: 3, scope: !739)
!784 = !DILocation(line: 73, column: 18, scope: !739)
!785 = !DILocation(line: 76, column: 3, scope: !739)
!786 = distinct !{!786, !785, !785}
!787 = !DILocalVariable(name: "_status", scope: !788, file: !1, line: 76, type: !789)
!788 = distinct !DILexicalBlock(scope: !739, file: !1, line: 76, column: 3)
!789 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaError_t", file: !4, line: 1420, baseType: !3)
!790 = !DILocation(line: 76, column: 3, scope: !788)
!791 = !DILocation(line: 76, column: 3, scope: !792)
!792 = !DILexicalBlockFile(scope: !788, file: !1, discriminator: 1)
!793 = !DILocation(line: 76, column: 3, scope: !794)
!794 = !DILexicalBlockFile(scope: !795, file: !1, discriminator: 1)
!795 = distinct !DILexicalBlock(scope: !788, file: !1, line: 76, column: 3)
!796 = !DILocation(line: 76, column: 3, scope: !797)
!797 = !DILexicalBlockFile(scope: !798, file: !1, discriminator: 2)
!798 = distinct !DILexicalBlock(scope: !795, file: !1, line: 76, column: 3)
!799 = !DILocation(line: 76, column: 3, scope: !800)
!800 = !DILexicalBlockFile(scope: !798, file: !1, discriminator: 4)
!801 = !DILocation(line: 76, column: 3, scope: !802)
!802 = !DILexicalBlockFile(scope: !798, file: !1, discriminator: 5)
!803 = !DILocation(line: 76, column: 3, scope: !804)
!804 = !DILexicalBlockFile(scope: !788, file: !1, discriminator: 3)
!805 = !DILocation(line: 77, column: 3, scope: !739)
!806 = distinct !{!806, !805, !805}
!807 = !DILocalVariable(name: "_status", scope: !808, file: !1, line: 77, type: !789)
!808 = distinct !DILexicalBlock(scope: !739, file: !1, line: 77, column: 3)
!809 = !DILocation(line: 77, column: 3, scope: !808)
!810 = !DILocation(line: 77, column: 3, scope: !811)
!811 = !DILexicalBlockFile(scope: !808, file: !1, discriminator: 1)
!812 = !DILocation(line: 77, column: 3, scope: !813)
!813 = !DILexicalBlockFile(scope: !814, file: !1, discriminator: 1)
!814 = distinct !DILexicalBlock(scope: !808, file: !1, line: 77, column: 3)
!815 = !DILocation(line: 77, column: 3, scope: !816)
!816 = !DILexicalBlockFile(scope: !817, file: !1, discriminator: 2)
!817 = distinct !DILexicalBlock(scope: !814, file: !1, line: 77, column: 3)
!818 = !DILocation(line: 77, column: 3, scope: !819)
!819 = !DILexicalBlockFile(scope: !817, file: !1, discriminator: 4)
!820 = !DILocation(line: 77, column: 3, scope: !821)
!821 = !DILexicalBlockFile(scope: !817, file: !1, discriminator: 5)
!822 = !DILocation(line: 77, column: 3, scope: !823)
!823 = !DILexicalBlockFile(scope: !808, file: !1, discriminator: 3)
!824 = !DILocation(line: 78, column: 3, scope: !739)
!825 = distinct !{!825, !824, !824}
!826 = !DILocalVariable(name: "_status", scope: !827, file: !1, line: 78, type: !789)
!827 = distinct !DILexicalBlock(scope: !739, file: !1, line: 78, column: 3)
!828 = !DILocation(line: 78, column: 3, scope: !827)
!829 = !DILocation(line: 78, column: 3, scope: !830)
!830 = !DILexicalBlockFile(scope: !827, file: !1, discriminator: 1)
!831 = !DILocation(line: 78, column: 3, scope: !832)
!832 = !DILexicalBlockFile(scope: !833, file: !1, discriminator: 1)
!833 = distinct !DILexicalBlock(scope: !827, file: !1, line: 78, column: 3)
!834 = !DILocation(line: 78, column: 3, scope: !835)
!835 = !DILexicalBlockFile(scope: !836, file: !1, discriminator: 2)
!836 = distinct !DILexicalBlock(scope: !833, file: !1, line: 78, column: 3)
!837 = !DILocation(line: 78, column: 3, scope: !838)
!838 = !DILexicalBlockFile(scope: !836, file: !1, discriminator: 4)
!839 = !DILocation(line: 78, column: 3, scope: !840)
!840 = !DILexicalBlockFile(scope: !836, file: !1, discriminator: 5)
!841 = !DILocation(line: 78, column: 3, scope: !842)
!842 = !DILexicalBlockFile(scope: !827, file: !1, discriminator: 3)
!843 = !DILocation(line: 80, column: 3, scope: !739)
!844 = distinct !{!844, !843, !843}
!845 = !DILocalVariable(name: "_status", scope: !846, file: !1, line: 80, type: !789)
!846 = distinct !DILexicalBlock(scope: !739, file: !1, line: 80, column: 3)
!847 = !DILocation(line: 80, column: 3, scope: !846)
!848 = !DILocation(line: 80, column: 3, scope: !849)
!849 = !DILexicalBlockFile(scope: !846, file: !1, discriminator: 1)
!850 = !DILocation(line: 80, column: 3, scope: !851)
!851 = !DILexicalBlockFile(scope: !852, file: !1, discriminator: 1)
!852 = distinct !DILexicalBlock(scope: !846, file: !1, line: 80, column: 3)
!853 = !DILocation(line: 80, column: 3, scope: !854)
!854 = !DILexicalBlockFile(scope: !855, file: !1, discriminator: 2)
!855 = distinct !DILexicalBlock(scope: !852, file: !1, line: 80, column: 3)
!856 = !DILocation(line: 80, column: 3, scope: !857)
!857 = !DILexicalBlockFile(scope: !855, file: !1, discriminator: 4)
!858 = !DILocation(line: 80, column: 3, scope: !859)
!859 = !DILexicalBlockFile(scope: !855, file: !1, discriminator: 5)
!860 = !DILocation(line: 80, column: 3, scope: !861)
!861 = !DILexicalBlockFile(scope: !846, file: !1, discriminator: 3)
!862 = !DILocation(line: 81, column: 3, scope: !739)
!863 = distinct !{!863, !862, !862}
!864 = !DILocalVariable(name: "_status", scope: !865, file: !1, line: 81, type: !789)
!865 = distinct !DILexicalBlock(scope: !739, file: !1, line: 81, column: 3)
!866 = !DILocation(line: 81, column: 3, scope: !865)
!867 = !DILocation(line: 81, column: 3, scope: !868)
!868 = !DILexicalBlockFile(scope: !865, file: !1, discriminator: 1)
!869 = !DILocation(line: 81, column: 3, scope: !870)
!870 = !DILexicalBlockFile(scope: !871, file: !1, discriminator: 1)
!871 = distinct !DILexicalBlock(scope: !865, file: !1, line: 81, column: 3)
!872 = !DILocation(line: 81, column: 3, scope: !873)
!873 = !DILexicalBlockFile(scope: !874, file: !1, discriminator: 2)
!874 = distinct !DILexicalBlock(scope: !871, file: !1, line: 81, column: 3)
!875 = !DILocation(line: 81, column: 3, scope: !876)
!876 = !DILexicalBlockFile(scope: !874, file: !1, discriminator: 4)
!877 = !DILocation(line: 81, column: 3, scope: !878)
!878 = !DILexicalBlockFile(scope: !874, file: !1, discriminator: 5)
!879 = !DILocation(line: 81, column: 3, scope: !880)
!880 = !DILexicalBlockFile(scope: !865, file: !1, discriminator: 3)
!881 = !DILocation(line: 83, column: 32, scope: !739)
!882 = !DILocation(line: 83, column: 30, scope: !739)
!883 = !DILocation(line: 83, column: 48, scope: !739)
!884 = !DILocation(line: 83, column: 55, scope: !739)
!885 = !DILocation(line: 83, column: 53, scope: !739)
!886 = !DILocation(line: 83, column: 17, scope: !739)
!887 = !DILocation(line: 84, column: 12, scope: !739)
!888 = !DILocation(line: 84, column: 27, scope: !739)
!889 = !DILocation(line: 84, column: 27, scope: !890)
!890 = !DILexicalBlockFile(scope: !739, file: !1, discriminator: 2)
!891 = !DILocation(line: 84, column: 47, scope: !739)
!892 = !DILocation(line: 84, column: 9, scope: !739)
!893 = !DILocation(line: 84, column: 9, scope: !894)
!894 = !DILexicalBlockFile(scope: !739, file: !1, discriminator: 3)
!895 = !DILocation(line: 84, column: 3, scope: !739)
!896 = !DILocation(line: 84, column: 57, scope: !897)
!897 = !DILexicalBlockFile(scope: !739, file: !1, discriminator: 1)
!898 = !DILocation(line: 84, column: 62, scope: !897)
!899 = !DILocation(line: 84, column: 67, scope: !897)
!900 = !DILocation(line: 84, column: 3, scope: !897)
!901 = !DILocation(line: 86, column: 12, scope: !739)
!902 = !DILocation(line: 86, column: 27, scope: !739)
!903 = !DILocation(line: 86, column: 27, scope: !890)
!904 = !DILocation(line: 86, column: 47, scope: !739)
!905 = !DILocation(line: 86, column: 9, scope: !739)
!906 = !DILocation(line: 86, column: 9, scope: !894)
!907 = !DILocation(line: 86, column: 3, scope: !739)
!908 = !DILocation(line: 86, column: 57, scope: !897)
!909 = !DILocation(line: 86, column: 62, scope: !897)
!910 = !DILocation(line: 86, column: 67, scope: !897)
!911 = !DILocation(line: 86, column: 3, scope: !897)
!912 = !DILocation(line: 88, column: 3, scope: !739)
!913 = distinct !{!913, !912, !912}
!914 = !DILocalVariable(name: "_status", scope: !915, file: !1, line: 88, type: !789)
!915 = distinct !DILexicalBlock(scope: !739, file: !1, line: 88, column: 3)
!916 = !DILocation(line: 88, column: 3, scope: !915)
!917 = !DILocation(line: 88, column: 3, scope: !918)
!918 = !DILexicalBlockFile(scope: !915, file: !1, discriminator: 1)
!919 = !DILocation(line: 88, column: 3, scope: !920)
!920 = !DILexicalBlockFile(scope: !921, file: !1, discriminator: 1)
!921 = distinct !DILexicalBlock(scope: !915, file: !1, line: 88, column: 3)
!922 = !DILocation(line: 88, column: 3, scope: !923)
!923 = !DILexicalBlockFile(scope: !924, file: !1, discriminator: 2)
!924 = distinct !DILexicalBlock(scope: !921, file: !1, line: 88, column: 3)
!925 = !DILocation(line: 88, column: 3, scope: !926)
!926 = !DILexicalBlockFile(scope: !924, file: !1, discriminator: 4)
!927 = !DILocation(line: 88, column: 3, scope: !928)
!928 = !DILexicalBlockFile(scope: !924, file: !1, discriminator: 5)
!929 = !DILocation(line: 88, column: 3, scope: !930)
!930 = !DILexicalBlockFile(scope: !915, file: !1, discriminator: 3)
!931 = !DILocation(line: 90, column: 7, scope: !932)
!932 = distinct !DILexicalBlock(scope: !739, file: !1, line: 90, column: 7)
!933 = !DILocation(line: 90, column: 14, scope: !932)
!934 = !DILocation(line: 90, column: 7, scope: !739)
!935 = !DILocation(line: 91, column: 5, scope: !932)
!936 = distinct !{!936, !935, !935}
!937 = !DILocalVariable(name: "_status", scope: !938, file: !1, line: 91, type: !789)
!938 = distinct !DILexicalBlock(scope: !932, file: !1, line: 91, column: 5)
!939 = !DILocation(line: 91, column: 5, scope: !938)
!940 = !DILocation(line: 91, column: 5, scope: !941)
!941 = !DILexicalBlockFile(scope: !938, file: !1, discriminator: 1)
!942 = !DILocation(line: 91, column: 5, scope: !943)
!943 = !DILexicalBlockFile(scope: !944, file: !1, discriminator: 1)
!944 = distinct !DILexicalBlock(scope: !938, file: !1, line: 91, column: 5)
!945 = !DILocation(line: 91, column: 5, scope: !946)
!946 = !DILexicalBlockFile(scope: !947, file: !1, discriminator: 2)
!947 = distinct !DILexicalBlock(scope: !944, file: !1, line: 91, column: 5)
!948 = !DILocation(line: 91, column: 5, scope: !949)
!949 = !DILexicalBlockFile(scope: !947, file: !1, discriminator: 5)
!950 = !DILocation(line: 91, column: 5, scope: !951)
!951 = !DILexicalBlockFile(scope: !947, file: !1, discriminator: 6)
!952 = !DILocation(line: 91, column: 5, scope: !953)
!953 = !DILexicalBlockFile(scope: !938, file: !1, discriminator: 3)
!954 = !DILocation(line: 91, column: 5, scope: !955)
!955 = !DILexicalBlockFile(scope: !938, file: !1, discriminator: 4)
!956 = !DILocation(line: 93, column: 5, scope: !932)
!957 = distinct !{!957, !956, !956}
!958 = !DILocalVariable(name: "_status", scope: !959, file: !1, line: 93, type: !789)
!959 = distinct !DILexicalBlock(scope: !932, file: !1, line: 93, column: 5)
!960 = !DILocation(line: 93, column: 5, scope: !959)
!961 = !DILocation(line: 93, column: 5, scope: !962)
!962 = !DILexicalBlockFile(scope: !959, file: !1, discriminator: 1)
!963 = !DILocation(line: 93, column: 5, scope: !964)
!964 = !DILexicalBlockFile(scope: !965, file: !1, discriminator: 1)
!965 = distinct !DILexicalBlock(scope: !959, file: !1, line: 93, column: 5)
!966 = !DILocation(line: 93, column: 5, scope: !967)
!967 = !DILexicalBlockFile(scope: !968, file: !1, discriminator: 2)
!968 = distinct !DILexicalBlock(scope: !965, file: !1, line: 93, column: 5)
!969 = !DILocation(line: 93, column: 5, scope: !970)
!970 = !DILexicalBlockFile(scope: !968, file: !1, discriminator: 4)
!971 = !DILocation(line: 93, column: 5, scope: !972)
!972 = !DILexicalBlockFile(scope: !968, file: !1, discriminator: 5)
!973 = !DILocation(line: 93, column: 5, scope: !974)
!974 = !DILexicalBlockFile(scope: !959, file: !1, discriminator: 3)
!975 = !DILocation(line: 95, column: 8, scope: !739)
!976 = !DILocation(line: 95, column: 3, scope: !739)
!977 = !DILocation(line: 96, column: 8, scope: !739)
!978 = !DILocation(line: 96, column: 3, scope: !739)
!979 = !DILocation(line: 97, column: 8, scope: !739)
!980 = !DILocation(line: 97, column: 3, scope: !739)
!981 = !DILocation(line: 98, column: 12, scope: !739)
!982 = !DILocation(line: 98, column: 3, scope: !739)
!983 = !DILocation(line: 99, column: 12, scope: !739)
!984 = !DILocation(line: 99, column: 3, scope: !739)
!985 = !DILocation(line: 100, column: 12, scope: !739)
!986 = !DILocation(line: 100, column: 3, scope: !739)
!987 = !DILocation(line: 101, column: 1, scope: !739)
!988 = distinct !DISubprogram(name: "initVec", linkageName: "_ZL7initVecPii", scope: !1, file: !1, line: 49, type: !989, isLocal: true, isDefinition: true, scopeLine: 50, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !661)
!989 = !DISubroutineType(types: !990)
!990 = !{null, !95, !96}
!991 = !DILocalVariable(name: "vec", arg: 1, scope: !988, file: !1, line: 49, type: !95)
!992 = !DILocation(line: 49, column: 14, scope: !988)
!993 = !DILocalVariable(name: "n", arg: 2, scope: !988, file: !1, line: 49, type: !96)
!994 = !DILocation(line: 49, column: 23, scope: !988)
!995 = !DILocalVariable(name: "i", scope: !996, file: !1, line: 51, type: !96)
!996 = distinct !DILexicalBlock(scope: !988, file: !1, line: 51, column: 3)
!997 = !DILocation(line: 51, column: 12, scope: !996)
!998 = !DILocation(line: 51, column: 8, scope: !996)
!999 = !DILocation(line: 51, column: 17, scope: !1000)
!1000 = !DILexicalBlockFile(scope: !1001, file: !1, discriminator: 1)
!1001 = distinct !DILexicalBlock(scope: !996, file: !1, line: 51, column: 3)
!1002 = !DILocation(line: 51, column: 20, scope: !1000)
!1003 = !DILocation(line: 51, column: 18, scope: !1000)
!1004 = !DILocation(line: 51, column: 3, scope: !1005)
!1005 = !DILexicalBlockFile(scope: !996, file: !1, discriminator: 1)
!1006 = !DILocation(line: 52, column: 14, scope: !1001)
!1007 = !DILocation(line: 52, column: 5, scope: !1001)
!1008 = !DILocation(line: 52, column: 9, scope: !1001)
!1009 = !DILocation(line: 52, column: 12, scope: !1001)
!1010 = !DILocation(line: 51, column: 24, scope: !1011)
!1011 = !DILexicalBlockFile(scope: !1001, file: !1, discriminator: 2)
!1012 = !DILocation(line: 51, column: 3, scope: !1011)
!1013 = distinct !{!1013, !1014, !1015}
!1014 = !DILocation(line: 51, column: 3, scope: !996)
!1015 = !DILocation(line: 52, column: 14, scope: !996)
!1016 = !DILocation(line: 53, column: 1, scope: !988)
!1017 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !1019, file: !1018, line: 421, type: !1025, isLocal: false, isDefinition: true, scopeLine: 421, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !1024, variables: !661)
!1018 = !DIFile(filename: "/opt/common/cuda/cuda-7.5.18/include/vector_types.h", directory: "/nfshomes/hzhang86/cuda-blame/tests")
!1019 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !1018, line: 417, size: 96, elements: !1020, identifier: "_ZTS4dim3")
!1020 = !{!1021, !1022, !1023, !1024, !1028, !1037}
!1021 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1019, file: !1018, line: 419, baseType: !457, size: 32)
!1022 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1019, file: !1018, line: 419, baseType: !457, size: 32, offset: 32)
!1023 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1019, file: !1018, line: 419, baseType: !457, size: 32, offset: 64)
!1024 = !DISubprogram(name: "dim3", scope: !1019, file: !1018, line: 421, type: !1025, isLocal: false, isDefinition: false, scopeLine: 421, flags: DIFlagPrototyped, isOptimized: false)
!1025 = !DISubroutineType(types: !1026)
!1026 = !{null, !1027, !457, !457, !457}
!1027 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1019, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1028 = !DISubprogram(name: "dim3", scope: !1019, file: !1018, line: 422, type: !1029, isLocal: false, isDefinition: false, scopeLine: 422, flags: DIFlagPrototyped, isOptimized: false)
!1029 = !DISubroutineType(types: !1030)
!1030 = !{null, !1027, !1031}
!1031 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !1018, line: 383, baseType: !1032)
!1032 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !1018, line: 190, size: 96, elements: !1033, identifier: "_ZTS5uint3")
!1033 = !{!1034, !1035, !1036}
!1034 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1032, file: !1018, line: 192, baseType: !457, size: 32)
!1035 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1032, file: !1018, line: 192, baseType: !457, size: 32, offset: 32)
!1036 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1032, file: !1018, line: 192, baseType: !457, size: 32, offset: 64)
!1037 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !1019, file: !1018, line: 423, type: !1038, isLocal: false, isDefinition: false, scopeLine: 423, flags: DIFlagPrototyped, isOptimized: false)
!1038 = !DISubroutineType(types: !1039)
!1039 = !{!1031, !1027}
!1040 = !DILocalVariable(name: "this", arg: 1, scope: !1017, type: !1041, flags: DIFlagArtificial | DIFlagObjectPointer)
!1041 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1019, size: 64)
!1042 = !DILocation(line: 0, scope: !1017)
!1043 = !DILocalVariable(name: "vx", arg: 2, scope: !1017, file: !1018, line: 421, type: !457)
!1044 = !DILocation(line: 421, column: 43, scope: !1017)
!1045 = !DILocalVariable(name: "vy", arg: 3, scope: !1017, file: !1018, line: 421, type: !457)
!1046 = !DILocation(line: 421, column: 64, scope: !1017)
!1047 = !DILocalVariable(name: "vz", arg: 4, scope: !1017, file: !1018, line: 421, type: !457)
!1048 = !DILocation(line: 421, column: 85, scope: !1017)
!1049 = !DILocation(line: 421, column: 95, scope: !1017)
!1050 = !DILocation(line: 421, column: 97, scope: !1017)
!1051 = !DILocation(line: 421, column: 102, scope: !1017)
!1052 = !DILocation(line: 421, column: 104, scope: !1017)
!1053 = !DILocation(line: 421, column: 109, scope: !1017)
!1054 = !DILocation(line: 421, column: 111, scope: !1017)
!1055 = !DILocation(line: 421, column: 116, scope: !1017)
