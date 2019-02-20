; ModuleID = 'gesummv.bc'
source_filename = "gesummv.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.timezone = type { i32, i32 }
%struct.timeval = type { i64, i64 }
%struct.cudaDeviceProp = type { [256 x i8], i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

@.str = private unnamed_addr constant [35 x i8] c"Error return from gettimeofday: %d\00", align 1
@.str.1 = private unnamed_addr constant [74 x i8] c"Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [32 x i8] c"setting device %d with name %s\0A\00", align 1
@stdout = external global %struct._IO_FILE*, align 8
@.str.3 = private unnamed_addr constant [22 x i8] c"GPU Runtime: %0.6lfs\0A\00", align 1
@.str.4 = private unnamed_addr constant [22 x i8] c"CPU Runtime: %0.6lfs\0A\00", align 1

; Function Attrs: noinline uwtable
define double @_Z7rtclockv() #0 !dbg !576 {
entry:
  %Tzp = alloca %struct.timezone, align 4
  %Tp = alloca %struct.timeval, align 8
  %stat = alloca i32, align 4
  call void @llvm.dbg.declare(metadata %struct.timezone* %Tzp, metadata !581, metadata !587), !dbg !588
  call void @llvm.dbg.declare(metadata %struct.timeval* %Tp, metadata !589, metadata !587), !dbg !598
  call void @llvm.dbg.declare(metadata i32* %stat, metadata !599, metadata !587), !dbg !600
  %call = call i32 @gettimeofday(%struct.timeval* %Tp, %struct.timezone* %Tzp) #8, !dbg !601
  store i32 %call, i32* %stat, align 4, !dbg !602
  %0 = load i32, i32* %stat, align 4, !dbg !603
  %cmp = icmp ne i32 %0, 0, !dbg !605
  br i1 %cmp, label %if.then, label %if.end, !dbg !606

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %stat, align 4, !dbg !607
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str, i32 0, i32 0), i32 %1), !dbg !609
  br label %if.end, !dbg !609

if.end:                                           ; preds = %if.then, %entry
  %tv_sec = getelementptr inbounds %struct.timeval, %struct.timeval* %Tp, i32 0, i32 0, !dbg !610
  %2 = load i64, i64* %tv_sec, align 8, !dbg !610
  %conv = sitofp i64 %2 to double, !dbg !611
  %tv_usec = getelementptr inbounds %struct.timeval, %struct.timeval* %Tp, i32 0, i32 1, !dbg !612
  %3 = load i64, i64* %tv_usec, align 8, !dbg !612
  %conv2 = sitofp i64 %3 to double, !dbg !613
  %mul = fmul double %conv2, 1.000000e-06, !dbg !614
  %add = fadd double %conv, %mul, !dbg !615
  ret double %add, !dbg !616
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i32 @gettimeofday(%struct.timeval*, %struct.timezone*) #2

declare i32 @printf(i8*, ...) #3

; Function Attrs: noinline nounwind uwtable
define float @_Z6absValf(float %a) #4 !dbg !617 {
entry:
  %retval = alloca float, align 4
  %a.addr = alloca float, align 4
  store float %a, float* %a.addr, align 4
  call void @llvm.dbg.declare(metadata float* %a.addr, metadata !618, metadata !587), !dbg !619
  %0 = load float, float* %a.addr, align 4, !dbg !620
  %cmp = fcmp olt float %0, 0.000000e+00, !dbg !622
  br i1 %cmp, label %if.then, label %if.else, !dbg !623

if.then:                                          ; preds = %entry
  %1 = load float, float* %a.addr, align 4, !dbg !624
  %mul = fmul float %1, -1.000000e+00, !dbg !626
  store float %mul, float* %retval, align 4, !dbg !627
  br label %return, !dbg !627

if.else:                                          ; preds = %entry
  %2 = load float, float* %a.addr, align 4, !dbg !628
  store float %2, float* %retval, align 4, !dbg !630
  br label %return, !dbg !630

return:                                           ; preds = %if.else, %if.then
  %3 = load float, float* %retval, align 4, !dbg !631
  ret float %3, !dbg !631
}

; Function Attrs: noinline nounwind uwtable
define float @_Z11percentDiffdd(double %val1, double %val2) #4 !dbg !632 {
entry:
  %retval = alloca float, align 4
  %val1.addr = alloca double, align 8
  %val2.addr = alloca double, align 8
  store double %val1, double* %val1.addr, align 8
  call void @llvm.dbg.declare(metadata double* %val1.addr, metadata !635, metadata !587), !dbg !636
  store double %val2, double* %val2.addr, align 8
  call void @llvm.dbg.declare(metadata double* %val2.addr, metadata !637, metadata !587), !dbg !638
  %0 = load double, double* %val1.addr, align 8, !dbg !639
  %conv = fptrunc double %0 to float, !dbg !639
  %call = call float @_Z6absValf(float %conv), !dbg !641
  %conv1 = fpext float %call to double, !dbg !641
  %cmp = fcmp olt double %conv1, 1.000000e-02, !dbg !642
  br i1 %cmp, label %land.lhs.true, label %if.else, !dbg !643

land.lhs.true:                                    ; preds = %entry
  %1 = load double, double* %val2.addr, align 8, !dbg !644
  %conv2 = fptrunc double %1 to float, !dbg !644
  %call3 = call float @_Z6absValf(float %conv2), !dbg !646
  %conv4 = fpext float %call3 to double, !dbg !646
  %cmp5 = fcmp olt double %conv4, 1.000000e-02, !dbg !647
  br i1 %cmp5, label %if.then, label %if.else, !dbg !648

if.then:                                          ; preds = %land.lhs.true
  store float 0.000000e+00, float* %retval, align 4, !dbg !650
  br label %return, !dbg !650

if.else:                                          ; preds = %land.lhs.true, %entry
  %2 = load double, double* %val1.addr, align 8, !dbg !652
  %3 = load double, double* %val2.addr, align 8, !dbg !654
  %sub = fsub double %2, %3, !dbg !655
  %conv6 = fptrunc double %sub to float, !dbg !652
  %call7 = call float @_Z6absValf(float %conv6), !dbg !656
  %4 = load double, double* %val1.addr, align 8, !dbg !657
  %add = fadd double %4, 0x3E45798EE0000000, !dbg !658
  %conv8 = fptrunc double %add to float, !dbg !657
  %call9 = call float @_Z6absValf(float %conv8), !dbg !659
  %div = fdiv float %call7, %call9, !dbg !661
  %call10 = call float @_Z6absValf(float %div), !dbg !662
  %mul = fmul float 1.000000e+02, %call10, !dbg !664
  store float %mul, float* %retval, align 4, !dbg !665
  br label %return, !dbg !665

return:                                           ; preds = %if.else, %if.then
  %5 = load float, float* %retval, align 4, !dbg !666
  ret float %5, !dbg !666
}

; Function Attrs: noinline nounwind uwtable
define void @_Z7gesummvPfS_S_S_S_(float* %A, float* %B, float* %x, float* %y, float* %tmp) #4 !dbg !667 {
entry:
  %A.addr = alloca float*, align 8
  %B.addr = alloca float*, align 8
  %x.addr = alloca float*, align 8
  %y.addr = alloca float*, align 8
  %tmp.addr = alloca float*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store float* %A, float** %A.addr, align 8
  call void @llvm.dbg.declare(metadata float** %A.addr, metadata !670, metadata !587), !dbg !671
  store float* %B, float** %B.addr, align 8
  call void @llvm.dbg.declare(metadata float** %B.addr, metadata !672, metadata !587), !dbg !673
  store float* %x, float** %x.addr, align 8
  call void @llvm.dbg.declare(metadata float** %x.addr, metadata !674, metadata !587), !dbg !675
  store float* %y, float** %y.addr, align 8
  call void @llvm.dbg.declare(metadata float** %y.addr, metadata !676, metadata !587), !dbg !677
  store float* %tmp, float** %tmp.addr, align 8
  call void @llvm.dbg.declare(metadata float** %tmp.addr, metadata !678, metadata !587), !dbg !679
  call void @llvm.dbg.declare(metadata i32* %i, metadata !680, metadata !587), !dbg !681
  call void @llvm.dbg.declare(metadata i32* %j, metadata !682, metadata !587), !dbg !683
  store i32 0, i32* %i, align 4, !dbg !684
  br label %for.cond, !dbg !686

for.cond:                                         ; preds = %for.inc39, %entry
  %0 = load i32, i32* %i, align 4, !dbg !687
  %cmp = icmp slt i32 %0, 4096, !dbg !690
  br i1 %cmp, label %for.body, label %for.end41, !dbg !691

for.body:                                         ; preds = %for.cond
  %1 = load float*, float** %tmp.addr, align 8, !dbg !693
  %2 = load i32, i32* %i, align 4, !dbg !695
  %idxprom = sext i32 %2 to i64, !dbg !693
  %arrayidx = getelementptr inbounds float, float* %1, i64 %idxprom, !dbg !693
  store float 0.000000e+00, float* %arrayidx, align 4, !dbg !696
  %3 = load float*, float** %y.addr, align 8, !dbg !697
  %4 = load i32, i32* %i, align 4, !dbg !698
  %idxprom3 = sext i32 %4 to i64, !dbg !697
  %arrayidx4 = getelementptr inbounds float, float* %3, i64 %idxprom3, !dbg !697
  store float 0.000000e+00, float* %arrayidx4, align 4, !dbg !699
  store i32 0, i32* %j, align 4, !dbg !700
  br label %for.cond5, !dbg !702

for.cond5:                                        ; preds = %for.inc, %for.body
  %5 = load i32, i32* %j, align 4, !dbg !703
  %cmp6 = icmp slt i32 %5, 4096, !dbg !706
  br i1 %cmp6, label %for.body7, label %for.end, !dbg !707

for.body7:                                        ; preds = %for.cond5
  %6 = load float*, float** %A.addr, align 8, !dbg !709
  %7 = load i32, i32* %i, align 4, !dbg !711
  %mul = mul nsw i32 %7, 4096, !dbg !712
  %8 = load i32, i32* %j, align 4, !dbg !713
  %add = add nsw i32 %mul, %8, !dbg !714
  %idxprom8 = sext i32 %add to i64, !dbg !709
  %arrayidx9 = getelementptr inbounds float, float* %6, i64 %idxprom8, !dbg !709
  %9 = load float, float* %arrayidx9, align 4, !dbg !709
  %10 = load float*, float** %x.addr, align 8, !dbg !715
  %11 = load i32, i32* %j, align 4, !dbg !716
  %idxprom10 = sext i32 %11 to i64, !dbg !715
  %arrayidx11 = getelementptr inbounds float, float* %10, i64 %idxprom10, !dbg !715
  %12 = load float, float* %arrayidx11, align 4, !dbg !715
  %mul12 = fmul float %9, %12, !dbg !717
  %13 = load float*, float** %tmp.addr, align 8, !dbg !718
  %14 = load i32, i32* %i, align 4, !dbg !719
  %idxprom13 = sext i32 %14 to i64, !dbg !718
  %arrayidx14 = getelementptr inbounds float, float* %13, i64 %idxprom13, !dbg !718
  %15 = load float, float* %arrayidx14, align 4, !dbg !718
  %add15 = fadd float %mul12, %15, !dbg !720
  %16 = load float*, float** %tmp.addr, align 8, !dbg !721
  %17 = load i32, i32* %i, align 4, !dbg !722
  %idxprom16 = sext i32 %17 to i64, !dbg !721
  %arrayidx17 = getelementptr inbounds float, float* %16, i64 %idxprom16, !dbg !721
  store float %add15, float* %arrayidx17, align 4, !dbg !723
  %18 = load float*, float** %B.addr, align 8, !dbg !724
  %19 = load i32, i32* %i, align 4, !dbg !725
  %mul18 = mul nsw i32 %19, 4096, !dbg !726
  %20 = load i32, i32* %j, align 4, !dbg !727
  %add19 = add nsw i32 %mul18, %20, !dbg !728
  %idxprom20 = sext i32 %add19 to i64, !dbg !724
  %arrayidx21 = getelementptr inbounds float, float* %18, i64 %idxprom20, !dbg !724
  %21 = load float, float* %arrayidx21, align 4, !dbg !724
  %22 = load float*, float** %x.addr, align 8, !dbg !729
  %23 = load i32, i32* %j, align 4, !dbg !730
  %idxprom22 = sext i32 %23 to i64, !dbg !729
  %arrayidx23 = getelementptr inbounds float, float* %22, i64 %idxprom22, !dbg !729
  %24 = load float, float* %arrayidx23, align 4, !dbg !729
  %mul24 = fmul float %21, %24, !dbg !731
  %25 = load float*, float** %y.addr, align 8, !dbg !732
  %26 = load i32, i32* %i, align 4, !dbg !733
  %idxprom25 = sext i32 %26 to i64, !dbg !732
  %arrayidx26 = getelementptr inbounds float, float* %25, i64 %idxprom25, !dbg !732
  %27 = load float, float* %arrayidx26, align 4, !dbg !732
  %add27 = fadd float %mul24, %27, !dbg !734
  %28 = load float*, float** %y.addr, align 8, !dbg !735
  %29 = load i32, i32* %i, align 4, !dbg !736
  %idxprom28 = sext i32 %29 to i64, !dbg !735
  %arrayidx29 = getelementptr inbounds float, float* %28, i64 %idxprom28, !dbg !735
  store float %add27, float* %arrayidx29, align 4, !dbg !737
  br label %for.inc, !dbg !738

for.inc:                                          ; preds = %for.body7
  %30 = load i32, i32* %j, align 4, !dbg !739
  %inc = add nsw i32 %30, 1, !dbg !739
  store i32 %inc, i32* %j, align 4, !dbg !739
  br label %for.cond5, !dbg !741, !llvm.loop !742

for.end:                                          ; preds = %for.cond5
  %31 = load float*, float** %tmp.addr, align 8, !dbg !745
  %32 = load i32, i32* %i, align 4, !dbg !746
  %idxprom30 = sext i32 %32 to i64, !dbg !745
  %arrayidx31 = getelementptr inbounds float, float* %31, i64 %idxprom30, !dbg !745
  %33 = load float, float* %arrayidx31, align 4, !dbg !745
  %mul32 = fmul float 4.353200e+04, %33, !dbg !747
  %34 = load float*, float** %y.addr, align 8, !dbg !748
  %35 = load i32, i32* %i, align 4, !dbg !749
  %idxprom33 = sext i32 %35 to i64, !dbg !748
  %arrayidx34 = getelementptr inbounds float, float* %34, i64 %idxprom33, !dbg !748
  %36 = load float, float* %arrayidx34, align 4, !dbg !748
  %mul35 = fmul float 1.231300e+04, %36, !dbg !750
  %add36 = fadd float %mul32, %mul35, !dbg !751
  %37 = load float*, float** %y.addr, align 8, !dbg !752
  %38 = load i32, i32* %i, align 4, !dbg !753
  %idxprom37 = sext i32 %38 to i64, !dbg !752
  %arrayidx38 = getelementptr inbounds float, float* %37, i64 %idxprom37, !dbg !752
  store float %add36, float* %arrayidx38, align 4, !dbg !754
  br label %for.inc39, !dbg !755

for.inc39:                                        ; preds = %for.end
  %39 = load i32, i32* %i, align 4, !dbg !756
  %inc40 = add nsw i32 %39, 1, !dbg !756
  store i32 %inc40, i32* %i, align 4, !dbg !756
  br label %for.cond, !dbg !758, !llvm.loop !759

for.end41:                                        ; preds = %for.cond
  ret void, !dbg !762
}

; Function Attrs: noinline nounwind uwtable
define void @_Z4initPfS_(float* %A, float* %x) #4 !dbg !763 {
entry:
  %A.addr = alloca float*, align 8
  %x.addr = alloca float*, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  store float* %A, float** %A.addr, align 8
  call void @llvm.dbg.declare(metadata float** %A.addr, metadata !766, metadata !587), !dbg !767
  store float* %x, float** %x.addr, align 8
  call void @llvm.dbg.declare(metadata float** %x.addr, metadata !768, metadata !587), !dbg !769
  call void @llvm.dbg.declare(metadata i32* %i, metadata !770, metadata !587), !dbg !771
  call void @llvm.dbg.declare(metadata i32* %j, metadata !772, metadata !587), !dbg !773
  store i32 0, i32* %i, align 4, !dbg !774
  br label %for.cond, !dbg !776

for.cond:                                         ; preds = %for.inc10, %entry
  %0 = load i32, i32* %i, align 4, !dbg !777
  %cmp = icmp slt i32 %0, 4096, !dbg !780
  br i1 %cmp, label %for.body, label %for.end12, !dbg !781

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4, !dbg !783
  %conv = sitofp i32 %1 to float, !dbg !783
  %div = fdiv float %conv, 4.096000e+03, !dbg !785
  %2 = load float*, float** %x.addr, align 8, !dbg !786
  %3 = load i32, i32* %i, align 4, !dbg !787
  %idxprom = sext i32 %3 to i64, !dbg !786
  %arrayidx = getelementptr inbounds float, float* %2, i64 %idxprom, !dbg !786
  store float %div, float* %arrayidx, align 4, !dbg !788
  store i32 0, i32* %j, align 4, !dbg !789
  br label %for.cond1, !dbg !791

for.cond1:                                        ; preds = %for.inc, %for.body
  %4 = load i32, i32* %j, align 4, !dbg !792
  %cmp2 = icmp slt i32 %4, 4096, !dbg !795
  br i1 %cmp2, label %for.body3, label %for.end, !dbg !796

for.body3:                                        ; preds = %for.cond1
  %5 = load i32, i32* %i, align 4, !dbg !798
  %conv4 = sitofp i32 %5 to float, !dbg !798
  %6 = load i32, i32* %j, align 4, !dbg !800
  %conv5 = sitofp i32 %6 to float, !dbg !800
  %mul = fmul float %conv4, %conv5, !dbg !801
  %div6 = fdiv float %mul, 4.096000e+03, !dbg !802
  %7 = load float*, float** %A.addr, align 8, !dbg !803
  %8 = load i32, i32* %i, align 4, !dbg !804
  %mul7 = mul nsw i32 %8, 4096, !dbg !805
  %9 = load i32, i32* %j, align 4, !dbg !806
  %add = add nsw i32 %mul7, %9, !dbg !807
  %idxprom8 = sext i32 %add to i64, !dbg !803
  %arrayidx9 = getelementptr inbounds float, float* %7, i64 %idxprom8, !dbg !803
  store float %div6, float* %arrayidx9, align 4, !dbg !808
  br label %for.inc, !dbg !809

for.inc:                                          ; preds = %for.body3
  %10 = load i32, i32* %j, align 4, !dbg !810
  %inc = add nsw i32 %10, 1, !dbg !810
  store i32 %inc, i32* %j, align 4, !dbg !810
  br label %for.cond1, !dbg !812, !llvm.loop !813

for.end:                                          ; preds = %for.cond1
  br label %for.inc10, !dbg !816

for.inc10:                                        ; preds = %for.end
  %11 = load i32, i32* %i, align 4, !dbg !817
  %inc11 = add nsw i32 %11, 1, !dbg !817
  store i32 %inc11, i32* %i, align 4, !dbg !817
  br label %for.cond, !dbg !819, !llvm.loop !820

for.end12:                                        ; preds = %for.cond
  ret void, !dbg !823
}

; Function Attrs: noinline uwtable
define void @_Z14compareResultsPfS_(float* %y, float* %y_outputFromGpu) #0 !dbg !824 {
entry:
  %y.addr = alloca float*, align 8
  %y_outputFromGpu.addr = alloca float*, align 8
  %i = alloca i32, align 4
  %fail = alloca i32, align 4
  store float* %y, float** %y.addr, align 8
  call void @llvm.dbg.declare(metadata float** %y.addr, metadata !825, metadata !587), !dbg !826
  store float* %y_outputFromGpu, float** %y_outputFromGpu.addr, align 8
  call void @llvm.dbg.declare(metadata float** %y_outputFromGpu.addr, metadata !827, metadata !587), !dbg !828
  call void @llvm.dbg.declare(metadata i32* %i, metadata !829, metadata !587), !dbg !830
  call void @llvm.dbg.declare(metadata i32* %fail, metadata !831, metadata !587), !dbg !832
  store i32 0, i32* %fail, align 4, !dbg !833
  store i32 0, i32* %i, align 4, !dbg !834
  br label %for.cond, !dbg !836

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4, !dbg !837
  %cmp = icmp slt i32 %0, 4096, !dbg !840
  br i1 %cmp, label %for.body, label %for.end, !dbg !841

for.body:                                         ; preds = %for.cond
  %1 = load float*, float** %y.addr, align 8, !dbg !843
  %2 = load i32, i32* %i, align 4, !dbg !846
  %idxprom = sext i32 %2 to i64, !dbg !843
  %arrayidx = getelementptr inbounds float, float* %1, i64 %idxprom, !dbg !843
  %3 = load float, float* %arrayidx, align 4, !dbg !843
  %conv = fpext float %3 to double, !dbg !843
  %4 = load float*, float** %y_outputFromGpu.addr, align 8, !dbg !847
  %5 = load i32, i32* %i, align 4, !dbg !848
  %idxprom1 = sext i32 %5 to i64, !dbg !847
  %arrayidx2 = getelementptr inbounds float, float* %4, i64 %idxprom1, !dbg !847
  %6 = load float, float* %arrayidx2, align 4, !dbg !847
  %conv3 = fpext float %6 to double, !dbg !847
  %call = call float @_Z11percentDiffdd(double %conv, double %conv3), !dbg !849
  %conv4 = fpext float %call to double, !dbg !849
  %cmp5 = fcmp ogt double %conv4, 5.000000e-02, !dbg !850
  br i1 %cmp5, label %if.then, label %if.end, !dbg !851

if.then:                                          ; preds = %for.body
  %7 = load i32, i32* %fail, align 4, !dbg !852
  %inc = add nsw i32 %7, 1, !dbg !852
  store i32 %inc, i32* %fail, align 4, !dbg !852
  br label %if.end, !dbg !854

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc, !dbg !855

for.inc:                                          ; preds = %if.end
  %8 = load i32, i32* %i, align 4, !dbg !856
  %inc6 = add nsw i32 %8, 1, !dbg !856
  store i32 %inc6, i32* %i, align 4, !dbg !856
  br label %for.cond, !dbg !858, !llvm.loop !859

for.end:                                          ; preds = %for.cond
  %9 = load i32, i32* %fail, align 4, !dbg !862
  %call7 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([74 x i8], [74 x i8]* @.str.1, i32 0, i32 0), double 5.000000e-02, i32 %9), !dbg !863
  ret void, !dbg !864
}

; Function Attrs: noinline uwtable
define void @_Z13GPU_argv_initv() #0 !dbg !865 {
entry:
  %deviceProp = alloca %struct.cudaDeviceProp, align 8
  call void @llvm.dbg.declare(metadata %struct.cudaDeviceProp* %deviceProp, metadata !866, metadata !587), !dbg !944
  %call = call i32 @cudaGetDeviceProperties(%struct.cudaDeviceProp* %deviceProp, i32 0), !dbg !945
  %name = getelementptr inbounds %struct.cudaDeviceProp, %struct.cudaDeviceProp* %deviceProp, i32 0, i32 0, !dbg !946
  %arraydecay = getelementptr inbounds [256 x i8], [256 x i8]* %name, i32 0, i32 0, !dbg !947
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.2, i32 0, i32 0), i32 0, i8* %arraydecay), !dbg !948
  %call2 = call i32 @cudaSetDevice(i32 0), !dbg !949
  ret void, !dbg !950
}

declare i32 @cudaGetDeviceProperties(%struct.cudaDeviceProp*, i32) #3

declare i32 @cudaSetDevice(i32) #3

; Function Attrs: noinline uwtable
define void @_Z14gesummv_kernelPfS_S_S_S_(float* %a, float* %b, float* %x, float* %y, float* %tmp) #0 !dbg !951 {
entry:
  %a.addr = alloca float*, align 8
  %b.addr = alloca float*, align 8
  %x.addr = alloca float*, align 8
  %y.addr = alloca float*, align 8
  %tmp.addr = alloca float*, align 8
  store float* %a, float** %a.addr, align 8
  call void @llvm.dbg.declare(metadata float** %a.addr, metadata !952, metadata !587), !dbg !953
  store float* %b, float** %b.addr, align 8
  call void @llvm.dbg.declare(metadata float** %b.addr, metadata !954, metadata !587), !dbg !955
  store float* %x, float** %x.addr, align 8
  call void @llvm.dbg.declare(metadata float** %x.addr, metadata !956, metadata !587), !dbg !957
  store float* %y, float** %y.addr, align 8
  call void @llvm.dbg.declare(metadata float** %y.addr, metadata !958, metadata !587), !dbg !959
  store float* %tmp, float** %tmp.addr, align 8
  call void @llvm.dbg.declare(metadata float** %tmp.addr, metadata !960, metadata !587), !dbg !961
  %0 = bitcast float** %a.addr to i8*, !dbg !962
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !962
  %2 = icmp eq i32 %1, 0, !dbg !962
  br i1 %2, label %setup.next, label %setup.end, !dbg !962

setup.next:                                       ; preds = %entry
  %3 = bitcast float** %b.addr to i8*, !dbg !963
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !963
  %5 = icmp eq i32 %4, 0, !dbg !963
  br i1 %5, label %setup.next1, label %setup.end, !dbg !963

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast float** %x.addr to i8*, !dbg !965
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !965
  %8 = icmp eq i32 %7, 0, !dbg !965
  br i1 %8, label %setup.next2, label %setup.end, !dbg !965

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast float** %y.addr to i8*, !dbg !967
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !967
  %11 = icmp eq i32 %10, 0, !dbg !967
  br i1 %11, label %setup.next3, label %setup.end, !dbg !967

setup.next3:                                      ; preds = %setup.next2
  %12 = bitcast float** %tmp.addr to i8*, !dbg !969
  %13 = call i32 @cudaSetupArgument(i8* %12, i64 8, i64 32), !dbg !969
  %14 = icmp eq i32 %13, 0, !dbg !969
  br i1 %14, label %setup.next4, label %setup.end, !dbg !969

setup.next4:                                      ; preds = %setup.next3
  %15 = call i32 @cudaLaunch(i8* bitcast (void (float*, float*, float*, float*, float*)* @_Z14gesummv_kernelPfS_S_S_S_ to i8*)), !dbg !971
  br label %setup.end, !dbg !971

setup.end:                                        ; preds = %setup.next4, %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !973
}

declare i32 @cudaSetupArgument(i8*, i64, i64)

declare i32 @cudaLaunch(i8*)

; Function Attrs: noinline uwtable
define void @_Z11gesummvCudaPfS_S_S_S_S_(float* %A, float* %B, float* %x, float* %y, float* %tmp, float* %y_outputFromGpu) #0 !dbg !974 {
entry:
  %A.addr = alloca float*, align 8
  %B.addr = alloca float*, align 8
  %x.addr = alloca float*, align 8
  %y.addr = alloca float*, align 8
  %tmp.addr = alloca float*, align 8
  %y_outputFromGpu.addr = alloca float*, align 8
  %t_start = alloca double, align 8
  %t_end = alloca double, align 8
  %A_gpu = alloca float*, align 8
  %B_gpu = alloca float*, align 8
  %x_gpu = alloca float*, align 8
  %y_gpu = alloca float*, align 8
  %tmp_gpu = alloca float*, align 8
  %block = alloca %struct.dim3, align 4
  %grid = alloca %struct.dim3, align 4
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp24 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp24.coerce = alloca { i64, i32 }, align 4
  store float* %A, float** %A.addr, align 8
  call void @llvm.dbg.declare(metadata float** %A.addr, metadata !977, metadata !587), !dbg !978
  store float* %B, float** %B.addr, align 8
  call void @llvm.dbg.declare(metadata float** %B.addr, metadata !979, metadata !587), !dbg !980
  store float* %x, float** %x.addr, align 8
  call void @llvm.dbg.declare(metadata float** %x.addr, metadata !981, metadata !587), !dbg !982
  store float* %y, float** %y.addr, align 8
  call void @llvm.dbg.declare(metadata float** %y.addr, metadata !983, metadata !587), !dbg !984
  store float* %tmp, float** %tmp.addr, align 8
  call void @llvm.dbg.declare(metadata float** %tmp.addr, metadata !985, metadata !587), !dbg !986
  store float* %y_outputFromGpu, float** %y_outputFromGpu.addr, align 8
  call void @llvm.dbg.declare(metadata float** %y_outputFromGpu.addr, metadata !987, metadata !587), !dbg !988
  call void @llvm.dbg.declare(metadata double* %t_start, metadata !989, metadata !587), !dbg !990
  call void @llvm.dbg.declare(metadata double* %t_end, metadata !991, metadata !587), !dbg !992
  call void @llvm.dbg.declare(metadata float** %A_gpu, metadata !993, metadata !587), !dbg !994
  call void @llvm.dbg.declare(metadata float** %B_gpu, metadata !995, metadata !587), !dbg !996
  call void @llvm.dbg.declare(metadata float** %x_gpu, metadata !997, metadata !587), !dbg !998
  call void @llvm.dbg.declare(metadata float** %y_gpu, metadata !999, metadata !587), !dbg !1000
  call void @llvm.dbg.declare(metadata float** %tmp_gpu, metadata !1001, metadata !587), !dbg !1002
  %0 = bitcast float** %A_gpu to i8**, !dbg !1003
  %call = call i32 @cudaMalloc(i8** %0, i64 67108864), !dbg !1004
  %1 = bitcast float** %B_gpu to i8**, !dbg !1005
  %call8 = call i32 @cudaMalloc(i8** %1, i64 67108864), !dbg !1006
  %2 = bitcast float** %x_gpu to i8**, !dbg !1007
  %call9 = call i32 @cudaMalloc(i8** %2, i64 16384), !dbg !1008
  %3 = bitcast float** %y_gpu to i8**, !dbg !1009
  %call10 = call i32 @cudaMalloc(i8** %3, i64 16384), !dbg !1010
  %4 = bitcast float** %tmp_gpu to i8**, !dbg !1011
  %call11 = call i32 @cudaMalloc(i8** %4, i64 16384), !dbg !1012
  %5 = load float*, float** %A_gpu, align 8, !dbg !1013
  %6 = bitcast float* %5 to i8*, !dbg !1013
  %7 = load float*, float** %A.addr, align 8, !dbg !1014
  %8 = bitcast float* %7 to i8*, !dbg !1014
  %call12 = call i32 @cudaMemcpy(i8* %6, i8* %8, i64 67108864, i32 1), !dbg !1015
  %9 = load float*, float** %B_gpu, align 8, !dbg !1016
  %10 = bitcast float* %9 to i8*, !dbg !1016
  %11 = load float*, float** %B.addr, align 8, !dbg !1017
  %12 = bitcast float* %11 to i8*, !dbg !1017
  %call13 = call i32 @cudaMemcpy(i8* %10, i8* %12, i64 67108864, i32 1), !dbg !1018
  %13 = load float*, float** %x_gpu, align 8, !dbg !1019
  %14 = bitcast float* %13 to i8*, !dbg !1019
  %15 = load float*, float** %x.addr, align 8, !dbg !1020
  %16 = bitcast float* %15 to i8*, !dbg !1020
  %call14 = call i32 @cudaMemcpy(i8* %14, i8* %16, i64 16384, i32 1), !dbg !1021
  %17 = load float*, float** %y_gpu, align 8, !dbg !1022
  %18 = bitcast float* %17 to i8*, !dbg !1022
  %19 = load float*, float** %y.addr, align 8, !dbg !1023
  %20 = bitcast float* %19 to i8*, !dbg !1023
  %call15 = call i32 @cudaMemcpy(i8* %18, i8* %20, i64 16384, i32 1), !dbg !1024
  %21 = load float*, float** %tmp_gpu, align 8, !dbg !1025
  %22 = bitcast float* %21 to i8*, !dbg !1025
  %23 = load float*, float** %tmp.addr, align 8, !dbg !1026
  %24 = bitcast float* %23 to i8*, !dbg !1026
  %call16 = call i32 @cudaMemcpy(i8* %22, i8* %24, i64 16384, i32 1), !dbg !1027
  call void @llvm.dbg.declare(metadata %struct.dim3* %block, metadata !1028, metadata !587), !dbg !1052
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %block, i32 256, i32 1, i32 1), !dbg !1052
  call void @llvm.dbg.declare(metadata %struct.dim3* %grid, metadata !1053, metadata !587), !dbg !1054
  %x19 = getelementptr inbounds %struct.dim3, %struct.dim3* %block, i32 0, i32 0, !dbg !1055
  %25 = load i32, i32* %x19, align 4, !dbg !1055
  %conv = uitofp i32 %25 to float, !dbg !1056
  %div = fdiv float 4.096000e+03, %conv, !dbg !1057
  %conv20 = fpext float %div to double, !dbg !1058
  %call21 = call double @ceil(double %conv20) #1, !dbg !1059
  %conv22 = fptoui double %call21 to i32, !dbg !1059
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %grid, i32 %conv22, i32 1, i32 1), !dbg !1060
  %call23 = call double @_Z7rtclockv(), !dbg !1062
  store double %call23, double* %t_start, align 8, !dbg !1063
  %26 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !1064
  %27 = bitcast %struct.dim3* %grid to i8*, !dbg !1064
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %26, i8* %27, i64 12, i32 4, i1 false), !dbg !1064
  %28 = bitcast %struct.dim3* %agg.tmp24 to i8*, !dbg !1065
  %29 = bitcast %struct.dim3* %block to i8*, !dbg !1065
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %28, i8* %29, i64 12, i32 4, i1 false), !dbg !1065
  %30 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !1066
  %31 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !1066
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %30, i8* %31, i64 12, i32 4, i1 false), !dbg !1066
  %32 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !1066
  %33 = load i64, i64* %32, align 4, !dbg !1066
  %34 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !1066
  %35 = load i32, i32* %34, align 4, !dbg !1066
  %36 = bitcast { i64, i32 }* %agg.tmp24.coerce to i8*, !dbg !1066
  %37 = bitcast %struct.dim3* %agg.tmp24 to i8*, !dbg !1066
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %36, i8* %37, i64 12, i32 4, i1 false), !dbg !1066
  %38 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp24.coerce, i32 0, i32 0, !dbg !1066
  %39 = load i64, i64* %38, align 4, !dbg !1066
  %40 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp24.coerce, i32 0, i32 1, !dbg !1066
  %41 = load i32, i32* %40, align 4, !dbg !1066
  %call25 = call i32 @cudaConfigureCall(i64 %33, i32 %35, i64 %39, i32 %41, i64 0, %struct.CUstream_st* null), !dbg !1066
  %tobool = icmp ne i32 %call25, 0, !dbg !1066
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !1067

kcall.configok:                                   ; preds = %entry
  %42 = load float*, float** %A_gpu, align 8, !dbg !1068
  %43 = load float*, float** %B_gpu, align 8, !dbg !1069
  %44 = load float*, float** %x_gpu, align 8, !dbg !1070
  %45 = load float*, float** %y_gpu, align 8, !dbg !1071
  %46 = load float*, float** %tmp_gpu, align 8, !dbg !1072
  call void @_Z14gesummv_kernelPfS_S_S_S_(float* %42, float* %43, float* %44, float* %45, float* %46), !dbg !1073
  br label %kcall.end, !dbg !1073

kcall.end:                                        ; preds = %kcall.configok, %entry
  %call26 = call i32 @cudaThreadSynchronize(), !dbg !1074
  %call27 = call double @_Z7rtclockv(), !dbg !1075
  store double %call27, double* %t_end, align 8, !dbg !1076
  %47 = load float*, float** %y_outputFromGpu.addr, align 8, !dbg !1077
  %48 = bitcast float* %47 to i8*, !dbg !1077
  %49 = load float*, float** %y_gpu, align 8, !dbg !1078
  %50 = bitcast float* %49 to i8*, !dbg !1078
  %call28 = call i32 @cudaMemcpy(i8* %48, i8* %50, i64 16384, i32 2), !dbg !1079
  %51 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8, !dbg !1080
  %52 = load double, double* %t_end, align 8, !dbg !1081
  %53 = load double, double* %t_start, align 8, !dbg !1082
  %sub = fsub double %52, %53, !dbg !1083
  %call29 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %51, i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.3, i32 0, i32 0), double %sub), !dbg !1084
  ret void, !dbg !1085
}

declare i32 @cudaMalloc(i8**, i64) #3

declare i32 @cudaMemcpy(i8*, i8*, i64, i32) #3

; Function Attrs: noinline nounwind uwtable
define linkonce_odr void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #4 comdat align 2 !dbg !1086 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %this.addr, metadata !1087, metadata !587), !dbg !1089
  store i32 %vx, i32* %vx.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vx.addr, metadata !1090, metadata !587), !dbg !1091
  store i32 %vy, i32* %vy.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vy.addr, metadata !1092, metadata !587), !dbg !1093
  store i32 %vz, i32* %vz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vz.addr, metadata !1094, metadata !587), !dbg !1095
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0, !dbg !1096
  %0 = load i32, i32* %vx.addr, align 4, !dbg !1097
  store i32 %0, i32* %x, align 4, !dbg !1096
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1, !dbg !1098
  %1 = load i32, i32* %vy.addr, align 4, !dbg !1099
  store i32 %1, i32* %y, align 4, !dbg !1098
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2, !dbg !1100
  %2 = load i32, i32* %vz.addr, align 4, !dbg !1101
  store i32 %2, i32* %z, align 4, !dbg !1100
  ret void, !dbg !1102
}

; Function Attrs: nounwind readnone
declare double @ceil(double) #5

declare i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #6

declare i32 @cudaThreadSynchronize() #3

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...) #3

; Function Attrs: noinline norecurse uwtable
define i32 @main(i32 %argc, i8** %argv) #7 !dbg !1103 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %t_start = alloca double, align 8
  %t_end = alloca double, align 8
  %A = alloca float*, align 8
  %B = alloca float*, align 8
  %x = alloca float*, align 8
  %y = alloca float*, align 8
  %y_outputFromGpu = alloca float*, align 8
  %tmp = alloca float*, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !1106, metadata !587), !dbg !1107
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !1108, metadata !587), !dbg !1109
  call void @llvm.dbg.declare(metadata double* %t_start, metadata !1110, metadata !587), !dbg !1111
  call void @llvm.dbg.declare(metadata double* %t_end, metadata !1112, metadata !587), !dbg !1113
  call void @llvm.dbg.declare(metadata float** %A, metadata !1114, metadata !587), !dbg !1115
  call void @llvm.dbg.declare(metadata float** %B, metadata !1116, metadata !587), !dbg !1117
  call void @llvm.dbg.declare(metadata float** %x, metadata !1118, metadata !587), !dbg !1119
  call void @llvm.dbg.declare(metadata float** %y, metadata !1120, metadata !587), !dbg !1121
  call void @llvm.dbg.declare(metadata float** %y_outputFromGpu, metadata !1122, metadata !587), !dbg !1123
  call void @llvm.dbg.declare(metadata float** %tmp, metadata !1124, metadata !587), !dbg !1125
  %call = call noalias i8* @malloc(i64 67108864) #8, !dbg !1126
  %0 = bitcast i8* %call to float*, !dbg !1127
  store float* %0, float** %A, align 8, !dbg !1128
  %call1 = call noalias i8* @malloc(i64 67108864) #8, !dbg !1129
  %1 = bitcast i8* %call1 to float*, !dbg !1130
  store float* %1, float** %B, align 8, !dbg !1131
  %call2 = call noalias i8* @malloc(i64 16384) #8, !dbg !1132
  %2 = bitcast i8* %call2 to float*, !dbg !1133
  store float* %2, float** %x, align 8, !dbg !1134
  %call3 = call noalias i8* @malloc(i64 16384) #8, !dbg !1135
  %3 = bitcast i8* %call3 to float*, !dbg !1136
  store float* %3, float** %y, align 8, !dbg !1137
  %call4 = call noalias i8* @malloc(i64 16384) #8, !dbg !1138
  %4 = bitcast i8* %call4 to float*, !dbg !1139
  store float* %4, float** %y_outputFromGpu, align 8, !dbg !1140
  %call5 = call noalias i8* @malloc(i64 16384) #8, !dbg !1141
  %5 = bitcast i8* %call5 to float*, !dbg !1142
  store float* %5, float** %tmp, align 8, !dbg !1143
  %6 = load float*, float** %A, align 8, !dbg !1144
  %7 = load float*, float** %x, align 8, !dbg !1145
  call void @_Z4initPfS_(float* %6, float* %7), !dbg !1146
  call void @_Z13GPU_argv_initv(), !dbg !1147
  call void @_Z9initTracev(), !dbg !1148
  %8 = load float*, float** %A, align 8, !dbg !1149
  %9 = load float*, float** %B, align 8, !dbg !1150
  %10 = load float*, float** %x, align 8, !dbg !1151
  %11 = load float*, float** %y, align 8, !dbg !1152
  %12 = load float*, float** %tmp, align 8, !dbg !1153
  %13 = load float*, float** %y_outputFromGpu, align 8, !dbg !1154
  call void @_Z11gesummvCudaPfS_S_S_S_S_(float* %8, float* %9, float* %10, float* %11, float* %12, float* %13), !dbg !1155
  call void @_Z9finiTracev(), !dbg !1156
  %call6 = call double @_Z7rtclockv(), !dbg !1157
  store double %call6, double* %t_start, align 8, !dbg !1158
  %14 = load float*, float** %A, align 8, !dbg !1159
  %15 = load float*, float** %B, align 8, !dbg !1160
  %16 = load float*, float** %x, align 8, !dbg !1161
  %17 = load float*, float** %y, align 8, !dbg !1162
  %18 = load float*, float** %tmp, align 8, !dbg !1163
  call void @_Z7gesummvPfS_S_S_S_(float* %14, float* %15, float* %16, float* %17, float* %18), !dbg !1164
  %call7 = call double @_Z7rtclockv(), !dbg !1165
  store double %call7, double* %t_end, align 8, !dbg !1166
  %19 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8, !dbg !1167
  %20 = load double, double* %t_end, align 8, !dbg !1168
  %21 = load double, double* %t_start, align 8, !dbg !1169
  %sub = fsub double %20, %21, !dbg !1170
  %call8 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %19, i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.4, i32 0, i32 0), double %sub), !dbg !1171
  %22 = load float*, float** %y, align 8, !dbg !1172
  %23 = load float*, float** %y_outputFromGpu, align 8, !dbg !1173
  call void @_Z14compareResultsPfS_(float* %22, float* %23), !dbg !1174
  %24 = load float*, float** %A, align 8, !dbg !1175
  %25 = bitcast float* %24 to i8*, !dbg !1175
  call void @free(i8* %25) #8, !dbg !1176
  %26 = load float*, float** %B, align 8, !dbg !1177
  %27 = bitcast float* %26 to i8*, !dbg !1177
  call void @free(i8* %27) #8, !dbg !1178
  %28 = load float*, float** %x, align 8, !dbg !1179
  %29 = bitcast float* %28 to i8*, !dbg !1179
  call void @free(i8* %29) #8, !dbg !1180
  %30 = load float*, float** %y, align 8, !dbg !1181
  %31 = bitcast float* %30 to i8*, !dbg !1181
  call void @free(i8* %31) #8, !dbg !1182
  %32 = load float*, float** %y_outputFromGpu, align 8, !dbg !1183
  %33 = bitcast float* %32 to i8*, !dbg !1183
  call void @free(i8* %33) #8, !dbg !1184
  %34 = load float*, float** %tmp, align 8, !dbg !1185
  %35 = bitcast float* %34 to i8*, !dbg !1185
  call void @free(i8* %35) #8, !dbg !1186
  ret i32 0, !dbg !1187
}

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) #2

declare void @_Z9initTracev() #3

declare void @_Z9finiTracev() #3

; Function Attrs: nounwind
declare void @free(i8*) #2

attributes #0 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly nounwind }
attributes #7 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!573, !574}
!llvm.ident = !{!575}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.1 (tags/RELEASE_401/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !11, imports: !18)
!1 = !DIFile(filename: "gesummv.cu", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaMemcpyKind", file: !4, line: 807, size: 32, elements: !5, identifier: "_ZTS14cudaMemcpyKind")
!4 = !DIFile(filename: "/software/cuda/8.0.44/rel70/x86_64/include/driver_types.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!5 = !{!6, !7, !8, !9, !10}
!6 = !DIEnumerator(name: "cudaMemcpyHostToHost", value: 0)
!7 = !DIEnumerator(name: "cudaMemcpyHostToDevice", value: 1)
!8 = !DIEnumerator(name: "cudaMemcpyDeviceToHost", value: 2)
!9 = !DIEnumerator(name: "cudaMemcpyDeviceToDevice", value: 3)
!10 = !DIEnumerator(name: "cudaMemcpyDefault", value: 4)
!11 = !{!12, !14, !16, !13, !17}
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "DATA_TYPE", file: !1, line: 39, baseType: !13)
!13 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!16 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!18 = !{!19, !26, !30, !32, !34, !36, !38, !42, !44, !46, !48, !50, !52, !54, !56, !58, !60, !62, !64, !66, !68, !70, !74, !76, !78, !80, !85, !90, !92, !94, !99, !103, !105, !107, !109, !111, !113, !115, !117, !119, !124, !128, !130, !132, !136, !138, !140, !142, !144, !146, !150, !152, !154, !159, !167, !171, !173, !175, !179, !181, !183, !187, !189, !191, !195, !197, !199, !201, !203, !205, !207, !209, !211, !213, !218, !220, !222, !226, !228, !230, !232, !234, !236, !238, !240, !244, !248, !250, !252, !257, !259, !261, !263, !265, !267, !269, !273, !279, !283, !287, !292, !294, !298, !302, !315, !319, !323, !327, !331, !336, !338, !342, !346, !350, !358, !362, !366, !370, !374, !378, !384, !388, !392, !394, !402, !406, !414, !416, !418, !422, !426, !430, !435, !439, !444, !445, !446, !447, !450, !451, !452, !453, !454, !455, !456, !459, !461, !463, !465, !467, !469, !471, !473, !476, !478, !480, !482, !484, !486, !488, !490, !492, !494, !496, !498, !500, !502, !504, !506, !508, !510, !512, !514, !516, !518, !520, !522, !524, !526, !528, !530, !532, !534, !536, !538, !540, !544, !545, !547, !549, !551, !553, !555, !557, !559, !561, !563, !565, !567, !569, !571}
!19 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !22, line: 201)
!20 = !DINamespace(name: "std", scope: null, file: !21, line: 195)
!21 = !DIFile(filename: "/home/hzhang86/packages/llvm/bin/../lib/clang/4.0.1/include/__clang_cuda_math_forward_declares.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!22 = !DISubprogram(name: "abs", linkageName: "_ZL3absx", scope: !21, file: !21, line: 44, type: !23, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!23 = !DISubroutineType(types: !24)
!24 = !{!25, !25}
!25 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!26 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !27, line: 202)
!27 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !21, file: !21, line: 46, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!28 = !DISubroutineType(types: !29)
!29 = !{!13, !13}
!30 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !31, line: 203)
!31 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !21, file: !21, line: 48, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!32 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !33, line: 204)
!33 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !21, file: !21, line: 50, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!34 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !35, line: 205)
!35 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !21, file: !21, line: 52, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!36 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !37, line: 206)
!37 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !21, file: !21, line: 56, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!38 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !39, line: 207)
!39 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !21, file: !21, line: 54, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!40 = !DISubroutineType(types: !41)
!41 = !{!13, !13, !13}
!42 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !43, line: 208)
!43 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !21, file: !21, line: 58, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!44 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !45, line: 209)
!45 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !21, file: !21, line: 60, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!46 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !47, line: 210)
!47 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !21, file: !21, line: 62, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!48 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !49, line: 211)
!49 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !21, file: !21, line: 64, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!50 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !51, line: 212)
!51 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !21, file: !21, line: 66, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!52 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !53, line: 213)
!53 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !21, file: !21, line: 68, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!54 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !55, line: 214)
!55 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !21, file: !21, line: 72, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!56 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !57, line: 215)
!57 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !21, file: !21, line: 70, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!58 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !59, line: 216)
!59 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !21, file: !21, line: 76, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!60 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !61, line: 217)
!61 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !21, file: !21, line: 74, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!62 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !63, line: 218)
!63 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !21, file: !21, line: 78, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!64 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !65, line: 219)
!65 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !21, file: !21, line: 80, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!66 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !67, line: 220)
!67 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !21, file: !21, line: 82, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!68 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !69, line: 221)
!69 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !21, file: !21, line: 84, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!70 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !71, line: 222)
!71 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !21, file: !21, line: 86, type: !72, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!72 = !DISubroutineType(types: !73)
!73 = !{!13, !13, !13, !13}
!74 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !75, line: 223)
!75 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !21, file: !21, line: 88, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!76 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !77, line: 224)
!77 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !21, file: !21, line: 90, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!78 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !79, line: 225)
!79 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !21, file: !21, line: 92, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!80 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !81, line: 226)
!81 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !21, file: !21, line: 94, type: !82, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!82 = !DISubroutineType(types: !83)
!83 = !{!84, !13}
!84 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!85 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !86, line: 227)
!86 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !21, file: !21, line: 96, type: !87, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!87 = !DISubroutineType(types: !88)
!88 = !{!13, !13, !89}
!89 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !84, size: 64)
!90 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !91, line: 228)
!91 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !21, file: !21, line: 98, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!92 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !93, line: 229)
!93 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !21, file: !21, line: 100, type: !82, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!94 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !95, line: 230)
!95 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !21, file: !21, line: 102, type: !96, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!96 = !DISubroutineType(types: !97)
!97 = !{!98, !13}
!98 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!99 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !100, line: 231)
!100 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !21, file: !21, line: 106, type: !101, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!101 = !DISubroutineType(types: !102)
!102 = !{!98, !13, !13}
!103 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !104, line: 232)
!104 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !21, file: !21, line: 105, type: !101, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!105 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !106, line: 233)
!106 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !21, file: !21, line: 108, type: !96, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!107 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !108, line: 234)
!108 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !21, file: !21, line: 112, type: !101, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!109 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !110, line: 235)
!110 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !21, file: !21, line: 111, type: !101, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!111 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !112, line: 236)
!112 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !21, file: !21, line: 114, type: !101, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!113 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !114, line: 237)
!114 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !21, file: !21, line: 116, type: !96, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!115 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !116, line: 238)
!116 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !21, file: !21, line: 118, type: !96, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!117 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !118, line: 239)
!118 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !21, file: !21, line: 120, type: !101, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!119 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !120, line: 240)
!120 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !21, file: !21, line: 121, type: !121, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!121 = !DISubroutineType(types: !122)
!122 = !{!123, !123}
!123 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!124 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !125, line: 241)
!125 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !21, file: !21, line: 123, type: !126, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!126 = !DISubroutineType(types: !127)
!127 = !{!13, !13, !84}
!128 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !129, line: 242)
!129 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !21, file: !21, line: 125, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !131, line: 243)
!131 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !21, file: !21, line: 126, type: !23, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!132 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !133, line: 244)
!133 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !21, file: !21, line: 128, type: !134, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!134 = !DISubroutineType(types: !135)
!135 = !{!25, !13}
!136 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !137, line: 245)
!137 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !21, file: !21, line: 138, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!138 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !139, line: 246)
!139 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !21, file: !21, line: 130, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!140 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !141, line: 247)
!141 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !21, file: !21, line: 132, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!142 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !143, line: 248)
!143 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !21, file: !21, line: 134, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!144 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !145, line: 249)
!145 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !21, file: !21, line: 136, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!146 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !147, line: 250)
!147 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !21, file: !21, line: 140, type: !148, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!148 = !DISubroutineType(types: !149)
!149 = !{!123, !13}
!150 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !151, line: 251)
!151 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !21, file: !21, line: 142, type: !148, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!152 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !153, line: 252)
!153 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !21, file: !21, line: 143, type: !134, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!154 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !155, line: 253)
!155 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !21, file: !21, line: 145, type: !156, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!156 = !DISubroutineType(types: !157)
!157 = !{!13, !13, !158}
!158 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!159 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !160, line: 254)
!160 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !21, file: !21, line: 146, type: !161, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!161 = !DISubroutineType(types: !162)
!162 = !{!163, !164}
!163 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!164 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !165, size: 64)
!165 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !166)
!166 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!167 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !168, line: 255)
!168 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !21, file: !21, line: 147, type: !169, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!169 = !DISubroutineType(types: !170)
!170 = !{!13, !164}
!171 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !172, line: 256)
!172 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !21, file: !21, line: 149, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!173 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !174, line: 257)
!174 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !21, file: !21, line: 151, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!175 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !176, line: 258)
!176 = !DISubprogram(name: "nexttoward", linkageName: "_ZL10nexttowardfd", scope: !21, file: !21, line: 153, type: !177, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!177 = !DISubroutineType(types: !178)
!178 = !{!13, !13, !163}
!179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !180, line: 259)
!180 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !21, file: !21, line: 158, type: !126, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !182, line: 260)
!182 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !21, file: !21, line: 160, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !184, line: 261)
!184 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !21, file: !21, line: 162, type: !185, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!185 = !DISubroutineType(types: !186)
!186 = !{!13, !13, !13, !89}
!187 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !188, line: 262)
!188 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !21, file: !21, line: 164, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !190, line: 263)
!190 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !21, file: !21, line: 166, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!191 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !192, line: 264)
!192 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !21, file: !21, line: 168, type: !193, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!193 = !DISubroutineType(types: !194)
!194 = !{!13, !13, !123}
!195 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !196, line: 265)
!196 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !21, file: !21, line: 170, type: !126, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!197 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !198, line: 266)
!198 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !21, file: !21, line: 172, type: !96, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !200, line: 267)
!200 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !21, file: !21, line: 174, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!201 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !202, line: 268)
!202 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !21, file: !21, line: 176, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!203 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !204, line: 269)
!204 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !21, file: !21, line: 178, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !206, line: 270)
!206 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !21, file: !21, line: 180, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!207 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !208, line: 271)
!208 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !21, file: !21, line: 182, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !210, line: 272)
!210 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !21, file: !21, line: 184, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!211 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !212, line: 273)
!212 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !21, file: !21, line: 186, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !214, line: 102)
!214 = !DISubprogram(name: "acos", scope: !215, file: !215, line: 54, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!215 = !DIFile(filename: "/usr/include/bits/mathcalls.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!216 = !DISubroutineType(types: !217)
!217 = !{!163, !163}
!218 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !219, line: 121)
!219 = !DISubprogram(name: "asin", scope: !215, file: !215, line: 56, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!220 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !221, line: 140)
!221 = !DISubprogram(name: "atan", scope: !215, file: !215, line: 58, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!222 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !223, line: 159)
!223 = !DISubprogram(name: "atan2", scope: !215, file: !215, line: 60, type: !224, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!224 = !DISubroutineType(types: !225)
!225 = !{!163, !163, !163}
!226 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !227, line: 180)
!227 = !DISubprogram(name: "ceil", scope: !215, file: !215, line: 179, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!228 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !229, line: 199)
!229 = !DISubprogram(name: "cos", scope: !215, file: !215, line: 63, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!230 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !231, line: 218)
!231 = !DISubprogram(name: "cosh", scope: !215, file: !215, line: 72, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!232 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !233, line: 237)
!233 = !DISubprogram(name: "exp", scope: !215, file: !215, line: 100, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!234 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !235, line: 256)
!235 = !DISubprogram(name: "fabs", scope: !215, file: !215, line: 182, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!236 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !237, line: 275)
!237 = !DISubprogram(name: "floor", scope: !215, file: !215, line: 185, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!238 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !239, line: 294)
!239 = !DISubprogram(name: "fmod", scope: !215, file: !215, line: 188, type: !224, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!240 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !241, line: 315)
!241 = !DISubprogram(name: "frexp", scope: !215, file: !215, line: 103, type: !242, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!242 = !DISubroutineType(types: !243)
!243 = !{!163, !163, !89}
!244 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !245, line: 334)
!245 = !DISubprogram(name: "ldexp", scope: !215, file: !215, line: 106, type: !246, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!246 = !DISubroutineType(types: !247)
!247 = !{!163, !163, !84}
!248 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !249, line: 353)
!249 = !DISubprogram(name: "log", scope: !215, file: !215, line: 109, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!250 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !251, line: 372)
!251 = !DISubprogram(name: "log10", scope: !215, file: !215, line: 112, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!252 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !253, line: 391)
!253 = !DISubprogram(name: "modf", scope: !215, file: !215, line: 115, type: !254, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!254 = !DISubroutineType(types: !255)
!255 = !{!163, !163, !256}
!256 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !163, size: 64)
!257 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !258, line: 403)
!258 = !DISubprogram(name: "pow", scope: !215, file: !215, line: 154, type: !224, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!259 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !260, line: 440)
!260 = !DISubprogram(name: "sin", scope: !215, file: !215, line: 65, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!261 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !262, line: 459)
!262 = !DISubprogram(name: "sinh", scope: !215, file: !215, line: 74, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!263 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !264, line: 478)
!264 = !DISubprogram(name: "sqrt", scope: !215, file: !215, line: 157, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!265 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !266, line: 497)
!266 = !DISubprogram(name: "tan", scope: !215, file: !215, line: 67, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!267 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !268, line: 516)
!268 = !DISubprogram(name: "tanh", scope: !215, file: !215, line: 76, type: !216, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!269 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !270, line: 118)
!270 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !271, line: 101, baseType: !272)
!271 = !DIFile(filename: "/usr/include/stdlib.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!272 = !DICompositeType(tag: DW_TAG_structure_type, file: !271, line: 97, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!273 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !274, line: 119)
!274 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !271, line: 109, baseType: !275)
!275 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !271, line: 105, size: 128, elements: !276, identifier: "_ZTS6ldiv_t")
!276 = !{!277, !278}
!277 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !275, file: !271, line: 107, baseType: !123, size: 64)
!278 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !275, file: !271, line: 108, baseType: !123, size: 64, offset: 64)
!279 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !280, line: 121)
!280 = !DISubprogram(name: "abort", scope: !271, file: !271, line: 514, type: !281, isLocal: false, isDefinition: false, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: false)
!281 = !DISubroutineType(types: !282)
!282 = !{null}
!283 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !284, line: 122)
!284 = !DISubprogram(name: "abs", scope: !271, file: !271, line: 770, type: !285, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!285 = !DISubroutineType(types: !286)
!286 = !{!84, !84}
!287 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !288, line: 123)
!288 = !DISubprogram(name: "atexit", scope: !271, file: !271, line: 518, type: !289, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!289 = !DISubroutineType(types: !290)
!290 = !{!84, !291}
!291 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !281, size: 64)
!292 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !293, line: 129)
!293 = !DISubprogram(name: "atof", scope: !271, file: !271, line: 144, type: !161, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!294 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !295, line: 130)
!295 = !DISubprogram(name: "atoi", scope: !271, file: !271, line: 147, type: !296, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!296 = !DISubroutineType(types: !297)
!297 = !{!84, !164}
!298 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !299, line: 131)
!299 = !DISubprogram(name: "atol", scope: !271, file: !271, line: 150, type: !300, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!300 = !DISubroutineType(types: !301)
!301 = !{!123, !164}
!302 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !303, line: 132)
!303 = !DISubprogram(name: "bsearch", scope: !271, file: !271, line: 754, type: !304, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!304 = !DISubroutineType(types: !305)
!305 = !{!15, !306, !306, !308, !308, !311}
!306 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !307, size: 64)
!307 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!308 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !309, line: 62, baseType: !310)
!309 = !DIFile(filename: "/home/hzhang86/packages/llvm/bin/../lib/clang/4.0.1/include/stddef.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!310 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!311 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !271, line: 741, baseType: !312)
!312 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !313, size: 64)
!313 = !DISubroutineType(types: !314)
!314 = !{!84, !306, !306}
!315 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !316, line: 133)
!316 = !DISubprogram(name: "calloc", scope: !271, file: !271, line: 467, type: !317, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!317 = !DISubroutineType(types: !318)
!318 = !{!15, !308, !308}
!319 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !320, line: 134)
!320 = !DISubprogram(name: "div", scope: !271, file: !271, line: 784, type: !321, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!321 = !DISubroutineType(types: !322)
!322 = !{!270, !84, !84}
!323 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !324, line: 135)
!324 = !DISubprogram(name: "exit", scope: !271, file: !271, line: 542, type: !325, isLocal: false, isDefinition: false, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: false)
!325 = !DISubroutineType(types: !326)
!326 = !{null, !84}
!327 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !328, line: 136)
!328 = !DISubprogram(name: "free", scope: !271, file: !271, line: 482, type: !329, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!329 = !DISubroutineType(types: !330)
!330 = !{null, !15}
!331 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !332, line: 137)
!332 = !DISubprogram(name: "getenv", scope: !271, file: !271, line: 563, type: !333, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!333 = !DISubroutineType(types: !334)
!334 = !{!335, !164}
!335 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !166, size: 64)
!336 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !337, line: 138)
!337 = !DISubprogram(name: "labs", scope: !271, file: !271, line: 771, type: !121, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!338 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !339, line: 139)
!339 = !DISubprogram(name: "ldiv", scope: !271, file: !271, line: 786, type: !340, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!340 = !DISubroutineType(types: !341)
!341 = !{!274, !123, !123}
!342 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !343, line: 140)
!343 = !DISubprogram(name: "malloc", scope: !271, file: !271, line: 465, type: !344, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!344 = !DISubroutineType(types: !345)
!345 = !{!15, !308}
!346 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !347, line: 142)
!347 = !DISubprogram(name: "mblen", scope: !271, file: !271, line: 859, type: !348, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!348 = !DISubroutineType(types: !349)
!349 = !{!84, !164, !308}
!350 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !351, line: 143)
!351 = !DISubprogram(name: "mbstowcs", scope: !271, file: !271, line: 870, type: !352, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!352 = !DISubroutineType(types: !353)
!353 = !{!308, !354, !357, !308}
!354 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !355)
!355 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !356, size: 64)
!356 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!357 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !164)
!358 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !359, line: 144)
!359 = !DISubprogram(name: "mbtowc", scope: !271, file: !271, line: 862, type: !360, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!360 = !DISubroutineType(types: !361)
!361 = !{!84, !354, !357, !308}
!362 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !363, line: 146)
!363 = !DISubprogram(name: "qsort", scope: !271, file: !271, line: 760, type: !364, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!364 = !DISubroutineType(types: !365)
!365 = !{null, !15, !308, !308, !311}
!366 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !367, line: 152)
!367 = !DISubprogram(name: "rand", scope: !271, file: !271, line: 374, type: !368, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!368 = !DISubroutineType(types: !369)
!369 = !{!84}
!370 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !371, line: 153)
!371 = !DISubprogram(name: "realloc", scope: !271, file: !271, line: 479, type: !372, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!372 = !DISubroutineType(types: !373)
!373 = !{!15, !15, !308}
!374 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !375, line: 154)
!375 = !DISubprogram(name: "srand", scope: !271, file: !271, line: 376, type: !376, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!376 = !DISubroutineType(types: !377)
!377 = !{null, !16}
!378 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !379, line: 155)
!379 = !DISubprogram(name: "strtod", scope: !271, file: !271, line: 164, type: !380, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!380 = !DISubroutineType(types: !381)
!381 = !{!163, !357, !382}
!382 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !383)
!383 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !335, size: 64)
!384 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !385, line: 156)
!385 = !DISubprogram(name: "strtol", scope: !271, file: !271, line: 183, type: !386, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!386 = !DISubroutineType(types: !387)
!387 = !{!123, !357, !382, !84}
!388 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !389, line: 157)
!389 = !DISubprogram(name: "strtoul", scope: !271, file: !271, line: 187, type: !390, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!390 = !DISubroutineType(types: !391)
!391 = !{!310, !357, !382, !84}
!392 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !393, line: 158)
!393 = !DISubprogram(name: "system", scope: !271, file: !271, line: 716, type: !296, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!394 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !395, line: 160)
!395 = !DISubprogram(name: "wcstombs", scope: !271, file: !271, line: 873, type: !396, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!396 = !DISubroutineType(types: !397)
!397 = !{!308, !398, !399, !308}
!398 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !335)
!399 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !400)
!400 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !401, size: 64)
!401 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !356)
!402 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !403, line: 161)
!403 = !DISubprogram(name: "wctomb", scope: !271, file: !271, line: 866, type: !404, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!404 = !DISubroutineType(types: !405)
!405 = !{!84, !335, !356}
!406 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !407, entity: !409, line: 201)
!407 = !DINamespace(name: "__gnu_cxx", scope: null, file: !408, line: 68)
!408 = !DIFile(filename: "/usr/lib/gcc/x86_64-redhat-linux/4.8.5/../../../../include/c++/4.8.5/bits/cpp_type_traits.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!409 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !271, line: 121, baseType: !410)
!410 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !271, line: 117, size: 128, elements: !411, identifier: "_ZTS7lldiv_t")
!411 = !{!412, !413}
!412 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !410, file: !271, line: 119, baseType: !25, size: 64)
!413 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !410, file: !271, line: 120, baseType: !25, size: 64, offset: 64)
!414 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !407, entity: !415, line: 207)
!415 = !DISubprogram(name: "_Exit", scope: !271, file: !271, line: 556, type: !325, isLocal: false, isDefinition: false, flags: DIFlagPrototyped | DIFlagNoReturn, isOptimized: false)
!416 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !407, entity: !417, line: 211)
!417 = !DISubprogram(name: "llabs", scope: !271, file: !271, line: 775, type: !23, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!418 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !407, entity: !419, line: 217)
!419 = !DISubprogram(name: "lldiv", scope: !271, file: !271, line: 792, type: !420, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!420 = !DISubroutineType(types: !421)
!421 = !{!409, !25, !25}
!422 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !407, entity: !423, line: 228)
!423 = !DISubprogram(name: "atoll", scope: !271, file: !271, line: 157, type: !424, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!424 = !DISubroutineType(types: !425)
!425 = !{!25, !164}
!426 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !407, entity: !427, line: 229)
!427 = !DISubprogram(name: "strtoll", scope: !271, file: !271, line: 209, type: !428, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!428 = !DISubroutineType(types: !429)
!429 = !{!25, !357, !382, !84}
!430 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !407, entity: !431, line: 230)
!431 = !DISubprogram(name: "strtoull", scope: !271, file: !271, line: 214, type: !432, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!432 = !DISubroutineType(types: !433)
!433 = !{!434, !357, !382, !84}
!434 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!435 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !407, entity: !436, line: 232)
!436 = !DISubprogram(name: "strtof", scope: !271, file: !271, line: 172, type: !437, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!437 = !DISubroutineType(types: !438)
!438 = !{!13, !357, !382}
!439 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !407, entity: !440, line: 233)
!440 = !DISubprogram(name: "strtold", scope: !271, file: !271, line: 175, type: !441, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!441 = !DISubroutineType(types: !442)
!442 = !{!443, !357, !382}
!443 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!444 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !409, line: 241)
!445 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !415, line: 243)
!446 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !417, line: 245)
!447 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !448, line: 246)
!448 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !407, file: !449, line: 214, type: !420, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!449 = !DIFile(filename: "/usr/lib/gcc/x86_64-redhat-linux/4.8.5/../../../../include/c++/4.8.5/cstdlib", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!450 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !419, line: 247)
!451 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !423, line: 249)
!452 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !436, line: 250)
!453 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !427, line: 251)
!454 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !431, line: 252)
!455 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !440, line: 253)
!456 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !457, line: 418)
!457 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !458, file: !458, line: 1342, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!458 = !DIFile(filename: "/software/cuda/8.0.44/rel70/x86_64/include/math_functions.hpp", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!459 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !460, line: 419)
!460 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !458, file: !458, line: 1370, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!461 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !462, line: 420)
!462 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !458, file: !458, line: 1337, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!463 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !464, line: 421)
!464 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !458, file: !458, line: 1375, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!465 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !466, line: 422)
!466 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !458, file: !458, line: 1327, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!467 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !468, line: 423)
!468 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !458, file: !458, line: 1332, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!469 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !470, line: 424)
!470 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !458, file: !458, line: 1380, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!471 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !472, line: 425)
!472 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !458, file: !458, line: 1430, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!473 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !474, line: 426)
!474 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !475, file: !475, line: 667, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!475 = !DIFile(filename: "/software/cuda/8.0.44/rel70/x86_64/include/device_functions.hpp", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!476 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !477, line: 427)
!477 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !458, file: !458, line: 1189, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!478 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !479, line: 428)
!479 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !458, file: !458, line: 1243, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!480 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !481, line: 429)
!481 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !458, file: !458, line: 1312, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!482 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !483, line: 430)
!483 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !458, file: !458, line: 1490, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!484 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !485, line: 431)
!485 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !458, file: !458, line: 1480, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!486 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !487, line: 432)
!487 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !475, file: !475, line: 657, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!488 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !489, line: 433)
!489 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !458, file: !458, line: 1294, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!490 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !491, line: 434)
!491 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !458, file: !458, line: 1385, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!492 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !493, line: 435)
!493 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !475, file: !475, line: 607, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!494 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !495, line: 436)
!495 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !458, file: !458, line: 1616, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!496 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !497, line: 437)
!497 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !475, file: !475, line: 597, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!498 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !499, line: 438)
!499 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !458, file: !458, line: 1568, type: !72, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!500 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !501, line: 439)
!501 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !475, file: !475, line: 622, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!502 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !503, line: 440)
!503 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !475, file: !475, line: 617, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!504 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !505, line: 441)
!505 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !458, file: !458, line: 1553, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!506 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !507, line: 442)
!507 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !458, file: !458, line: 1543, type: !87, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!508 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !509, line: 443)
!509 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !458, file: !458, line: 1390, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!510 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !511, line: 444)
!511 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !458, file: !458, line: 1621, type: !82, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!512 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !513, line: 445)
!513 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !458, file: !458, line: 1520, type: !126, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!514 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !515, line: 446)
!515 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !458, file: !458, line: 1515, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!516 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !517, line: 447)
!517 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !458, file: !458, line: 1149, type: !134, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!518 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !519, line: 448)
!519 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !458, file: !458, line: 1602, type: !134, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!520 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !521, line: 449)
!521 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !458, file: !458, line: 1356, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!522 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !523, line: 450)
!523 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !458, file: !458, line: 1365, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!524 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !525, line: 451)
!525 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !458, file: !458, line: 1285, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!526 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !527, line: 452)
!527 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !458, file: !458, line: 1626, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!528 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !529, line: 453)
!529 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !458, file: !458, line: 1347, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!530 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !531, line: 454)
!531 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !458, file: !458, line: 1140, type: !148, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!532 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !533, line: 455)
!533 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !458, file: !458, line: 1607, type: !148, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!534 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !535, line: 456)
!535 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !458, file: !458, line: 1548, type: !156, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!536 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !537, line: 457)
!537 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !458, file: !458, line: 1154, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!538 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !539, line: 458)
!539 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !458, file: !458, line: 1218, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!540 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !541, line: 459)
!541 = !DISubprogram(name: "nexttowardf", scope: !215, file: !215, line: 285, type: !542, isLocal: false, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!542 = !DISubroutineType(types: !543)
!543 = !{!13, !13, !443}
!544 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !541, line: 460)
!545 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !546, line: 461)
!546 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !458, file: !458, line: 1583, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!547 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !548, line: 462)
!548 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !458, file: !458, line: 1558, type: !40, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!549 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !550, line: 463)
!550 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !458, file: !458, line: 1563, type: !185, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!551 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !552, line: 464)
!552 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !458, file: !458, line: 1135, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!553 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !554, line: 465)
!554 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !458, file: !458, line: 1597, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!555 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !556, line: 466)
!556 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !458, file: !458, line: 1530, type: !193, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!557 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !558, line: 467)
!558 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !458, file: !458, line: 1525, type: !126, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!559 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !560, line: 468)
!560 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !458, file: !458, line: 1234, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!561 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !562, line: 469)
!562 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !458, file: !458, line: 1317, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!563 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !564, line: 470)
!564 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !475, file: !475, line: 907, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!565 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !566, line: 471)
!566 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !458, file: !458, line: 1276, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!567 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !568, line: 472)
!568 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !458, file: !458, line: 1322, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!569 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !570, line: 473)
!570 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !458, file: !458, line: 1592, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!571 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !20, entity: !572, line: 474)
!572 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !475, file: !475, line: 662, type: !28, isLocal: true, isDefinition: false, flags: DIFlagPrototyped, isOptimized: false)
!573 = !{i32 2, !"Dwarf Version", i32 4}
!574 = !{i32 2, !"Debug Info Version", i32 3}
!575 = !{!"clang version 4.0.1 (tags/RELEASE_401/final)"}
!576 = distinct !DISubprogram(name: "rtclock", linkageName: "_Z7rtclockv", scope: !577, file: !577, line: 11, type: !578, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!577 = !DIFile(filename: "./../../common/polybenchUtilFuncts.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!578 = !DISubroutineType(types: !579)
!579 = !{!163}
!580 = !{}
!581 = !DILocalVariable(name: "Tzp", scope: !576, file: !577, line: 13, type: !582)
!582 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "timezone", file: !583, line: 56, size: 64, elements: !584, identifier: "_ZTS8timezone")
!583 = !DIFile(filename: "/usr/include/sys/time.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!584 = !{!585, !586}
!585 = !DIDerivedType(tag: DW_TAG_member, name: "tz_minuteswest", scope: !582, file: !583, line: 58, baseType: !84, size: 32)
!586 = !DIDerivedType(tag: DW_TAG_member, name: "tz_dsttime", scope: !582, file: !583, line: 59, baseType: !84, size: 32, offset: 32)
!587 = !DIExpression()
!588 = !DILocation(line: 13, column: 21, scope: !576)
!589 = !DILocalVariable(name: "Tp", scope: !576, file: !577, line: 14, type: !590)
!590 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "timeval", file: !591, line: 30, size: 128, elements: !592, identifier: "_ZTS7timeval")
!591 = !DIFile(filename: "/usr/include/bits/time.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!592 = !{!593, !596}
!593 = !DIDerivedType(tag: DW_TAG_member, name: "tv_sec", scope: !590, file: !591, line: 32, baseType: !594, size: 64)
!594 = !DIDerivedType(tag: DW_TAG_typedef, name: "__time_t", file: !595, line: 148, baseType: !123)
!595 = !DIFile(filename: "/usr/include/bits/types.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!596 = !DIDerivedType(tag: DW_TAG_member, name: "tv_usec", scope: !590, file: !591, line: 33, baseType: !597, size: 64, offset: 64)
!597 = !DIDerivedType(tag: DW_TAG_typedef, name: "__suseconds_t", file: !595, line: 150, baseType: !123)
!598 = !DILocation(line: 14, column: 20, scope: !576)
!599 = !DILocalVariable(name: "stat", scope: !576, file: !577, line: 15, type: !84)
!600 = !DILocation(line: 15, column: 9, scope: !576)
!601 = !DILocation(line: 16, column: 12, scope: !576)
!602 = !DILocation(line: 16, column: 10, scope: !576)
!603 = !DILocation(line: 17, column: 9, scope: !604)
!604 = distinct !DILexicalBlock(scope: !576, file: !577, line: 17, column: 9)
!605 = !DILocation(line: 17, column: 14, scope: !604)
!606 = !DILocation(line: 17, column: 9, scope: !576)
!607 = !DILocation(line: 17, column: 64, scope: !608)
!608 = !DILexicalBlockFile(scope: !604, file: !577, discriminator: 1)
!609 = !DILocation(line: 17, column: 20, scope: !608)
!610 = !DILocation(line: 18, column: 15, scope: !576)
!611 = !DILocation(line: 18, column: 12, scope: !576)
!612 = !DILocation(line: 18, column: 27, scope: !576)
!613 = !DILocation(line: 18, column: 24, scope: !576)
!614 = !DILocation(line: 18, column: 34, scope: !576)
!615 = !DILocation(line: 18, column: 22, scope: !576)
!616 = !DILocation(line: 18, column: 5, scope: !576)
!617 = distinct !DISubprogram(name: "absVal", linkageName: "_Z6absValf", scope: !577, file: !577, line: 22, type: !28, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!618 = !DILocalVariable(name: "a", arg: 1, scope: !617, file: !577, line: 22, type: !13)
!619 = !DILocation(line: 22, column: 20, scope: !617)
!620 = !DILocation(line: 24, column: 5, scope: !621)
!621 = distinct !DILexicalBlock(scope: !617, file: !577, line: 24, column: 5)
!622 = !DILocation(line: 24, column: 7, scope: !621)
!623 = !DILocation(line: 24, column: 5, scope: !617)
!624 = !DILocation(line: 26, column: 11, scope: !625)
!625 = distinct !DILexicalBlock(scope: !621, file: !577, line: 25, column: 2)
!626 = !DILocation(line: 26, column: 13, scope: !625)
!627 = !DILocation(line: 26, column: 3, scope: !625)
!628 = !DILocation(line: 30, column: 10, scope: !629)
!629 = distinct !DILexicalBlock(scope: !621, file: !577, line: 29, column: 2)
!630 = !DILocation(line: 30, column: 3, scope: !629)
!631 = !DILocation(line: 32, column: 1, scope: !617)
!632 = distinct !DISubprogram(name: "percentDiff", linkageName: "_Z11percentDiffdd", scope: !577, file: !577, line: 36, type: !633, isLocal: false, isDefinition: true, scopeLine: 37, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!633 = !DISubroutineType(types: !634)
!634 = !{!13, !163, !163}
!635 = !DILocalVariable(name: "val1", arg: 1, scope: !632, file: !577, line: 36, type: !163)
!636 = !DILocation(line: 36, column: 26, scope: !632)
!637 = !DILocalVariable(name: "val2", arg: 2, scope: !632, file: !577, line: 36, type: !163)
!638 = !DILocation(line: 36, column: 39, scope: !632)
!639 = !DILocation(line: 38, column: 14, scope: !640)
!640 = distinct !DILexicalBlock(scope: !632, file: !577, line: 38, column: 6)
!641 = !DILocation(line: 38, column: 7, scope: !640)
!642 = !DILocation(line: 38, column: 20, scope: !640)
!643 = !DILocation(line: 38, column: 28, scope: !640)
!644 = !DILocation(line: 38, column: 39, scope: !645)
!645 = !DILexicalBlockFile(scope: !640, file: !577, discriminator: 1)
!646 = !DILocation(line: 38, column: 32, scope: !645)
!647 = !DILocation(line: 38, column: 45, scope: !645)
!648 = !DILocation(line: 38, column: 6, scope: !649)
!649 = !DILexicalBlockFile(scope: !632, file: !577, discriminator: 1)
!650 = !DILocation(line: 40, column: 3, scope: !651)
!651 = distinct !DILexicalBlock(scope: !640, file: !577, line: 39, column: 2)
!652 = !DILocation(line: 45, column: 38, scope: !653)
!653 = distinct !DILexicalBlock(scope: !640, file: !577, line: 44, column: 2)
!654 = !DILocation(line: 45, column: 45, scope: !653)
!655 = !DILocation(line: 45, column: 43, scope: !653)
!656 = !DILocation(line: 45, column: 31, scope: !653)
!657 = !DILocation(line: 45, column: 60, scope: !653)
!658 = !DILocation(line: 45, column: 65, scope: !653)
!659 = !DILocation(line: 45, column: 53, scope: !660)
!660 = !DILexicalBlockFile(scope: !653, file: !577, discriminator: 1)
!661 = !DILocation(line: 45, column: 51, scope: !653)
!662 = !DILocation(line: 45, column: 24, scope: !663)
!663 = !DILexicalBlockFile(scope: !653, file: !577, discriminator: 2)
!664 = !DILocation(line: 45, column: 21, scope: !653)
!665 = !DILocation(line: 45, column: 7, scope: !653)
!666 = !DILocation(line: 47, column: 1, scope: !632)
!667 = distinct !DISubprogram(name: "gesummv", linkageName: "_Z7gesummvPfS_S_S_S_", scope: !1, file: !1, line: 43, type: !668, isLocal: false, isDefinition: true, scopeLine: 44, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!668 = !DISubroutineType(types: !669)
!669 = !{null, !17, !17, !17, !17, !17}
!670 = !DILocalVariable(name: "A", arg: 1, scope: !667, file: !1, line: 43, type: !17)
!671 = !DILocation(line: 43, column: 25, scope: !667)
!672 = !DILocalVariable(name: "B", arg: 2, scope: !667, file: !1, line: 43, type: !17)
!673 = !DILocation(line: 43, column: 39, scope: !667)
!674 = !DILocalVariable(name: "x", arg: 3, scope: !667, file: !1, line: 43, type: !17)
!675 = !DILocation(line: 43, column: 53, scope: !667)
!676 = !DILocalVariable(name: "y", arg: 4, scope: !667, file: !1, line: 43, type: !17)
!677 = !DILocation(line: 43, column: 67, scope: !667)
!678 = !DILocalVariable(name: "tmp", arg: 5, scope: !667, file: !1, line: 43, type: !17)
!679 = !DILocation(line: 43, column: 81, scope: !667)
!680 = !DILocalVariable(name: "i", scope: !667, file: !1, line: 45, type: !84)
!681 = !DILocation(line: 45, column: 6, scope: !667)
!682 = !DILocalVariable(name: "j", scope: !667, file: !1, line: 45, type: !84)
!683 = !DILocation(line: 45, column: 9, scope: !667)
!684 = !DILocation(line: 47, column: 9, scope: !685)
!685 = distinct !DILexicalBlock(scope: !667, file: !1, line: 47, column: 2)
!686 = !DILocation(line: 47, column: 7, scope: !685)
!687 = !DILocation(line: 47, column: 14, scope: !688)
!688 = !DILexicalBlockFile(scope: !689, file: !1, discriminator: 1)
!689 = distinct !DILexicalBlock(scope: !685, file: !1, line: 47, column: 2)
!690 = !DILocation(line: 47, column: 16, scope: !688)
!691 = !DILocation(line: 47, column: 2, scope: !692)
!692 = !DILexicalBlockFile(scope: !685, file: !1, discriminator: 1)
!693 = !DILocation(line: 49, column: 3, scope: !694)
!694 = distinct !DILexicalBlock(scope: !689, file: !1, line: 48, column: 2)
!695 = !DILocation(line: 49, column: 7, scope: !694)
!696 = !DILocation(line: 49, column: 10, scope: !694)
!697 = !DILocation(line: 50, column: 3, scope: !694)
!698 = !DILocation(line: 50, column: 5, scope: !694)
!699 = !DILocation(line: 50, column: 8, scope: !694)
!700 = !DILocation(line: 51, column: 10, scope: !701)
!701 = distinct !DILexicalBlock(scope: !694, file: !1, line: 51, column: 3)
!702 = !DILocation(line: 51, column: 8, scope: !701)
!703 = !DILocation(line: 51, column: 15, scope: !704)
!704 = !DILexicalBlockFile(scope: !705, file: !1, discriminator: 1)
!705 = distinct !DILexicalBlock(scope: !701, file: !1, line: 51, column: 3)
!706 = !DILocation(line: 51, column: 17, scope: !704)
!707 = !DILocation(line: 51, column: 3, scope: !708)
!708 = !DILexicalBlockFile(scope: !701, file: !1, discriminator: 1)
!709 = !DILocation(line: 53, column: 13, scope: !710)
!710 = distinct !DILexicalBlock(scope: !705, file: !1, line: 52, column: 3)
!711 = !DILocation(line: 53, column: 15, scope: !710)
!712 = !DILocation(line: 53, column: 16, scope: !710)
!713 = !DILocation(line: 53, column: 21, scope: !710)
!714 = !DILocation(line: 53, column: 19, scope: !710)
!715 = !DILocation(line: 53, column: 26, scope: !710)
!716 = !DILocation(line: 53, column: 28, scope: !710)
!717 = !DILocation(line: 53, column: 24, scope: !710)
!718 = !DILocation(line: 53, column: 33, scope: !710)
!719 = !DILocation(line: 53, column: 37, scope: !710)
!720 = !DILocation(line: 53, column: 31, scope: !710)
!721 = !DILocation(line: 53, column: 4, scope: !710)
!722 = !DILocation(line: 53, column: 8, scope: !710)
!723 = !DILocation(line: 53, column: 11, scope: !710)
!724 = !DILocation(line: 54, column: 11, scope: !710)
!725 = !DILocation(line: 54, column: 13, scope: !710)
!726 = !DILocation(line: 54, column: 14, scope: !710)
!727 = !DILocation(line: 54, column: 19, scope: !710)
!728 = !DILocation(line: 54, column: 17, scope: !710)
!729 = !DILocation(line: 54, column: 24, scope: !710)
!730 = !DILocation(line: 54, column: 26, scope: !710)
!731 = !DILocation(line: 54, column: 22, scope: !710)
!732 = !DILocation(line: 54, column: 31, scope: !710)
!733 = !DILocation(line: 54, column: 33, scope: !710)
!734 = !DILocation(line: 54, column: 29, scope: !710)
!735 = !DILocation(line: 54, column: 4, scope: !710)
!736 = !DILocation(line: 54, column: 6, scope: !710)
!737 = !DILocation(line: 54, column: 9, scope: !710)
!738 = !DILocation(line: 55, column: 3, scope: !710)
!739 = !DILocation(line: 51, column: 23, scope: !740)
!740 = !DILexicalBlockFile(scope: !705, file: !1, discriminator: 2)
!741 = !DILocation(line: 51, column: 3, scope: !740)
!742 = distinct !{!742, !743, !744}
!743 = !DILocation(line: 51, column: 3, scope: !701)
!744 = !DILocation(line: 55, column: 3, scope: !701)
!745 = !DILocation(line: 57, column: 18, scope: !694)
!746 = !DILocation(line: 57, column: 22, scope: !694)
!747 = !DILocation(line: 57, column: 16, scope: !694)
!748 = !DILocation(line: 57, column: 34, scope: !694)
!749 = !DILocation(line: 57, column: 36, scope: !694)
!750 = !DILocation(line: 57, column: 32, scope: !694)
!751 = !DILocation(line: 57, column: 25, scope: !694)
!752 = !DILocation(line: 57, column: 3, scope: !694)
!753 = !DILocation(line: 57, column: 5, scope: !694)
!754 = !DILocation(line: 57, column: 8, scope: !694)
!755 = !DILocation(line: 58, column: 2, scope: !694)
!756 = !DILocation(line: 47, column: 22, scope: !757)
!757 = !DILexicalBlockFile(scope: !689, file: !1, discriminator: 2)
!758 = !DILocation(line: 47, column: 2, scope: !757)
!759 = distinct !{!759, !760, !761}
!760 = !DILocation(line: 47, column: 2, scope: !685)
!761 = !DILocation(line: 58, column: 2, scope: !685)
!762 = !DILocation(line: 59, column: 1, scope: !667)
!763 = distinct !DISubprogram(name: "init", linkageName: "_Z4initPfS_", scope: !1, file: !1, line: 62, type: !764, isLocal: false, isDefinition: true, scopeLine: 63, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!764 = !DISubroutineType(types: !765)
!765 = !{null, !17, !17}
!766 = !DILocalVariable(name: "A", arg: 1, scope: !763, file: !1, line: 62, type: !17)
!767 = !DILocation(line: 62, column: 22, scope: !763)
!768 = !DILocalVariable(name: "x", arg: 2, scope: !763, file: !1, line: 62, type: !17)
!769 = !DILocation(line: 62, column: 36, scope: !763)
!770 = !DILocalVariable(name: "i", scope: !763, file: !1, line: 64, type: !84)
!771 = !DILocation(line: 64, column: 8, scope: !763)
!772 = !DILocalVariable(name: "j", scope: !763, file: !1, line: 64, type: !84)
!773 = !DILocation(line: 64, column: 11, scope: !763)
!774 = !DILocation(line: 66, column: 10, scope: !775)
!775 = distinct !DILexicalBlock(scope: !763, file: !1, line: 66, column: 3)
!776 = !DILocation(line: 66, column: 8, scope: !775)
!777 = !DILocation(line: 66, column: 15, scope: !778)
!778 = !DILexicalBlockFile(scope: !779, file: !1, discriminator: 1)
!779 = distinct !DILexicalBlock(scope: !775, file: !1, line: 66, column: 3)
!780 = !DILocation(line: 66, column: 17, scope: !778)
!781 = !DILocation(line: 66, column: 3, scope: !782)
!782 = !DILexicalBlockFile(scope: !775, file: !1, discriminator: 1)
!783 = !DILocation(line: 68, column: 26, scope: !784)
!784 = distinct !DILexicalBlock(scope: !779, file: !1, line: 67, column: 5)
!785 = !DILocation(line: 68, column: 29, scope: !784)
!786 = !DILocation(line: 68, column: 6, scope: !784)
!787 = !DILocation(line: 68, column: 8, scope: !784)
!788 = !DILocation(line: 68, column: 11, scope: !784)
!789 = !DILocation(line: 70, column: 10, scope: !790)
!790 = distinct !DILexicalBlock(scope: !784, file: !1, line: 70, column: 3)
!791 = !DILocation(line: 70, column: 8, scope: !790)
!792 = !DILocation(line: 70, column: 15, scope: !793)
!793 = !DILexicalBlockFile(scope: !794, file: !1, discriminator: 1)
!794 = distinct !DILexicalBlock(scope: !790, file: !1, line: 70, column: 3)
!795 = !DILocation(line: 70, column: 17, scope: !793)
!796 = !DILocation(line: 70, column: 3, scope: !797)
!797 = !DILexicalBlockFile(scope: !790, file: !1, discriminator: 1)
!798 = !DILocation(line: 72, column: 30, scope: !799)
!799 = distinct !DILexicalBlock(scope: !794, file: !1, line: 71, column: 3)
!800 = !DILocation(line: 72, column: 32, scope: !799)
!801 = !DILocation(line: 72, column: 31, scope: !799)
!802 = !DILocation(line: 72, column: 35, scope: !799)
!803 = !DILocation(line: 72, column: 4, scope: !799)
!804 = !DILocation(line: 72, column: 6, scope: !799)
!805 = !DILocation(line: 72, column: 7, scope: !799)
!806 = !DILocation(line: 72, column: 12, scope: !799)
!807 = !DILocation(line: 72, column: 10, scope: !799)
!808 = !DILocation(line: 72, column: 15, scope: !799)
!809 = !DILocation(line: 73, column: 3, scope: !799)
!810 = !DILocation(line: 70, column: 23, scope: !811)
!811 = !DILexicalBlockFile(scope: !794, file: !1, discriminator: 2)
!812 = !DILocation(line: 70, column: 3, scope: !811)
!813 = distinct !{!813, !814, !815}
!814 = !DILocation(line: 70, column: 3, scope: !790)
!815 = !DILocation(line: 73, column: 3, scope: !790)
!816 = !DILocation(line: 74, column: 5, scope: !784)
!817 = !DILocation(line: 66, column: 23, scope: !818)
!818 = !DILexicalBlockFile(scope: !779, file: !1, discriminator: 2)
!819 = !DILocation(line: 66, column: 3, scope: !818)
!820 = distinct !{!820, !821, !822}
!821 = !DILocation(line: 66, column: 3, scope: !775)
!822 = !DILocation(line: 74, column: 5, scope: !775)
!823 = !DILocation(line: 75, column: 1, scope: !763)
!824 = distinct !DISubprogram(name: "compareResults", linkageName: "_Z14compareResultsPfS_", scope: !1, file: !1, line: 78, type: !764, isLocal: false, isDefinition: true, scopeLine: 79, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!825 = !DILocalVariable(name: "y", arg: 1, scope: !824, file: !1, line: 78, type: !17)
!826 = !DILocation(line: 78, column: 32, scope: !824)
!827 = !DILocalVariable(name: "y_outputFromGpu", arg: 2, scope: !824, file: !1, line: 78, type: !17)
!828 = !DILocation(line: 78, column: 46, scope: !824)
!829 = !DILocalVariable(name: "i", scope: !824, file: !1, line: 80, type: !84)
!830 = !DILocation(line: 80, column: 6, scope: !824)
!831 = !DILocalVariable(name: "fail", scope: !824, file: !1, line: 80, type: !84)
!832 = !DILocation(line: 80, column: 9, scope: !824)
!833 = !DILocation(line: 81, column: 7, scope: !824)
!834 = !DILocation(line: 83, column: 8, scope: !835)
!835 = distinct !DILexicalBlock(scope: !824, file: !1, line: 83, column: 2)
!836 = !DILocation(line: 83, column: 7, scope: !835)
!837 = !DILocation(line: 83, column: 12, scope: !838)
!838 = !DILexicalBlockFile(scope: !839, file: !1, discriminator: 1)
!839 = distinct !DILexicalBlock(scope: !835, file: !1, line: 83, column: 2)
!840 = !DILocation(line: 83, column: 13, scope: !838)
!841 = !DILocation(line: 83, column: 2, scope: !842)
!842 = !DILexicalBlockFile(scope: !835, file: !1, discriminator: 1)
!843 = !DILocation(line: 85, column: 19, scope: !844)
!844 = distinct !DILexicalBlock(scope: !845, file: !1, line: 85, column: 7)
!845 = distinct !DILexicalBlock(scope: !839, file: !1, line: 84, column: 2)
!846 = !DILocation(line: 85, column: 21, scope: !844)
!847 = !DILocation(line: 85, column: 25, scope: !844)
!848 = !DILocation(line: 85, column: 41, scope: !844)
!849 = !DILocation(line: 85, column: 7, scope: !844)
!850 = !DILocation(line: 85, column: 45, scope: !844)
!851 = !DILocation(line: 85, column: 7, scope: !845)
!852 = !DILocation(line: 87, column: 8, scope: !853)
!853 = distinct !DILexicalBlock(scope: !844, file: !1, line: 86, column: 3)
!854 = !DILocation(line: 88, column: 3, scope: !853)
!855 = !DILocation(line: 89, column: 2, scope: !845)
!856 = !DILocation(line: 83, column: 20, scope: !857)
!857 = !DILexicalBlockFile(scope: !839, file: !1, discriminator: 2)
!858 = !DILocation(line: 83, column: 2, scope: !857)
!859 = distinct !{!859, !860, !861}
!860 = !DILocation(line: 83, column: 2, scope: !835)
!861 = !DILocation(line: 89, column: 2, scope: !835)
!862 = !DILocation(line: 92, column: 117, scope: !824)
!863 = !DILocation(line: 92, column: 2, scope: !824)
!864 = !DILocation(line: 93, column: 1, scope: !824)
!865 = distinct !DISubprogram(name: "GPU_argv_init", linkageName: "_Z13GPU_argv_initv", scope: !1, file: !1, line: 96, type: !281, isLocal: false, isDefinition: true, scopeLine: 97, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!866 = !DILocalVariable(name: "deviceProp", scope: !865, file: !1, line: 98, type: !867)
!867 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "cudaDeviceProp", file: !4, line: 1307, size: 5184, elements: !868, identifier: "_ZTS14cudaDeviceProp")
!868 = !{!869, !873, !874, !875, !876, !877, !878, !879, !883, !884, !885, !886, !887, !888, !889, !890, !891, !892, !893, !894, !895, !896, !897, !898, !899, !903, !904, !905, !906, !907, !908, !909, !910, !911, !912, !913, !914, !915, !916, !917, !918, !919, !920, !921, !922, !923, !924, !925, !926, !927, !928, !929, !930, !931, !932, !933, !934, !935, !936, !937, !938, !939, !940, !941, !942, !943}
!869 = !DIDerivedType(tag: DW_TAG_member, name: "name", scope: !867, file: !4, line: 1309, baseType: !870, size: 2048)
!870 = !DICompositeType(tag: DW_TAG_array_type, baseType: !166, size: 2048, elements: !871)
!871 = !{!872}
!872 = !DISubrange(count: 256)
!873 = !DIDerivedType(tag: DW_TAG_member, name: "totalGlobalMem", scope: !867, file: !4, line: 1310, baseType: !308, size: 64, offset: 2048)
!874 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerBlock", scope: !867, file: !4, line: 1311, baseType: !308, size: 64, offset: 2112)
!875 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerBlock", scope: !867, file: !4, line: 1312, baseType: !84, size: 32, offset: 2176)
!876 = !DIDerivedType(tag: DW_TAG_member, name: "warpSize", scope: !867, file: !4, line: 1313, baseType: !84, size: 32, offset: 2208)
!877 = !DIDerivedType(tag: DW_TAG_member, name: "memPitch", scope: !867, file: !4, line: 1314, baseType: !308, size: 64, offset: 2240)
!878 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerBlock", scope: !867, file: !4, line: 1315, baseType: !84, size: 32, offset: 2304)
!879 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsDim", scope: !867, file: !4, line: 1316, baseType: !880, size: 96, offset: 2336)
!880 = !DICompositeType(tag: DW_TAG_array_type, baseType: !84, size: 96, elements: !881)
!881 = !{!882}
!882 = !DISubrange(count: 3)
!883 = !DIDerivedType(tag: DW_TAG_member, name: "maxGridSize", scope: !867, file: !4, line: 1317, baseType: !880, size: 96, offset: 2432)
!884 = !DIDerivedType(tag: DW_TAG_member, name: "clockRate", scope: !867, file: !4, line: 1318, baseType: !84, size: 32, offset: 2528)
!885 = !DIDerivedType(tag: DW_TAG_member, name: "totalConstMem", scope: !867, file: !4, line: 1319, baseType: !308, size: 64, offset: 2560)
!886 = !DIDerivedType(tag: DW_TAG_member, name: "major", scope: !867, file: !4, line: 1320, baseType: !84, size: 32, offset: 2624)
!887 = !DIDerivedType(tag: DW_TAG_member, name: "minor", scope: !867, file: !4, line: 1321, baseType: !84, size: 32, offset: 2656)
!888 = !DIDerivedType(tag: DW_TAG_member, name: "textureAlignment", scope: !867, file: !4, line: 1322, baseType: !308, size: 64, offset: 2688)
!889 = !DIDerivedType(tag: DW_TAG_member, name: "texturePitchAlignment", scope: !867, file: !4, line: 1323, baseType: !308, size: 64, offset: 2752)
!890 = !DIDerivedType(tag: DW_TAG_member, name: "deviceOverlap", scope: !867, file: !4, line: 1324, baseType: !84, size: 32, offset: 2816)
!891 = !DIDerivedType(tag: DW_TAG_member, name: "multiProcessorCount", scope: !867, file: !4, line: 1325, baseType: !84, size: 32, offset: 2848)
!892 = !DIDerivedType(tag: DW_TAG_member, name: "kernelExecTimeoutEnabled", scope: !867, file: !4, line: 1326, baseType: !84, size: 32, offset: 2880)
!893 = !DIDerivedType(tag: DW_TAG_member, name: "integrated", scope: !867, file: !4, line: 1327, baseType: !84, size: 32, offset: 2912)
!894 = !DIDerivedType(tag: DW_TAG_member, name: "canMapHostMemory", scope: !867, file: !4, line: 1328, baseType: !84, size: 32, offset: 2944)
!895 = !DIDerivedType(tag: DW_TAG_member, name: "computeMode", scope: !867, file: !4, line: 1329, baseType: !84, size: 32, offset: 2976)
!896 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1D", scope: !867, file: !4, line: 1330, baseType: !84, size: 32, offset: 3008)
!897 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DMipmap", scope: !867, file: !4, line: 1331, baseType: !84, size: 32, offset: 3040)
!898 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLinear", scope: !867, file: !4, line: 1332, baseType: !84, size: 32, offset: 3072)
!899 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2D", scope: !867, file: !4, line: 1333, baseType: !900, size: 64, offset: 3104)
!900 = !DICompositeType(tag: DW_TAG_array_type, baseType: !84, size: 64, elements: !901)
!901 = !{!902}
!902 = !DISubrange(count: 2)
!903 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DMipmap", scope: !867, file: !4, line: 1334, baseType: !900, size: 64, offset: 3168)
!904 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLinear", scope: !867, file: !4, line: 1335, baseType: !880, size: 96, offset: 3232)
!905 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DGather", scope: !867, file: !4, line: 1336, baseType: !900, size: 64, offset: 3328)
!906 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3D", scope: !867, file: !4, line: 1337, baseType: !880, size: 96, offset: 3392)
!907 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3DAlt", scope: !867, file: !4, line: 1338, baseType: !880, size: 96, offset: 3488)
!908 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemap", scope: !867, file: !4, line: 1339, baseType: !84, size: 32, offset: 3584)
!909 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLayered", scope: !867, file: !4, line: 1340, baseType: !900, size: 64, offset: 3616)
!910 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLayered", scope: !867, file: !4, line: 1341, baseType: !880, size: 96, offset: 3680)
!911 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemapLayered", scope: !867, file: !4, line: 1342, baseType: !900, size: 64, offset: 3776)
!912 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1D", scope: !867, file: !4, line: 1343, baseType: !84, size: 32, offset: 3840)
!913 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2D", scope: !867, file: !4, line: 1344, baseType: !900, size: 64, offset: 3872)
!914 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface3D", scope: !867, file: !4, line: 1345, baseType: !880, size: 96, offset: 3936)
!915 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1DLayered", scope: !867, file: !4, line: 1346, baseType: !900, size: 64, offset: 4032)
!916 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2DLayered", scope: !867, file: !4, line: 1347, baseType: !880, size: 96, offset: 4096)
!917 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemap", scope: !867, file: !4, line: 1348, baseType: !84, size: 32, offset: 4192)
!918 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemapLayered", scope: !867, file: !4, line: 1349, baseType: !900, size: 64, offset: 4224)
!919 = !DIDerivedType(tag: DW_TAG_member, name: "surfaceAlignment", scope: !867, file: !4, line: 1350, baseType: !308, size: 64, offset: 4288)
!920 = !DIDerivedType(tag: DW_TAG_member, name: "concurrentKernels", scope: !867, file: !4, line: 1351, baseType: !84, size: 32, offset: 4352)
!921 = !DIDerivedType(tag: DW_TAG_member, name: "ECCEnabled", scope: !867, file: !4, line: 1352, baseType: !84, size: 32, offset: 4384)
!922 = !DIDerivedType(tag: DW_TAG_member, name: "pciBusID", scope: !867, file: !4, line: 1353, baseType: !84, size: 32, offset: 4416)
!923 = !DIDerivedType(tag: DW_TAG_member, name: "pciDeviceID", scope: !867, file: !4, line: 1354, baseType: !84, size: 32, offset: 4448)
!924 = !DIDerivedType(tag: DW_TAG_member, name: "pciDomainID", scope: !867, file: !4, line: 1355, baseType: !84, size: 32, offset: 4480)
!925 = !DIDerivedType(tag: DW_TAG_member, name: "tccDriver", scope: !867, file: !4, line: 1356, baseType: !84, size: 32, offset: 4512)
!926 = !DIDerivedType(tag: DW_TAG_member, name: "asyncEngineCount", scope: !867, file: !4, line: 1357, baseType: !84, size: 32, offset: 4544)
!927 = !DIDerivedType(tag: DW_TAG_member, name: "unifiedAddressing", scope: !867, file: !4, line: 1358, baseType: !84, size: 32, offset: 4576)
!928 = !DIDerivedType(tag: DW_TAG_member, name: "memoryClockRate", scope: !867, file: !4, line: 1359, baseType: !84, size: 32, offset: 4608)
!929 = !DIDerivedType(tag: DW_TAG_member, name: "memoryBusWidth", scope: !867, file: !4, line: 1360, baseType: !84, size: 32, offset: 4640)
!930 = !DIDerivedType(tag: DW_TAG_member, name: "l2CacheSize", scope: !867, file: !4, line: 1361, baseType: !84, size: 32, offset: 4672)
!931 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerMultiProcessor", scope: !867, file: !4, line: 1362, baseType: !84, size: 32, offset: 4704)
!932 = !DIDerivedType(tag: DW_TAG_member, name: "streamPrioritiesSupported", scope: !867, file: !4, line: 1363, baseType: !84, size: 32, offset: 4736)
!933 = !DIDerivedType(tag: DW_TAG_member, name: "globalL1CacheSupported", scope: !867, file: !4, line: 1364, baseType: !84, size: 32, offset: 4768)
!934 = !DIDerivedType(tag: DW_TAG_member, name: "localL1CacheSupported", scope: !867, file: !4, line: 1365, baseType: !84, size: 32, offset: 4800)
!935 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerMultiprocessor", scope: !867, file: !4, line: 1366, baseType: !308, size: 64, offset: 4864)
!936 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerMultiprocessor", scope: !867, file: !4, line: 1367, baseType: !84, size: 32, offset: 4928)
!937 = !DIDerivedType(tag: DW_TAG_member, name: "managedMemory", scope: !867, file: !4, line: 1368, baseType: !84, size: 32, offset: 4960)
!938 = !DIDerivedType(tag: DW_TAG_member, name: "isMultiGpuBoard", scope: !867, file: !4, line: 1369, baseType: !84, size: 32, offset: 4992)
!939 = !DIDerivedType(tag: DW_TAG_member, name: "multiGpuBoardGroupID", scope: !867, file: !4, line: 1370, baseType: !84, size: 32, offset: 5024)
!940 = !DIDerivedType(tag: DW_TAG_member, name: "hostNativeAtomicSupported", scope: !867, file: !4, line: 1371, baseType: !84, size: 32, offset: 5056)
!941 = !DIDerivedType(tag: DW_TAG_member, name: "singleToDoublePrecisionPerfRatio", scope: !867, file: !4, line: 1372, baseType: !84, size: 32, offset: 5088)
!942 = !DIDerivedType(tag: DW_TAG_member, name: "pageableMemoryAccess", scope: !867, file: !4, line: 1373, baseType: !84, size: 32, offset: 5120)
!943 = !DIDerivedType(tag: DW_TAG_member, name: "concurrentManagedAccess", scope: !867, file: !4, line: 1374, baseType: !84, size: 32, offset: 5152)
!944 = !DILocation(line: 98, column: 17, scope: !865)
!945 = !DILocation(line: 99, column: 2, scope: !865)
!946 = !DILocation(line: 100, column: 66, scope: !865)
!947 = !DILocation(line: 100, column: 55, scope: !865)
!948 = !DILocation(line: 100, column: 2, scope: !865)
!949 = !DILocation(line: 101, column: 2, scope: !865)
!950 = !DILocation(line: 102, column: 1, scope: !865)
!951 = distinct !DISubprogram(name: "gesummv_kernel", linkageName: "_Z14gesummv_kernelPfS_S_S_S_", scope: !1, file: !1, line: 105, type: !668, isLocal: false, isDefinition: true, scopeLine: 106, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!952 = !DILocalVariable(name: "a", arg: 1, scope: !951, file: !1, line: 105, type: !17)
!953 = !DILocation(line: 105, column: 43, scope: !951)
!954 = !DILocalVariable(name: "b", arg: 2, scope: !951, file: !1, line: 105, type: !17)
!955 = !DILocation(line: 105, column: 57, scope: !951)
!956 = !DILocalVariable(name: "x", arg: 3, scope: !951, file: !1, line: 105, type: !17)
!957 = !DILocation(line: 105, column: 71, scope: !951)
!958 = !DILocalVariable(name: "y", arg: 4, scope: !951, file: !1, line: 105, type: !17)
!959 = !DILocation(line: 105, column: 85, scope: !951)
!960 = !DILocalVariable(name: "tmp", arg: 5, scope: !951, file: !1, line: 105, type: !17)
!961 = !DILocation(line: 105, column: 99, scope: !951)
!962 = !DILocation(line: 106, column: 1, scope: !951)
!963 = !DILocation(line: 106, column: 1, scope: !964)
!964 = !DILexicalBlockFile(scope: !951, file: !1, discriminator: 1)
!965 = !DILocation(line: 106, column: 1, scope: !966)
!966 = !DILexicalBlockFile(scope: !951, file: !1, discriminator: 2)
!967 = !DILocation(line: 106, column: 1, scope: !968)
!968 = !DILexicalBlockFile(scope: !951, file: !1, discriminator: 3)
!969 = !DILocation(line: 106, column: 1, scope: !970)
!970 = !DILexicalBlockFile(scope: !951, file: !1, discriminator: 4)
!971 = !DILocation(line: 106, column: 1, scope: !972)
!972 = !DILexicalBlockFile(scope: !951, file: !1, discriminator: 5)
!973 = !DILocation(line: 119, column: 1, scope: !951)
!974 = distinct !DISubprogram(name: "gesummvCuda", linkageName: "_Z11gesummvCudaPfS_S_S_S_S_", scope: !1, file: !1, line: 121, type: !975, isLocal: false, isDefinition: true, scopeLine: 122, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!975 = !DISubroutineType(types: !976)
!976 = !{null, !17, !17, !17, !17, !17, !17}
!977 = !DILocalVariable(name: "A", arg: 1, scope: !974, file: !1, line: 121, type: !17)
!978 = !DILocation(line: 121, column: 29, scope: !974)
!979 = !DILocalVariable(name: "B", arg: 2, scope: !974, file: !1, line: 121, type: !17)
!980 = !DILocation(line: 121, column: 43, scope: !974)
!981 = !DILocalVariable(name: "x", arg: 3, scope: !974, file: !1, line: 121, type: !17)
!982 = !DILocation(line: 121, column: 57, scope: !974)
!983 = !DILocalVariable(name: "y", arg: 4, scope: !974, file: !1, line: 121, type: !17)
!984 = !DILocation(line: 121, column: 71, scope: !974)
!985 = !DILocalVariable(name: "tmp", arg: 5, scope: !974, file: !1, line: 121, type: !17)
!986 = !DILocation(line: 121, column: 85, scope: !974)
!987 = !DILocalVariable(name: "y_outputFromGpu", arg: 6, scope: !974, file: !1, line: 121, type: !17)
!988 = !DILocation(line: 121, column: 101, scope: !974)
!989 = !DILocalVariable(name: "t_start", scope: !974, file: !1, line: 123, type: !163)
!990 = !DILocation(line: 123, column: 9, scope: !974)
!991 = !DILocalVariable(name: "t_end", scope: !974, file: !1, line: 123, type: !163)
!992 = !DILocation(line: 123, column: 18, scope: !974)
!993 = !DILocalVariable(name: "A_gpu", scope: !974, file: !1, line: 125, type: !17)
!994 = !DILocation(line: 125, column: 13, scope: !974)
!995 = !DILocalVariable(name: "B_gpu", scope: !974, file: !1, line: 126, type: !17)
!996 = !DILocation(line: 126, column: 13, scope: !974)
!997 = !DILocalVariable(name: "x_gpu", scope: !974, file: !1, line: 127, type: !17)
!998 = !DILocation(line: 127, column: 13, scope: !974)
!999 = !DILocalVariable(name: "y_gpu", scope: !974, file: !1, line: 128, type: !17)
!1000 = !DILocation(line: 128, column: 13, scope: !974)
!1001 = !DILocalVariable(name: "tmp_gpu", scope: !974, file: !1, line: 129, type: !17)
!1002 = !DILocation(line: 129, column: 13, scope: !974)
!1003 = !DILocation(line: 131, column: 13, scope: !974)
!1004 = !DILocation(line: 131, column: 2, scope: !974)
!1005 = !DILocation(line: 132, column: 13, scope: !974)
!1006 = !DILocation(line: 132, column: 2, scope: !974)
!1007 = !DILocation(line: 133, column: 13, scope: !974)
!1008 = !DILocation(line: 133, column: 2, scope: !974)
!1009 = !DILocation(line: 134, column: 13, scope: !974)
!1010 = !DILocation(line: 134, column: 2, scope: !974)
!1011 = !DILocation(line: 135, column: 13, scope: !974)
!1012 = !DILocation(line: 135, column: 2, scope: !974)
!1013 = !DILocation(line: 137, column: 13, scope: !974)
!1014 = !DILocation(line: 137, column: 20, scope: !974)
!1015 = !DILocation(line: 137, column: 2, scope: !974)
!1016 = !DILocation(line: 138, column: 13, scope: !974)
!1017 = !DILocation(line: 138, column: 20, scope: !974)
!1018 = !DILocation(line: 138, column: 2, scope: !974)
!1019 = !DILocation(line: 139, column: 13, scope: !974)
!1020 = !DILocation(line: 139, column: 20, scope: !974)
!1021 = !DILocation(line: 139, column: 2, scope: !974)
!1022 = !DILocation(line: 140, column: 13, scope: !974)
!1023 = !DILocation(line: 140, column: 20, scope: !974)
!1024 = !DILocation(line: 140, column: 2, scope: !974)
!1025 = !DILocation(line: 141, column: 13, scope: !974)
!1026 = !DILocation(line: 141, column: 22, scope: !974)
!1027 = !DILocation(line: 141, column: 2, scope: !974)
!1028 = !DILocalVariable(name: "block", scope: !974, file: !1, line: 143, type: !1029)
!1029 = !DIDerivedType(tag: DW_TAG_typedef, name: "dim3", file: !1030, line: 427, baseType: !1031)
!1030 = !DIFile(filename: "/software/cuda/8.0.44/rel70/x86_64/include/vector_types.h", directory: "/home/hzhang86/cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV")
!1031 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !1030, line: 417, size: 96, elements: !1032, identifier: "_ZTS4dim3")
!1032 = !{!1033, !1034, !1035, !1036, !1040, !1049}
!1033 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1031, file: !1030, line: 419, baseType: !16, size: 32)
!1034 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1031, file: !1030, line: 419, baseType: !16, size: 32, offset: 32)
!1035 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1031, file: !1030, line: 419, baseType: !16, size: 32, offset: 64)
!1036 = !DISubprogram(name: "dim3", scope: !1031, file: !1030, line: 421, type: !1037, isLocal: false, isDefinition: false, scopeLine: 421, flags: DIFlagPrototyped, isOptimized: false)
!1037 = !DISubroutineType(types: !1038)
!1038 = !{null, !1039, !16, !16, !16}
!1039 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1031, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1040 = !DISubprogram(name: "dim3", scope: !1031, file: !1030, line: 422, type: !1041, isLocal: false, isDefinition: false, scopeLine: 422, flags: DIFlagPrototyped, isOptimized: false)
!1041 = !DISubroutineType(types: !1042)
!1042 = !{null, !1039, !1043}
!1043 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !1030, line: 383, baseType: !1044)
!1044 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !1030, line: 190, size: 96, elements: !1045, identifier: "_ZTS5uint3")
!1045 = !{!1046, !1047, !1048}
!1046 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1044, file: !1030, line: 192, baseType: !16, size: 32)
!1047 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1044, file: !1030, line: 192, baseType: !16, size: 32, offset: 32)
!1048 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1044, file: !1030, line: 192, baseType: !16, size: 32, offset: 64)
!1049 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !1031, file: !1030, line: 423, type: !1050, isLocal: false, isDefinition: false, scopeLine: 423, flags: DIFlagPrototyped, isOptimized: false)
!1050 = !DISubroutineType(types: !1051)
!1051 = !{!1043, !1039}
!1052 = !DILocation(line: 143, column: 7, scope: !974)
!1053 = !DILocalVariable(name: "grid", scope: !974, file: !1, line: 144, type: !1029)
!1054 = !DILocation(line: 144, column: 7, scope: !974)
!1055 = !DILocation(line: 144, column: 59, scope: !974)
!1056 = !DILocation(line: 144, column: 53, scope: !974)
!1057 = !DILocation(line: 144, column: 43, scope: !974)
!1058 = !DILocation(line: 144, column: 32, scope: !974)
!1059 = !DILocation(line: 144, column: 26, scope: !974)
!1060 = !DILocation(line: 144, column: 7, scope: !1061)
!1061 = !DILexicalBlockFile(scope: !974, file: !1, discriminator: 1)
!1062 = !DILocation(line: 147, column: 12, scope: !974)
!1063 = !DILocation(line: 147, column: 10, scope: !974)
!1064 = !DILocation(line: 148, column: 20, scope: !974)
!1065 = !DILocation(line: 148, column: 26, scope: !974)
!1066 = !DILocation(line: 148, column: 16, scope: !974)
!1067 = !DILocation(line: 148, column: 2, scope: !974)
!1068 = !DILocation(line: 148, column: 35, scope: !1061)
!1069 = !DILocation(line: 148, column: 41, scope: !1061)
!1070 = !DILocation(line: 148, column: 47, scope: !1061)
!1071 = !DILocation(line: 148, column: 54, scope: !1061)
!1072 = !DILocation(line: 148, column: 61, scope: !1061)
!1073 = !DILocation(line: 148, column: 2, scope: !1061)
!1074 = !DILocation(line: 149, column: 2, scope: !974)
!1075 = !DILocation(line: 150, column: 10, scope: !974)
!1076 = !DILocation(line: 150, column: 8, scope: !974)
!1077 = !DILocation(line: 151, column: 13, scope: !974)
!1078 = !DILocation(line: 151, column: 30, scope: !974)
!1079 = !DILocation(line: 151, column: 2, scope: !974)
!1080 = !DILocation(line: 153, column: 10, scope: !974)
!1081 = !DILocation(line: 153, column: 44, scope: !974)
!1082 = !DILocation(line: 153, column: 52, scope: !974)
!1083 = !DILocation(line: 153, column: 50, scope: !974)
!1084 = !DILocation(line: 153, column: 2, scope: !974)
!1085 = !DILocation(line: 154, column: 1, scope: !974)
!1086 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !1031, file: !1030, line: 421, type: !1037, isLocal: false, isDefinition: true, scopeLine: 421, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !1036, variables: !580)
!1087 = !DILocalVariable(name: "this", arg: 1, scope: !1086, type: !1088, flags: DIFlagArtificial | DIFlagObjectPointer)
!1088 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1031, size: 64)
!1089 = !DILocation(line: 0, scope: !1086)
!1090 = !DILocalVariable(name: "vx", arg: 2, scope: !1086, file: !1030, line: 421, type: !16)
!1091 = !DILocation(line: 421, column: 43, scope: !1086)
!1092 = !DILocalVariable(name: "vy", arg: 3, scope: !1086, file: !1030, line: 421, type: !16)
!1093 = !DILocation(line: 421, column: 64, scope: !1086)
!1094 = !DILocalVariable(name: "vz", arg: 4, scope: !1086, file: !1030, line: 421, type: !16)
!1095 = !DILocation(line: 421, column: 85, scope: !1086)
!1096 = !DILocation(line: 421, column: 95, scope: !1086)
!1097 = !DILocation(line: 421, column: 97, scope: !1086)
!1098 = !DILocation(line: 421, column: 102, scope: !1086)
!1099 = !DILocation(line: 421, column: 104, scope: !1086)
!1100 = !DILocation(line: 421, column: 109, scope: !1086)
!1101 = !DILocation(line: 421, column: 111, scope: !1086)
!1102 = !DILocation(line: 421, column: 116, scope: !1086)
!1103 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 157, type: !1104, isLocal: false, isDefinition: true, scopeLine: 158, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !580)
!1104 = !DISubroutineType(types: !1105)
!1105 = !{!84, !84, !383}
!1106 = !DILocalVariable(name: "argc", arg: 1, scope: !1103, file: !1, line: 157, type: !84)
!1107 = !DILocation(line: 157, column: 14, scope: !1103)
!1108 = !DILocalVariable(name: "argv", arg: 2, scope: !1103, file: !1, line: 157, type: !383)
!1109 = !DILocation(line: 157, column: 26, scope: !1103)
!1110 = !DILocalVariable(name: "t_start", scope: !1103, file: !1, line: 159, type: !163)
!1111 = !DILocation(line: 159, column: 9, scope: !1103)
!1112 = !DILocalVariable(name: "t_end", scope: !1103, file: !1, line: 159, type: !163)
!1113 = !DILocation(line: 159, column: 18, scope: !1103)
!1114 = !DILocalVariable(name: "A", scope: !1103, file: !1, line: 161, type: !17)
!1115 = !DILocation(line: 161, column: 13, scope: !1103)
!1116 = !DILocalVariable(name: "B", scope: !1103, file: !1, line: 162, type: !17)
!1117 = !DILocation(line: 162, column: 13, scope: !1103)
!1118 = !DILocalVariable(name: "x", scope: !1103, file: !1, line: 163, type: !17)
!1119 = !DILocation(line: 163, column: 13, scope: !1103)
!1120 = !DILocalVariable(name: "y", scope: !1103, file: !1, line: 164, type: !17)
!1121 = !DILocation(line: 164, column: 13, scope: !1103)
!1122 = !DILocalVariable(name: "y_outputFromGpu", scope: !1103, file: !1, line: 165, type: !17)
!1123 = !DILocation(line: 165, column: 13, scope: !1103)
!1124 = !DILocalVariable(name: "tmp", scope: !1103, file: !1, line: 166, type: !17)
!1125 = !DILocation(line: 166, column: 13, scope: !1103)
!1126 = !DILocation(line: 168, column: 18, scope: !1103)
!1127 = !DILocation(line: 168, column: 6, scope: !1103)
!1128 = !DILocation(line: 168, column: 4, scope: !1103)
!1129 = !DILocation(line: 169, column: 18, scope: !1103)
!1130 = !DILocation(line: 169, column: 6, scope: !1103)
!1131 = !DILocation(line: 169, column: 4, scope: !1103)
!1132 = !DILocation(line: 170, column: 18, scope: !1103)
!1133 = !DILocation(line: 170, column: 6, scope: !1103)
!1134 = !DILocation(line: 170, column: 4, scope: !1103)
!1135 = !DILocation(line: 171, column: 18, scope: !1103)
!1136 = !DILocation(line: 171, column: 6, scope: !1103)
!1137 = !DILocation(line: 171, column: 4, scope: !1103)
!1138 = !DILocation(line: 172, column: 32, scope: !1103)
!1139 = !DILocation(line: 172, column: 20, scope: !1103)
!1140 = !DILocation(line: 172, column: 18, scope: !1103)
!1141 = !DILocation(line: 173, column: 20, scope: !1103)
!1142 = !DILocation(line: 173, column: 8, scope: !1103)
!1143 = !DILocation(line: 173, column: 6, scope: !1103)
!1144 = !DILocation(line: 175, column: 7, scope: !1103)
!1145 = !DILocation(line: 175, column: 10, scope: !1103)
!1146 = !DILocation(line: 175, column: 2, scope: !1103)
!1147 = !DILocation(line: 177, column: 2, scope: !1103)
!1148 = !DILocation(line: 180, column: 5, scope: !1103)
!1149 = !DILocation(line: 182, column: 17, scope: !1103)
!1150 = !DILocation(line: 182, column: 20, scope: !1103)
!1151 = !DILocation(line: 182, column: 23, scope: !1103)
!1152 = !DILocation(line: 182, column: 26, scope: !1103)
!1153 = !DILocation(line: 182, column: 29, scope: !1103)
!1154 = !DILocation(line: 182, column: 34, scope: !1103)
!1155 = !DILocation(line: 182, column: 5, scope: !1103)
!1156 = !DILocation(line: 185, column: 5, scope: !1103)
!1157 = !DILocation(line: 187, column: 12, scope: !1103)
!1158 = !DILocation(line: 187, column: 10, scope: !1103)
!1159 = !DILocation(line: 188, column: 10, scope: !1103)
!1160 = !DILocation(line: 188, column: 13, scope: !1103)
!1161 = !DILocation(line: 188, column: 16, scope: !1103)
!1162 = !DILocation(line: 188, column: 19, scope: !1103)
!1163 = !DILocation(line: 188, column: 22, scope: !1103)
!1164 = !DILocation(line: 188, column: 2, scope: !1103)
!1165 = !DILocation(line: 189, column: 10, scope: !1103)
!1166 = !DILocation(line: 189, column: 8, scope: !1103)
!1167 = !DILocation(line: 190, column: 10, scope: !1103)
!1168 = !DILocation(line: 190, column: 44, scope: !1103)
!1169 = !DILocation(line: 190, column: 52, scope: !1103)
!1170 = !DILocation(line: 190, column: 50, scope: !1103)
!1171 = !DILocation(line: 190, column: 2, scope: !1103)
!1172 = !DILocation(line: 192, column: 17, scope: !1103)
!1173 = !DILocation(line: 192, column: 20, scope: !1103)
!1174 = !DILocation(line: 192, column: 2, scope: !1103)
!1175 = !DILocation(line: 194, column: 7, scope: !1103)
!1176 = !DILocation(line: 194, column: 2, scope: !1103)
!1177 = !DILocation(line: 195, column: 7, scope: !1103)
!1178 = !DILocation(line: 195, column: 2, scope: !1103)
!1179 = !DILocation(line: 196, column: 7, scope: !1103)
!1180 = !DILocation(line: 196, column: 2, scope: !1103)
!1181 = !DILocation(line: 197, column: 7, scope: !1103)
!1182 = !DILocation(line: 197, column: 2, scope: !1103)
!1183 = !DILocation(line: 198, column: 7, scope: !1103)
!1184 = !DILocation(line: 198, column: 2, scope: !1103)
!1185 = !DILocation(line: 199, column: 7, scope: !1103)
!1186 = !DILocation(line: 199, column: 2, scope: !1103)
!1187 = !DILocation(line: 201, column: 2, scope: !1103)
