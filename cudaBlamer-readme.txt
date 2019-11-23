==============================================================================================	
INSTALL
==============================================================================================
Attention:
	1. Tested and built on UMD deepthought2 cluster https://www.glue.umd.edu/hpcc/dt2.html
	2. Scripts and configurations need to be adjusted to different workspaces before run.
	3. More details can be checked together with chplBlamer: 
		https://github.com/hzhang86/BForChapel/blob/BFC-ml-llvm3.3-chpl1.15/README-chplBlamer.txt

1. Versions and Machine used in CUDABlamer:
	a. compilers: 	nvcc 8.0, gcc 4.8.5 and clang 4.0.1
	b. login node:  login-1.juggernaut.umd.edu
	c. dependency:	libunwind, CUPTI library, boost, ant
2. Build LLVM+clang following instructions: 
	https://releases.llvm.org/4.0.1/docs/GettingStarted.html
3. Copy the entire folder: cuda-blame/cb-staticAnalyzer to LLVM and build custom LLVM pass "BFC"
	a. check: https://releases.llvm.org/4.0.1/docs/WritingAnLLVMPass.html see how to write and build a custom pass
	b. To build:
		> make REQUIRES_RTTI=1 ENABLE_CXX11=1	
		or if build using a different node:
		> sbatch ibv-job.sh
4. Build cb-sampler:
	> cd cuda-blame/cb-sampler
	> ./build.sh
5. Build cb-postAnalyzer:
	> cd cuda-blame/cb-postAnalyzer
	> ./compileAlt.sh
6. Build cb-gui:
	> cd cuda-blame/cb-postAnalyzer
	> ./ant

===============================================================================================
END  INSTALL
===============================================================================================



==============================================================================================
RUN (Using GESUMMV as an example)
==============================================================================================
1. Change to the benchmark home:
	> cd cuda-blame/benchmarks/polybench-gpu-1.0/CUDA/GESUMMV/
2. Build binary and bitcode of the benchmark:
	> ./clang-compile.sh
3. Run static analysis on the bitcode:
	> ./static-analysis.sh
4. Run execution and get the runtime information:
	> sbatch ./gpu-job.sh
5. Prepare config files (same rule as chplBlamer):
	post_process_config.txt, post_process_config.txt
5. Run postmortem process:
	> ./post-analysis.sh
6. Run GUI:
	> ./second_step_gui.sh
==============================================================================================
END RUN
==============================================================================================