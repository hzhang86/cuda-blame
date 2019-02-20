# cuda-blame
cudaBlamer: a data-centric profiler for CUDA programs

There are Three significant differences between this tool and existing gpu profilers:
1. We offer finer-grained kernel profiling and analysis (code/data info within the kernel/devices)
2. We offer the data-centric instead of code-based profiling metrics
3. We offer full stack information from GPU to CPU (device->kernel->kenerl_launch->main)

Framework:
There are total 5 components:
1. cb-staticAnalyzer -- llvm pass to analyze the program bitcode (IR)
2. cb-sampler        -- using CUPTI and libunwind to get pc_sampling info and calling contexts
3. cb-postAnalyzer   -- using dyninst to get source info of samples and reconstruct the full calling contexts; combining static info and runtime info to generate blamed variables for each sample on each frame
4. cb-gui            -- Java-based GUI to aggregate and present profiling data
