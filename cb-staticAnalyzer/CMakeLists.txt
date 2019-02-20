if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

# Find and add boost library
find_package(Boost 1.67.0 COMPONENTS graph)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

include_directories(${LLVM_MAIN_SRC_DIR}/lib/Target/NVPTX)

add_llvm_loadable_module( LLVMbfc
  BFC.cpp
  FunctionBFC.cpp
  FunctionBFCCFG.cpp
  FunctionBFCExFuncPass.cpp
  FunctionBFCGraph.cpp
  FunctionBFCLLVMParser.cpp
  FunctionBFCOutput.cpp
  FunctionBFCRuntime.cpp
  ModuleBFC.cpp
  Parameters.cpp

  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )
