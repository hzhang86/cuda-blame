/*
 *  Copyright 2014-2017 Hui Zhang
 *  All rights reserved.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef _PARAMETERS_H
#define _PARAMETERS_H

#include <string>
#include <cstring>


#define DEBUG_P
#define DEBUG_SUMMARY_CC //check completeness
#define DEBUG_ERROR
//////////////////////////
#define ENABLE_FORTRAN 
#define HUI_CHPL
#define DEBUG_CFG_CONTROLDEP
#define DEBUG_AGAINCHECK
//#define DEBUG_EXTERNFUNC
//#define HUI_C
#define REVERSE_CP_REL2
#define PARAMS_CONTRIBUTOR_FIELDS
#define ONLY_FOR_PARAM1
#define TRIM_NUMBERED_STR
#define TEMP_WORKROUND_CFG
#define ONLY_FOR_MINIMD_LINE_FROM_FIELDS2
#define ONLY_FOR_MINIMD_LINE_FROM_FIELDS
#define ONLY_FOR_MINIMD_LINE_FROM_LOADFORCALLS
#define ONLY_FOR_MINIMD_LINE_FROM_ALIASESOUT 
#define TEMP_FOR_MINIMD
//#define ADD_MULTI_LOCALE
//#define DATAPTRS_FROM_FIELDS
//#define DATAPTRS_FROM_FIELDS2
//#define NEW_FOR_PARAM1 //08/11/16
#define SPECIAL_FUNC_PTR //1/23/18 deal with ptx math intrinsics
//////////////////////////
#define DEBUG_CFG_STORELINES
#define DEBUG_GRAPH_BUILD
#define DEBUG_GRAPH_BUILD_EDGES
#define DEBUG_GRAPH_COLLAPSE
#define DEBUG_SPECIAL_PROC
#define DEBUG_RESOLVEPPA
#define DEBUG_GRAPH_IMPLICIT
#define DEBUG_LLVM_IMPLICIT
#define DEBUG_GRAPH_TRUNC

#define DEBUG_STRUCTS
#define DEBUG_ARRAYS

#define DEBUG_VP_CREATE

#define DEBUG_LLVM
#define DEBUG_LLVM_L2

#define DEBUG_LLVM_IMPLICIT

#define DEBUG_GLOBALS
//#define DEBUG_LOCALS

#define DEBUG_CFG

#define DEBUG_RP
#define DEBUG_RP_SUMMARY
#define DEBUG_IMPORTANT_VERTICES
#define DEBUG_RECURSIVE_EX_CHILDREN
#define DEBUG_LINE_NUMS
////////////////////////////
#define DEBUG_PRINT_LINE_NUMS
/////////////////////////
#define DEBUG_COMPLETENESS

#define DEBUG_EXTERN_CALLS
#define DEBUG_OUTPUT

#define DEBUG_EXIT
#define DEBUG_EXIT_OUT

#define DEBUG_SIDE_EFFECTS

#define DEBUG_CALC_RECURSIVE
#define DEBUG_ERR_RET

#define DEBUG_A_READS
#define DEBUG_A_LINE_NUMS
#define DEBUG_A_ERROR
#define DEBUG_A_IMPORTANT_VERTICES
#define DEBUG_A_LOOPS
#define DEBUG_A_GRAPH_BUILD
#define DEBUG_A_CFG
#define DEBUG_A_GRAPH_COLLAPSE
#define DEBUG_A_GRAPH_TRUNC
#define DEBUG_A_RP
#define DEBUG_A_RP_SUMMARY
#define DEBUG_A_EXIT
#define DEBUG_A_SUMMARY
#define DEBUG_A_RECURSIVE_EX_CHILDREN
#define DEBUG_A_SIDE_EFFECTS

#define PRINT_IMPLICIT 1
#define PRINT_INST_TYPE 1
#define PRINT_LINE_NUM 1
#define NO_PRINT 0

#define ALIAS_OP          100

#define MAX_PARAMS       128

#define GEP_COLLAPSE      0
#define LOAD_COLLAPSE     1
#define BITCAST_COLLAPSE  2
#define INVOKE_COLLAPSE   3

#define NO_DEF            0

#define IMPLICIT_OP           0
#define RESOLVED_EXTERN_OP    500
#define RESOLVED_MALLOC_OP    501
#define RESOLVED_L_S_OP       502
#define RESOLVED_OUTPUT_OP    997
#define GEP_BASE_OP           1000
#define GEP_OFFSET_OP         1001
#define GEP_S_FIELD_VAR_OFF_OP 1002
#define GEP_S_FIELD_OFFSET_OP  1003
//the following edge_type represents all nvvm intrinsics
#define NVVM_PTX_INTRINSIC     1004

// define PRIMITIVE for some special calls
#define NO_SPECIAL              0
#define GET_PRIVATIZEDCOPY      1
#define GET_PRIVATIZEDCLASS     2
#define GEN_COMM_GET            3
#define GEN_COMM_PUT            4
#define ACCESSHELPER            5 //not processed currently
#define CONVERTRTTYPETOVALUE    6

extern const char* PRJ_HOME_DIR;
extern const char *PARAM_REC;
extern const char *PARAM_REC2;
extern bool exclusive_blame;
#endif

/* from llvm:Type.h, retval of getTypeID()

  enum TypeID {
    // PrimitiveTypes - make sure LastPrimitiveTyID stays up to date.
    VoidTyID = 0,    ///<  0: type with no size
    HalfTyID,        ///<  1: 16-bit floating point type
    FloatTyID,       ///<  2: 32-bit floating point type
    DoubleTyID,      ///<  3: 64-bit floating point type
    X86_FP80TyID,    ///<  4: 80-bit floating point type (X87)
    FP128TyID,       ///<  5: 128-bit floating point type (112-bit mantissa)
    PPC_FP128TyID,   ///<  6: 128-bit floating point type (two 64-bits, PowerPC)
    LabelTyID,       ///<  7: Labels
    MetadataTyID,    ///<  8: Metadata
    X86_MMXTyID,     ///<  9: MMX vectors (64 bits, X86 specific)

    // Derived types... see DerivedTypes.h file.
    // Make sure FirstDerivedTyID stays up to date!
    IntegerTyID,     ///< 10: Arbitrary bit width integers
    FunctionTyID,    ///< 11: Functions
    StructTyID,      ///< 12: Structures
    ArrayTyID,       ///< 13: Arrays
    PointerTyID,     ///< 14: Pointers
    VectorTyID,      ///< 15: SIMD 'packed' format, or other vector type

    NumTypeIDs,                         // Must remain as last defined ID
    LastPrimitiveTyID = X86_MMXTyID,
    FirstDerivedTyID = IntegerTyID
  };

*/
