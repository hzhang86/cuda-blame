/*
 *  FunctionBFCCFG.h
 *  
 *  Function Control Flow Analysis part implementation
 *  Shared the same header file: FunctionBFC.h
 *
 *  Created by Hui Zhang on 03/25/15.
 *  Previous contribution by Nick Rutar
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _FUNCTION_BFC_CFG_H
#define _FUNCTION_BFC_CFG_H

//#define DEBUG_CFG_ERROR

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Instruction.def"

#include <set>

#include "NodeProps.h"
#include "FunctionBFC.h"

#include <iostream>
#include <fstream>
#include <unordered_map> //substitute for hash_map
/*
#ifdef __GNUC__
#include <ext/hash_map>
#else
#include <hash_map>
#endif

namespace std
{
  using namespace __gnu_cxx;
}
*/

/* namespace explanation
std::__gnu_cxx::list

1
std::__gnu_cxx::list mylist;

2
using namespace std;
__gnu_cxx::list mylist;

3
using namespace std::__gnuy_cxx;
list mylist;
*/

struct eqstr2 //used for unordered_map, key type should be string
{
  bool operator()(std::string s1, std::string s2) const
  {
    return s1 == s2;
  }
};

using namespace llvm;

class FunctionBFC;

class FunctionBFCBB {

public:
    std::string bbName;
	std::set<int> lineNumbers;
    std::string getName() {return bbName;}

	// To figure out the control flow and how it 
	//  affects variables we set up the gen/kill sets
	//  to use reaching definitions
	std::set<NodeProps *>  genBB;
	std::set<NodeProps *>  killBB;

	std::set<NodeProps *>  inBB;
	std::set<NodeProps *>  outBB;
	
	// Same thing but for the reaching defs for  Pointers
	std::set<NodeProps *>  genPTR_BB;
	std::set<NodeProps *>  killPTR_BB;

	std::set<NodeProps *>  inPTR_BB;
	std::set<NodeProps *>  outPTR_BB;
	
	// Ancestors and Descendants in CFG
	std::set<FunctionBFCBB *> ancestors;
	std::set<FunctionBFCBB *> descendants;
	
	void genD(FunctionBFCBB *fbb, std::set<FunctionBFCBB *> &visited);
	void genA(FunctionBFCBB *fbb, std::set<FunctionBFCBB *> &visited);
	// Predecessors and Successors in CFG
	std::set<FunctionBFCBB *>  preds;
	std::set<FunctionBFCBB *>  succs;
	
	void assignGenKill();
	void assignPTRGenKill();
	
	void sortInstructions() {std::sort(relevantInstructions.begin(), 
            relevantInstructions.end(), NodeProps::LinenumSort);}

	//  This should be in the same order as they are laid
	//    out int he LLVM code, we can start by sorting
	//    by line numbers, and then do tie breakers by 
	//    order in the LLVM code
	//  Only the NodeProps that contain stores are considered
	std::vector<NodeProps *> relevantInstructions;
	
	// relevantInstructions is usually applicable to local (non-pointer) variables
	// singleStores are for GEP grabs of pointers.  singleStores is a little misleading,
	// as the on GEP will only have one store to it, the variable the GEP grabbed from
	// may have multiple stores
	std::vector<NodeProps *> singleStores;
	llvm::BasicBlock *llvmBB;

public:
	FunctionBFCBB(llvm::BasicBlock * lbb) 
		{llvmBB = lbb; bbName = lbb->getName(); }

};


typedef std::unordered_map<std::string, FunctionBFCBB*, std::hash<std::string>, eqstr2> BBHash;

class FunctionBFCCFG {

public:
	// order doesn't really matter since we are just iterating
	//  through until we have convergence
	BBHash  LLVM_BBs;
	FunctionBFC *fb; //points to the function it builds from

	// gen pred/succ edges between FunctionBFCBBs
	void genEdges();
	// gen ancestors & descendants
	void genAD();
	void setDomLines(NodeProps * vp);
	void printCFG(std::ostream & O);
	
	// for each BB in CFG, sort the edges
	void sortCFG(); 
	// Reaching Definitions for Primitives
	void reachingDefs();
	void calcStoreLines();
	void assignBBGenKill();

	// Reaching Definitions for Pointers
	void reachingPTRDefs();
	void calcPTRStoreLines();
	void assignPTRBBGenKill();
	
    bool controlDep(NodeProps * target, NodeProps * anchor, std::ofstream &blame_info);

    //default constructor
    FunctionBFCCFG(FunctionBFC *funcBFC){
        fb = funcBFC;
    };
};

#endif
