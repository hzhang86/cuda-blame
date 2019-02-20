/*
 *  Copyright 2014-2017 Hui Zhang
 *  Previous contribution by Nick Rutar 
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

#ifndef _FUNCTION_BFC_H
#define _FUNCTION_BFC_H

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
//#include "llvm/Analysis/DominatorInternals.h"

#include "llvm/Pass.h"
#include "llvm/IR/InstVisitor.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Instruction.def"
#include "llvm/CodeGen/PseudoSourceValue.h"

#include "llvm/Support/Compiler.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/Dwarf.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <iterator>
#include <iostream>
#include <fstream>

#include "boost/graph/graphviz.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/iteration_macros.hpp"

//Added by Hui to substitute hash_map
#include <unordered_map>

#include "NodeProps.h"
#include "ModuleBFC.h"
#include "FunctionBFCCFG.h"
#include "Parameters.h"
#include "ExitVars.h"

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
//TC: the way to use boost
enum vertex_var_name_t { vertex_var_name };
enum vertex_props_t {vertex_props};
enum edge_iore_t { edge_iore };
// for cuda
enum func_type {KERNEL, DEVICE, HOST};

namespace boost {
  BOOST_INSTALL_PROPERTY(vertex, var_name);
  BOOST_INSTALL_PROPERTY(edge, iore);
  BOOST_INSTALL_PROPERTY(vertex, props);
}

using namespace llvm;
using namespace boost;

class ExitSuper;
class ExitVariable;
class ExitProgram;
class ExitOutput;
//struct FuncParam;
class FunctionBFC;
class ExternFunctionBFC;
class FunctionBFCCFG;
class FunctionBFCBB;
//class ModuleBFC;

struct CallInfo
{
	std::string funcName;
	int paramNumber;
};

struct FuncParam
{
	int paramNumber;
	std::string paramName;
	short writeStatus;
	std::set<FuncCall *> calledFuncs;
};

/*  Could add more metadata if we need to
 !7 = metadata !{
 i32,      ;; Tag (see below)
 metadata, ;; Context
 metadata, ;; Name
 metadata, ;; Reference to compile unit where defined
 i32,      ;; Line number where defined
 metadata  ;; Type descriptor
 }
 */


// TODO:  add this to the destructor
// This is for when we have multiple references to one field, we can then 
//  collapes all those references into one (essentially the first time we see one)
struct CollapsePair
{
// this is associated with a malloc so we'll need to free this
    std::string nameFieldCombo;
	NodeProps * collapseVertex;
	NodeProps * destVertex;
};


struct BFCLoopInfo
{
    std::set<int> lineNumbers;
	FunctionBFC * bf;
};

struct FuncCallSEElement
{
	ImpFuncCall * ifc;
	NodeProps * paramEVNode;
};

struct FuncCallSE
{
	NodeProps * callNode;
	std::vector<FuncCallSEElement *> parameters;
	std::vector<FuncCallSEElement *> NEVparams; // non EV params
};

struct ExternFunctionBFC {
    std::string funcName;
    std::set<int> paramNums;
    
    ExternFunctionBFC(std::string s) {
        funcName = s;
    }
};

struct LocalVar {
    int definedLine;
    std::string varName;
};

struct FuncStores {
    NodeProps *receiver;
    NodeProps *contents;
    int line_num;
    int lineNumOrder;
};

//The real arg format for this function
struct FuncFormalArg {
    std::string name;
    int argIdx;
    const llvm::Type *argType;
};

// Comparators have been defined in ModuleBFC.h
/*struct eqstr { //used in unordered_map, key type should be string
    bool operator()(std::string s1, std::string s2) const {
        return s1 == s2;
    }
};

struct ltstr { //used in set, key type can be const char*
    bool operator()(const char *s1, const char *s2) const {
        return strcmp(s1, s2) < 0;
    }
};*/

struct FuncSignature {
    std::string fname;
    std::string fLinkageName;
    std::vector<FuncFormalArg*> args;
    const llvm::Type *returnType;
    func_type ft; //specific for cuda, enumator type
    
    //expanded to be truncated FuncBFC
    //copied fields from FuncBFC: funcCallNames
    std::set<const char*, ltstr> calleeNames; 
    //all funcs that call "this": reverse mapping of funcCallNames
    std::set<std::string> callerNames; 
    //07/07/17:whether this function has "on/forall/coforall" constructs to rep
    //parallelism or distribution, where it can created generated wrap function with
    //unique fid used in runtime instrumentation. To warn a potential mismatch of 
    //the preSpawn/preFork stacktraces
    bool hasParOrDistCallsReal = false; //copied from FuncBFC at first, then refined
    //whether this calls a parDist func that has >1 callsites
    bool maybeMissBlamed = false;
};

typedef adjacency_list<hash_setS, vecS, bidirectionalS, property<vertex_props_t, NodeProps *>, property<edge_iore_t, int> > MyGraphType;

// needed a new graph type to support multiple edges coming from a node
typedef adjacency_list<vecS, vecS, bidirectionalS, property<vertex_props_t, NodeProps *>, property<edge_iore_t, int> > MyTruncGraphType;

typedef std::unordered_map<std::string, ExternFunctionBFC*, std::hash<std::string>, eqstr> ExternFuncBFCHash;

typedef std::unordered_map<std::string, FunctionBFC*, std::hash<std::string>, eqstr> FuncBFCHash;

typedef std::unordered_map<std::string, NodeProps*, std::hash<std::string>, eqstr> RegHashProps;

// Augmented knownFuncNames, including name&type info of its formal args
typedef std::unordered_map<std::string, FuncSignature*, std::hash<std::string>, eqstr> FuncSigHash;

typedef std::unordered_map<int, int> LineNumHash;

typedef std::unordered_map<std::string, std::set<const char*, ltstr>, std::hash<std::string>, eqstr> ImpRegSet; 

typedef std::unordered_map<std::string, NodeProps*, std::hash<std::string>, eqstr> CollapsePairHash;


class FunctionBFC {
///////////////////////// Constructors/Destructor ///////////////////////////
public:
    FunctionBFC(Function *F, FuncSigHash &kFI); //kFI=knownFuncsInfo

    ~FunctionBFC();
////////////////////////// Variables ////////////////////////////////////////
public:
    std::ofstream blame_info;
    std::set<int> blamedArgs; //Only for internal module functions 12/19/16
    bool isExternFunc; //03/07/17: distinguish extern funcs with user funcs
    bool hasParOrDistCalls; //whether this has dist/par calls
    std::set<const char*, ltstr> funcCallNames; //different called func names
                                         //if "bar" called twice in foo,only one
                                         // "bar" kept here
    func_type ft; //same as in FuncSig
private:
    //Graph representations
    MyGraphType G; // All nodes
    MyTruncGraphType G_trunc; // only important nodes
    MyTruncGraphType G_abbr; // only exit variables and calls
    //map from source lineNum to how many statements appeared in this line
    LineNumHash lnm; //#nodes that are in the same certain line, for vp->lineNumOrder
    
    std::vector<FuncStores *> allStores;

    // These two are tied together
	//std::set<std::string> collapsableFields;
	std::vector<CollapsePair *> collapsePairs;
    std::vector<CollapsePair *> autoCopyCollapsePairs;
	CollapsePairHash cpHash; //each pair is nameFieldCombo <-> Instance NodeProps*

    //Underlying LLVM Objects
    Function *func;
    //Upwards pointer to module
    Module *M;
    //Pointer to ModuleBFC
    ModuleBFC *mb;
    //unmangled name
    std::string realName; 
    //Exit Variables/Programs
    std::vector<ExitVariable *> exitVariables;
    std::vector<ExitProgram *> exitPrograms;
    ExitOutput *exitOutput;

    //Pointers
    std::set<NodeProps *> pointers;
    
    //Summary Information
    int startLineNum; // start of function
    int endLineNum; // End of function
    int numParams; // Num of params for function, MAX_PARAMS+1 for variable argument
    int numPointerParams; // Num of params that are pointers
    bool voidReturn; // true->void
    std::string moduleName; // name of file where this func found
    std::string modulePathName; // full path to the file
    bool moduleSet; // true if above two variables have been set
    bool isVarLen; // true if parameters are of variable length
    bool isBFCPoint; // whether or not it's explicit blame point(main,
                       //   V param/ V return)
    int impVertCount; // total imp vertices minus the ones not exported

    //All Func Calls occuring in the function
    std::set<FuncCall *> funcCalls;
    //All local variables in function
    std::vector<LocalVar *> localVars;
    //Important Nodes
    std::set<NodeProps *> impVertices;
    //Implicit BFC Nodes
    ImpRegSet iReg;

    RegHashProps variables; //Hash of registers that gets turned into graph
    //CFG for the class
    FunctionBFCCFG *cfg;
    //Set of all valid line numbers in this function scope
    std::set<int> allLineNums;
    //All other function informs in this module
    FuncSigHash knownFuncsInfo; //This should just be a reference since we
                                //only keep ONE instantiation for each module

	std::vector< std::pair<NodeProps *, NodeProps *> > seAliases;
	std::vector< std::pair<NodeProps *, NodeProps *> > seRelations;
	std::vector< FuncCallSE *> seCalls;

    //for specialProcess 
    //vector of pairs of <pid, obj>
	std::vector<std::pair<NodeProps*, NodeProps*>> distObjs;

///////////////////////// Generic Public Calls ///////////////////////////////
public:
    void firstPass(Function *F, std::vector<NodeProps *> &globalVars,
            ExternFuncBFCHash &efInfo, std::ostream &blame_file,
            std::ostream &blame_se_file, std::ostream &call_file, int &numMissing);

    void externFuncPass(Function *F, std::vector<NodeProps *> &globalVars, 
            ExternFuncBFCHash &efInfo, std::ostream &args_file);

    void tweakBlamedArgs();
    std::string getSourceFuncName() {return func->getName().str();}
    std::string getRealName() {return realName;}
    std::string getModuleName() {return moduleName;}
    std::string getModulePathName() {return modulePathName;}
    Module* getModule() {return M;}
    ModuleBFC* getModuleBFC() {return mb;}
  
    int numExitVariables() {return exitVariables.size();}
    int getStartLineNum() {return startLineNum;}
    int getEndLineNum() {return endLineNum;}
	
	void setModule(Module *mod) {M = mod;} 
	void setModuleBFC(ModuleBFC *modb){ mb = modb; }
	void setModuleAndPathNames(std::string file, std::string path);
	void setRealName(std::string rn) {realName = rn;}

private:
    std::string getTruncStr(std::string fullStr);
    char* trimTruncStr(const char *truncStr);
  //////////////////// Important Vertices ////////////////////////////////////////
	void populateImportantVertices();
	void recursiveExamineChildren(NodeProps *v, NodeProps *origVP, std::set<int> &visited, int preOpcode);
	
	void resolveIVCalls();
	void resolveCallsDomLines();
	
	void resolveSideEffects();
	void recursiveSEAliasHelper(std::set<NodeProps *> & visited, NodeProps *orig, NodeProps *target);
	NodeProps *resolveSideEffectsCheckParentEV(NodeProps *vp, std::set<NodeProps *> &visited);
	NodeProps *resolveSideEffectsCheckParentLV(NodeProps *vp, std::set<NodeProps *> &visited);

	void resolveSideEffectsHelper(NodeProps *rootVP, NodeProps *vp, std::set<NodeProps *> &visited);
	void resolveSideEffectCalls();
	
	void addSEAlias(NodeProps *source, NodeProps *target);
	void addSERelation(NodeProps *source, NodeProps *target);
	void resolveLooseStructs();
	void resolveTransitiveAliases();
	void resolveFieldAliases();

	void makeNewTruncGraph();
	void trimLocalVarPointers();
	int checkCompleteness();
  /////////////////////////////////////////////////////////////////////////////////////////

//////////////////////// Graph Generation ////////////////////////////////////
    //Graph generation that calls all the rest of these functions
    void genGraph(ExternFuncBFCHash &efInfo);
    void genGraphTrunc(ExternFuncBFCHash &efInfo);//Only for Chapel internal module
    //Wrapper function for boost edge adding
    void addEdge(std::string source, std::string dest, Instruction *pi, int place);
    //Edge generation
    void genEdges(Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum, std::set<NodeProps *> &seenCall);
	//generate edges based on opcode
    void geCall(Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum, std::set<NodeProps *> &seenCall);
    void geCallWrapFunc(Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum, std::set<NodeProps *> &seenCall);
	
    //Added by Hui 08/20/15
    std::string getRealStructName(std::string rawStructVarName, Value *v, User *pi,  std::string instName);
    //added by Hui 01/14/16
    std::string getUpperLevelFieldName(std::string rawStructVarName, User *pi, std::string instName);
    Value* getValueFromOrig(Instruction *vInstLoad);
    std::string getUniqueNameAsFieldForNode(Value *reg, int errNo, std::string rawStructVarName);
    std::string resolveGEPBaseFromLoad(Instruction *vInst, NodeProps *vBaseNode, std::string rawStructVarName);
    std::string getNameForVal(Value *val);

	std::string geGetElementPtr(User *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);
														 
	std::string geExtractValue(User *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);
														 
	std::string geInsertValue(User *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);
														 
	void geDefault(Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);
														 
	void geDefaultPTXIntrinsic(Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);
														 
	void geLoad(Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);
														 
	void geStore(Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);
														 
	void geMemAtomic(Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);

    void geAtomicLoadPart(Value *ptr, Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);

    void geAtomicStorePart(Value *ptr, Value *val, Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);

    void geBitCast(Instruction *pi, std::set<const char*, ltstr> &iSet, property_map<MyGraphType, vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type, int &currentLineNum);
														 
	void geBlank(Instruction *pi);
		
	void geInvoke();
		
	void addImplicitEdges(Value *v, std::set<const char*, ltstr> &iSet,property_map<MyGraphType, edge_iore_t>::type edge_type, std::string vName, bool useName);

//////////////////////// Graph CFGs  ////////////////////////////////////////
	// Takes care of assigning aliases based on the CFG using reaching definitions
    void resolveStores();
	// Cases where a store happens on a line where we also need the def in to count
	//  Cases like 
	//   (1) int x = 5;
	//   (2) x++
	//   Line (2) kills (1) but we still need the def of (1) to count
	bool resolveBorderLine(NodeProps *storeVP, NodeProps *sourceVP, NodeProps *origStoreVP, int sourceLine);
    // Cases like: x = x*10; where load and store in the same line, but store comes
    // after the load, so there shouldn't be a RLS relation between THE load and store
    bool resolveStoreLine(NodeProps *storeVP, NodeProps *sourceVP);

    void sortCFG();
    void printCFG();

    void adjustMainGraph();
	void addControlFlowChildren(NodeProps *oP, NodeProps *tP);
	void goThroughAllAliases(NodeProps *oP, NodeProps *tP, std::set<NodeProps *> &visited);
//////////////////////// Graph Analysis //////////////////////////////////////
    void findOrCreateExitVariable(std::string LLVM_node, std::string var_name);
    void addExitVar(ExitVariable *ev){exitVariables.push_back(ev);}
    void addExitProg(ExitProgram *ep){exitPrograms.push_back(ep);}
    ExitProgram *findOrCreateExitProgram(std::string &name);
    void addFuncCalls(FuncCall *fc);
	void identifyExternCalls(ExternFuncBFCHash &efInfo);
	
	int checkForUnreadReturn(NodeProps *v, int v_index);
    void recheckExitVars();
	void determineBFCHoldersLite();
    void determineBFCHoldersTrunc(); //Only for Chapel internal modules
	bool isLibraryOutput(const char* tStr);
    //for nvvm ptx intrinsics in llvm
    bool isPTXIntrinsic(Instruction *pi);
	void determineBFCForOutputVertexLite(NodeProps *v, int v_index);
	void determineBFCForVertexLite(NodeProps *v);
    void determineBFCForVertexTrunc(NodeProps *v);//Only for Chapel internal
	void LLVMtoOutputVar(NodeProps *v);
	
	void handleOneExternCall(ExternFunctionBFC *efb, NodeProps *v);
	void transferExternCallEdges(std::vector< std::pair<int, int> > &blamedNodes, int callNode, std::vector< std::pair<int, int> > &blameeNodes);
	

////////////////////  Graph Collapsing  ///////////////////////////////////////
	// Parent function
    void collapseGraph();
	// Implicit collapse -- need some work
	void collapseImplicit();
	void collapseRedundantImplicit();
	void findLocalVariable(int v_index, int top_index);
	void transferImplicitEdges(int alloc_num, int bool_num);
    void deleteOldImplicit(int v_index);
	
	// Main function that does the work
	int collapseAll(std::set<int> &collapseInstructions);

	// Sub functions that do the work
	bool shouldTransferLineNums(NodeProps *v);
    bool shouldKeepEdge(int movedOpCode, NodeProps *inTargetV);
	int transferEdgesAndDeleteNode(NodeProps *dN, NodeProps *rN,  bool transferLineNumbers=true, bool fromGEP=false);

    //It's for when multiple GEP point to one field, we don't need all the references
	void collapseRedundantFields();
    //added by Hui 05/09/16 to move in-param of a chpl__autoCopy func call, use
    //the return value as the in-param
    void collapseAutoCopyPairs();

	// Need some work
	void collapseEH();
    void collapseIO();
    int collapseInvoke();
	void handleMallocs();

	////////////////////////////////////////////////////////////////////////////////
	
    ///////////// Graph Analysis(Pointers) ////////////////////////////////////////
	void resolvePointers2();
    void againCheckIfWrittenForAllNodes();
	void resolvePointersHelper2(NodeProps *origV, int origPLevel, NodeProps *targetV, std::set<int> &visited, std::set<NodeProps *> &tempPointers, NodeProps *alias, int origOpCode);
	void resolveLocalAliases2(NodeProps *exitCand, NodeProps *currNode, std::set<int> &visited, NodeProps *exitV);
																			
	void resolveAliases2(NodeProps *exitCand, NodeProps *currNode, std::set<int> &visited, NodeProps *exitV);
																		
	bool checkIfWritten2(NodeProps *currNode, std::set<int> &visited);
																		
	void resolvePointersForNode2(NodeProps *v, std::set<NodeProps *> &tempPointers);		
	void resolveLocalDFA(NodeProps *v, std::set<NodeProps *> &pointers);		
	
	void resolveArrays(NodeProps *origV, NodeProps *v, std::set<NodeProps *> &tempPointers);

	void resolveDataReads();

	int pointerLevel(const llvm::Type *t, int counter);
    unsigned getPointedTypeID(const llvm::Type *t);
    const llvm::Type* getPointedType(const llvm::Type *t);

	void calcAggregateLN();
	void calcAggregateLNRecursive(NodeProps *ivp, std::set<NodeProps *> &vStack, std::set<NodeProps *> &vRevisit);
	void calcAggCallRecursive(NodeProps *ivp, std::set<NodeProps *> &vStack_call, std::set<NodeProps *> &vRevisit_call);
	bool isTargetNode(NodeProps *ivp);
	
	///////////////////////////////////////////////////////////////////////////////

    ///////////// Graph Analysis(Pointers) ////////////////////////////////////////
    int needExProc(std::string callName);
    void specialProcess(Instruction *pi, int specialCall, std::string callName);
    void spGetPrivatizedCopy(Instruction *pi);
    void spGetPrivatizedClass(Instruction *pi);
    void spConvertRTTypeToValue(Instruction *pi);
    void spGenCommGet(Instruction *pi);
    void spGenCommPut(Instruction *pi);
    void spAccessHelper(Instruction *pi);
    void resolvePidAliases(void);
    void resolveObjAliases(void);
    void resolvePPA(void); //find all potential pid aliases for formalArgs
    void resolvePreRLS(NodeProps *origV, NodeProps *currNode, std::set<int> &visited);
    void resolvePPAFromRLSNode(NodeProps *origV, NodeProps *currNode, std::set<int> &visited);
    void resolvePidAliasForNode_bw_new(NodeProps *currNode, std::set<int> &visited);//for Chapel 1.15
    void resolvePidAliasForNode_fw_new(NodeProps *currNode, std::set<int> &visited);//for Chapel 1.15
    void resolvePidAliasForNode_fw(NodeProps *currNode, std::set<int> &visited);//still needed for Chapel 1.15
    void resolveTransitivePidAliases(void);
    void resolveTransitiveObjAliases(void);
	///////////////////////////////////////////////////////////////////////////////
    
    //////////////////////// LLVM Parser /////////////////////////////////////////
public:
    static std::string returnTypeName(const llvm::Type *t, std::string prefix);
    bool has_suffix(const std::string &str, const std::string &suffix){
        return str.size() >= suffix.size() &&
            str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    }
private:
    // This is the function that calls all the rest of these
    void parseLLVM(std::vector<NodeProps *> &globalVars);

    // LLVM UTILITY
    void structDump(Value *compUnit);
    void structResolve(Value *v, int fieldNum, NodeProps *fieldVP);
    void pidArrayResolve(Value *v, int fieldNum, NodeProps *fieldVP, int numElems);

    // GENERAL
    void populateGlobals(std::vector<NodeProps *> &gvs);
    void adjustLVnEVs();
    bool varLengthParams();
    void printValueIDName(Value *v);
    std::string calcMetaFuncName(RegHashProps &variables, Value *v, bool isTradName, std::string nonTradName, int currentLineNum);
    void printCurrentVariables();

    // IMPLICIT
    void generateImplicits();
    void handleLoops(LoopBase<BasicBlock, Loop> *lb);
    bool errorRetCheck(User *v);
    
    void propagateConditional(DominatorTreeBase<BasicBlock> *DT , const DomTreeNodeBase<BasicBlock> *N, const char *condName, BasicBlock *termNode);
    void handleOneConditional(DominatorTreeBase<BasicBlock> *DT , const DomTreeNodeBase<BasicBlock> *N, BranchInst *br);
    void handleAllConditionals(DominatorTreeBase<BasicBlock> *DT, const DomTreeNodeBase<BasicBlock> *N, LoopInfoBase<BasicBlock, Loop> &LI, std::set<BasicBlock *> &termBlocks);
	void gatherAllDescendants(DominatorTreeBase<BasicBlock> *DT , BasicBlock *original, BasicBlock *&b, std::set<BasicBlock *> &cfgDesc, std::set<BasicBlock *> &visited);
	
    // EXPLICIT
    // Utility function to recursively get the original node that "val" bitcast from 
    NodeProps* getNodeBitCasted(Value *val);
    // Check the match situation between param and arg
    int paramTypeMatch(const llvm::Type *t1, const llvm::Type *t2);
    // Helper func for the following 2 funcs
    Value *get_args_for(Instruction *pi);
    // get real params for on_fn_chpl* and coforall_fn_chpl*
    void getParamsForCoforall(Instruction *pi, Value **params, int numArgs, std::vector<FuncFormalArg*> &args); 
    void getParamsForOn(Instruction *pi, Value **params, int numArgs, std::vector<FuncFormalArg*> &args);

    // Hash that contains unique monotonically increasing ID hashed to name of var
    void determineFunctionExitStatus();
    void examineInstruction(Instruction *pi, int &varCount, int &currentLineNum, RegHashProps &variables, FunctionBFCBB *fbb);
    void createNPFromConstantExpr(ConstantExpr *ce, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    bool firstGEPCheck(User* pi); //for ENABLE_FORTRAN
    void genDILocationInfo(Instruction *pi, int &currentLineNum, FunctionBFCBB *fbb);
    bool parseDeclareIntrinsic(Instruction *pi, int &currentLineNum, FunctionBFCBB *fbb);
    void grabVarInformation(llvm::Value *varDeclare);    
    
    void ieGen_LHS(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieGen_LHS_Alloca(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieGen_Operands(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieGen_OperandsPTXIntrinsic(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieGen_OperandsStore(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieGen_OperandsGEP(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieGen_OperandsIstVal(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieGen_OperandsExtVal(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieGen_OperandsAtomic(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieDefault(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieDefaultPTXIntrinsic(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieBlank(Instruction *pi, int &currentLineNum);
    void ieMemAtomic(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieInvoke(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieCall(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieCallWrapFunc(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieLoad(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieGetElementPtr(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieAlloca(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieStore(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieBitCast(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieSelect(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieExtractValue(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
    void ieInsertValue(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb);
 
    
    ////////////////////  OUTPUT ////////////////////////////////////////////////////////////
public:
	void exportEverything(std::ostream &O, bool reads);
	void exportSideEffects(std::ostream &O);
	void exportCalls(std::ostream &O, ExternFuncBFCHash &efInfo);
	void exportParams(std::ostream &O);
	void moreThanOneEV(int &numMultipleEV, int &afterOp1, int &afterOp2);
	void debugPrintLineNumbers(NodeProps *ivp, NodeProps *target, int locale);
    
    void printFunctionDetails(std::ostream &O);
    void printDotFiles(const char *strExtension, bool printImplicit);
	void printTruncDotFiles(const char *strExtension, bool printImplicit);
	void printFinalDot(bool printAllLines, std::string ext);
	void printFinalDotPretty(bool printAllLines, std::string ext);
	void printFinalDotAbbr(std::string ext);
	
private:
    void printToDot(std::ostream &O, bool printImplicit, bool printInstType, 
									bool printLineNum,int *opSet, int opSetSize);
	void printToDotPretty(std::ostream &O, bool printImplicit, bool printInstType, 
									bool printLineNum,int *opSet, int opSetSize );				
    void printToDotTrunc(std::ostream &O);
	void printFinalLineNums(std::ostream &O);
	
  /////////////////////////////////////////////////////////////////////////////////////////
	

};

#endif
