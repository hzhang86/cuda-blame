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

#ifndef _NODE_PROPS_H
#define _NODE_PROPS_H
 
#include <string>
#include <set>
#include <vector>
#include "llvm/IR/Value.h"
#include "Parameters.h"

using namespace llvm;
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
#include <unordered_map> //sub for hash_map


#define NOT_RESOLVED 0
#define PART_RESOLVED_BLAMED 1
#define PART_RESOLVED_BLAMEE 2
#define FULL_RESOLVED_BLAMED 3
#define FULL_RESOLVED_BLAMEE 4

#define NODE_PROPS_SIZE 20

//#define NO_EXIT  0

#define ANY_EXIT 0
#define EXIT_PROG 1
#define EXIT_OUTP 2
#define EXIT_VAR 3 


#define EXIT_VAR_ALIAS 4
#define EXIT_VAR_PTR   5
#define EXIT_VAR_FIELD 6
#define EXIT_VAR_FIELD_ALIAS 7

#define EXIT_VAR_A_ALIAS 8

#define LOCAL_VAR 9
#define LOCAL_VAR_ALIAS 10
#define LOCAL_VAR_PTR   11
#define LOCAL_VAR_FIELD 12
#define LOCAL_VAR_FIELD_ALIAS 13

#define LOCAL_VAR_A_ALIAS  14

#define CALL_NODE 16
#define CALL_PARAM 17
#define CALL_RETURN 18

//we need to keep important registers like %1,%2 in `store %1, %2`
#define IMP_REG 19 //added by Hui 03/22/16

//#define EXIT_VAR_GLOBAL      19
//#define EXIT_VAR_RETURN     20
//#define EXIT_VAR_PARAM  EXIT_VAR_RETURN

#define EXIT_VAR_UNWRITTEN -1
#define NO_EXIT          0
//#define EXIT_PROG        1
//#define EXIT_OUTP        2
#define EXIT_VAR_GLOBAL  3
#define EXIT_VAR_RETURN  4
//#define EXIT_VAR_PARAM  EXIT_VAR_RETURN
#define EXIT_VAR_PARAM 5

//#define EXIT_VAR 99  // Temporary, will be overwritten



#define NO_EDGE 0
#define PARENT_EDGE 1
#define CHILD_EDGE  2
#define ALIAS_EDGE  3
#define DATA_EDGE   4
#define FIELD_EDGE  5
#define DF_ALIAS_EDGE 6
#define DF_INST_EDGE 7
#define DF_CHILD_EDGE 8
#define CALL_PARAM_EDGE 9
#define CALL_EDGE   10


//#define CALL_NODE 0
//#define CALL_PARAM 1
//#define CALL_RETURN 2

 
 
typedef std::unordered_map<int, int> LineReadHash;

 
class ExitSuper;
class NodeProps;
struct StructField;
struct StructBFC;
struct ImpNodeProps;
class FunctionBFCBB;
 
struct FuncCall {
    int paramNumber;//changed by Hui 12/31/15: now -1 represents return val,
    std::string funcName;//and -2 represents the callnode, normal params start from 0  
    int lineNum;
    ExitSuper * es;
	short resolveStatus;
    NodeProps * param;
	bool outputEvaluated;
	
    FuncCall(int pN, std::string fn) {
        paramNumber = pN;
        funcName = fn;
		lineNum = 0;
		resolveStatus = NOT_RESOLVED;
		param = NULL;
		outputEvaluated = false;
  }
  
};

struct ImpFuncCall 
{
	int paramNumber;
	
	// For call nodes it is a 'callNode', for 
	//  parameter node this is the VP for the parameter
	NodeProps * callNode;
	
	ImpFuncCall(int pn, NodeProps * cn)
	{
		paramNumber = pn;
		callNode = cn;
	}
};


struct ReadProps
{
	int lineNumber;
	int lineCount;
};


class NodeProps {

public:

	bool operator<(NodeProps rhs) { return line_num < rhs.line_num; }
	
	
	static bool LinenumSort(const NodeProps *d1, const NodeProps *d2)
	{
		return d1->line_num < d2->line_num;
	}
	
	std::string & getFullName();
	int getParamNum(std::set<NodeProps *> & visited);
	void getStructName(std::string & structName, std::set<NodeProps *> & visited);
	
    int number;
	int impNumber; // Don't know what's it
	
	
	// this is for imp vertices, assume they are all exported
	//  default value is true, if we don't want to export,
	//  then we make this false
	bool isExported;
	
    ///added by Hui 01/14/16///////////////////////////////////////////
    //We should always use string instead of const char* in c++ !!!
    //uninitialized string is empty
    std::string uniqueNameAsField; //2.P.I.8.P.3.P.topStructName
    //////////////////////////////////////////////////////////////////

    std::string name; //linkage name , mangled
    std::string realName; //demangled name, used for global vars
	
	bool calcName;
	
	// If the Vertex is a EV value > 0, otherise 0
	int paramNum; 
	
	std::string fullName;
    int line_num; //declared line number for vars, or where the instruction appears
	
	bool calcAgg;  // has the aggregate line number been calculated
	// for this vertex yet, used in calcAggregateLN()
	bool calcAggCall;
	
	
    bool isGlobal;
	bool isLocalVar;
	bool isFakeLocal;
	bool isStructure;
    bool isFormalArg; //formal args that are added to EVs (ptr>=1)
	
	// arrays treated slightly different
	bool isArr;
	bool isPtr;
	
	short ptrStatus;
	
	bool isWritten;

    //New added for special calls 03/04/17
    bool isPid;
    bool isTempPid; //helper in recursively finding pid
    bool isPidFromT2V; //helper in forward resolving pids from convertRTTValue
	bool isObj;
    bool isRemoteWritten; //due to chpl_gen_comm_* calls
    NodeProps *myObj; //If "this" is a pid, non-NULL
    NodeProps *myPid; //If "this" is a obj, non-NULL
    // very similar to aliases, except these are for pid/obj and can add lines from
    std::set<NodeProps *> objAliasesIn; //representations of same obj
    std::set<NodeProps *> objAliasesOut; 
    std::set<NodeProps *> objAliases; //representations of same obj
    
    std::set<NodeProps *> pidAliasesIn; //representations of same pid
    std::set<NodeProps *> pidAliasesOut; 
    std::set<NodeProps *> pidAliases; //representations of same pid
    std::set<NodeProps *> PPAs; //potential pid aliases for EV nodes only
    //helper set for pidAliases from collapsable pair
    std::set<NodeProps *> collapseNodes; //all deleted nodes from cp pairs
    NodeProps *collapseTo;  //the recipient node

    //used to propagate line# for GEP bases
    std::set<NodeProps *> GEPChildren; //each one has to be GEP base, bottom GEP base has no it
    NodeProps *GEPFather; //for top GEP, it's either LV or EV, for others, it has to be GEP base

	// true if this is a param that takes the blame for an extern call
	bool isBlamedExternCallParam;
	std::set<NodeProps *> blameesFromExFunc;

	bool deleted;
	
	bool resolved;
	
	//short exitStatus;
	
	// You could theoretically have one that is a return and a param (most likely)
	//  or in the case of a function pointer a call_node and a param/return
	
	//bool callStatus[3];  
	
	short eStatus;
	bool  nStatus[NODE_PROPS_SIZE];
	
	// Pointer Info (raw, refined into the sets for IVVs below
	//e.g we have: *a=load **b; store *a **c; then
    std::set<NodeProps *> aliasesIn; // c.aliasesIn.insert(b);
	std::set<NodeProps *> aliasesOut;// b.aliasesOut.insert(c);
	/*
       difference between fields and GEP, if 'this' node is a struct, then a GEP_BASE
       edge generates a field; otherwise a GEP_BASE edge generates a GEP; besides, if:
       a = load/GEP/RLS this;
       b = GEP a;
       and if this is NOT a struct and b's ptrLevel>0, then: this.GEPs.insert(b), a.GEPs.insert(b)
    */
    std::set<NodeProps *> fields; //for structures, shouldn't include itself
	std::set<NodeProps *> GEPs;  //a=GEP array, .... Then a is a GEP of array
	std::set<NodeProps *> loads; //%val = load i32* %ptr, then ptr.loads.insert(val)
	std::set<NodeProps *> nonAliasStores;//if v has no almostAlias, then it has nonAliasStores
                            //for a_(GEP/LOAD)_>b, c_(STORE)_>a, then b.nonAliasStores.insert(c)
	std::set<NodeProps *> arrayAccess;//if array A, you access a in A, then A.arrayAccess.insert(a)
	std::set<NodeProps *> almostAlias; // it's an alias to one instantiation of
	// a variable,though technically the pointers arent' the same level
	//if *a=load **b; store *a **c; store *d **c; 
    //then a and b are almostAliases respectively

	// The list of nodes that resolves to the VP through a RESOLVED_L_S_OP
    //RESOLVED_L_S_OP: resolved from the load-store operation
    //They are used as part of dataPtrs
	std::set<NodeProps *> resolvedLS;//e.g. if we have: store a, b; c=load b;
	                            //then we create c->a, a.resolvedLS.insert(c)
	
	std::set<NodeProps *> preRLS;//Before deleting any edges,kept only for EVs
    // The list of nodes that are resolved from the VP through a R_LS
    // There are some operations about it in calcLineNum(not directly adding all lines from it)
	std::set<NodeProps *> resolvedLSFrom; //c.resolvedLSFrom.insert(a);
	
	// A subset of the resolvedLS nodes that write to the data range
	//  thus causing potential side effects to all the other nodes that
	//  interact through the R_L_S
	std::set<NodeProps *> resolvedLSSideEffects;
	
	
	std::set<NodeProps *> blameeEVs;
	
	
	// Important Vertex Vectors
	std::set<NodeProps *> parents; //if store a @b, then b is a parent of a, a is a child of b
	std::set<NodeProps *> children; //if b->a(a is blamed for b), then a is b's child

	// DF_CHILD_EDGE
	std::set<NodeProps *> dfChildren; 
	std::set<NodeProps *> dfParents;
	
	std::set<NodeProps *> aliases; // taken care of with pointer analysis, can include itself
	std::set<NodeProps *> dfAliases; //aliases dictated by data flow, e.g its storesTo
                                //as long as it's a localVar
	std::set<NodeProps *> dataPtrs; //if this node is int** array, and we have 
	std::set<ImpFuncCall *> calls;  // store int* a int** array
	                           // int* b = load int** array
	                        // int* c = GEP b, 4
                            //then to array: a is child, b is load, c is dataPtr

    //added by Hui 05/10/16 for cases: store %6,a; %1=load a; call %myfunc(%1,..);
    //if %1 is written inside myfunc, then a shall get blamed as well. %1 was not put
    //into a's loads because there's a RLS between %6 and %1, so we made a new set to hold these
    std::set<NodeProps *> loadForCalls; 
    // Field info          if we have a = GEP b, 0, 1...    then	
    StructField * sField;  //b.sBFC = a.sField.parentStruct    	
    StructBFC * sBFC;
	
    std::set<int> lineNumbers; //loop line + lines that this node is as a lhs val
	std::set<int> descLineNumbers;
	
	// Mostly used with Fortran, these are the line numbers that 
	//  get put into a given paramater.  Fortran uses parm.* pain
	// in the ass struct parameters so we need to keep track of these
	// so that way the fields can access the parent struct and get the 
	// line numbers
	std::set<int> externCallLineNumbers;
	
	// the std::set of lines that are dominated by this vertex
	std::set<int> domLineNumbers;
	
	std::set<ImpFuncCall *> descCalls;
	std::set<NodeProps *> descParams;
	std::set<NodeProps *> aliasParams;
	
    std::set<NodeProps *> pointedTo; //if a=GEP b, 0, 1, 1.. then b.pointedTo.insert(a)
	std::set<NodeProps *> dominatedBy;
	
	
	// For DataFlow analysis (reaching defs for primitives)
	std::set<NodeProps *> genVP;
	std::set<NodeProps *> killVP;
	
	std::set<NodeProps *> inVP;
	std::set<NodeProps *> outVP;	
	
	NodeProps * storeFrom;   // if store int a int* b  edge: b->a
	std::set<NodeProps *> storesTo;// then a.storeFrom = b, b.storesTo.insert(a)
	//storeLines has all lines that this node's definition reaches/valid
    //all lines that this node reaches as a definition
    std::set<int> storeLines;      //one node can have many sources(like a),  
    //borderLines has farthest line# for this node's valid definition in each fbb
    //The farthest line that this node's def can reach, or where this node is killed as a definition
	std::set<int> borderLines;     //but it can only have single destination(like b)
	/////////////////////////
	
	// More DataFlow Analysis (reaching defs for pointers)
	std::set<NodeProps *> genPTR_VP;
	std::set<NodeProps *> killPTR_VP;
	
	std::set<NodeProps *> inPTR_VP;
	std::set<NodeProps *> outPTR_VP;	

	std::set<int> storePTR_Lines;
	std::set<int> borderPTR_Lines;
	
	std::set<NodeProps *>  dataWritesFrom;
	/////////////////////////////////////////////
	//added by Hui 05/10/16: help variables's lineNumOrder since that's not reliable
	//It maps line# to the order of this node appeared in that line
    //We hope there won't be multiple stores/loads to a single node on the same line#
    LineReadHash storeLineNumOrder; //case: store a, xx
    LineReadHash loadLineNumOrder;  //case: a = load xx
	///////// For Alias only operations /////
	LineReadHash readLines;
	//////////////////////////////////////////////
	
	std::set<NodeProps *> suckedInEVs;
	
	
    std::set<FuncCall *> funcCalls;//all the func calls that this node was involved(
                            //being as param/return value/the callNode(func))
    Value * llvm_inst;//the first instruction this node associated with
	//for var, it's usually alloca(for lv), for reg, it can be any inst
    //it also can be a constantExpr when the node is a gv

	// For BitCast / AddrSpaceCast Instructions
	Value * collapsed_inst;
	
	BasicBlock * bb;
    FunctionBFCBB * fbb;
	
  //Up Pointers
	NodeProps * dpUpPtr;//%val = load i32* %ptr, then val.dpUpPtr = ptr
	NodeProps * dfaUpPtr;
	
	// TODO: Probably should tweak how we handle this for
	// field aliases
	NodeProps * fieldUpPtr;

	// For field aliases
	// TODO: probably should make this a vector
	NodeProps * fieldAlias;
	
    //TODO: not sure what the following two nodes used for 
    NodeProps * pointsTo; //if a=GEP b, 0, 1, 1.. then a.pointsTo = b;
    NodeProps * exitV;
  
    // For Constants
    //int constValue;
	
	
	// For each line number, the order at which the statement appeared
	int lineNumOrder; //only effect for registers(instructions), not reliable 
                      //for variables, since they were all declared in the first
                      //line of the function with "alloca"
  
    NodeProps(int nu, std::string na, int ln, Value *pi)
    {
        number = nu;
		impNumber = -1;
        name = na;
		paramNum = 0;
		
		// We have calculated the more elaborate name
		calcName = false;
		
        line_num = ln;
		lineNumOrder = 0;
        llvm_inst = pi; // the value could be a constant(for global var) here
		collapsed_inst = NULL;
		
		dpUpPtr = this; //changed by Hui 08/11/16 from "this" to NULL
		dfaUpPtr = NULL;
		fieldUpPtr = NULL;
		fieldAlias = NULL;
		
		isExported = true;
		
		storeFrom = NULL;
		
		deleted = false;
		
		calcAgg = false;
		calcAggCall = false;
		
		resolved = false;
		
		sField = NULL;
		sBFC = NULL;
		
		bb = NULL;
		fbb = NULL;
		
		//exitStatus = NO_EXIT;
		
		eStatus = NO_EXIT;
		
		//callStatus[CALL_NODE] = false;
		//callStatus[CALL_PARAM] = false;
		//callStatus[CALL_RETURN] = false;
				
		for (int a = 0; a < NODE_PROPS_SIZE; a++)
			nStatus[a] = false;
		
		
        isGlobal = false;
		isLocalVar = false;
		isFakeLocal = false;
		
        isFormalArg = false;
		isStructure = false;
		isPtr = false;
		isWritten = false;
		
        isPid = false;
        isTempPid = false;
        isPidFromT2V = false;
        isObj = false;
        isRemoteWritten = false;
        myPid = NULL;
        myObj = NULL;
		GEPFather = NULL;//only GEP base node will have this field

		isBlamedExternCallParam = false;
		
        //printf("Address of pointsTo for %s is 0x%x\n", name.c_str(), pointsTo);
        pointsTo = NULL;
		exitV = NULL;
	
    }
	
	~NodeProps()
	{
		parents.clear();
		children.clear();
		aliases.clear();
		dataPtrs.clear();
		aliasesIn.clear();
		aliasesOut.clear();
	
        //for special calls
        pidAliasesOut.clear();
        pidAliasesIn.clear();
        pidAliases.clear();
        PPAs.clear(); //07/18/18 reserved for postmortem2 use
        objAliasesOut.clear();
        objAliasesIn.clear();
        objAliases.clear();
        blameesFromExFunc.clear();
        GEPChildren.clear();

		funcCalls.clear();
		
		fields.clear();
		GEPs.clear();
		loads.clear();
		nonAliasStores.clear();

        arrayAccess.clear();
        almostAlias.clear();
        resolvedLS.clear();
        preRLS.clear();
        resolvedLSFrom.clear();
        resolvedLSSideEffects.clear();
        blameeEVs.clear();
		
        // DF_CHILD_EDGE
        dfChildren.clear();
        dfParents.clear();
        dfAliases.clear();
        loadForCalls.clear();

		lineNumbers.clear();
		descLineNumbers.clear();
		pointedTo.clear();
		dominatedBy.clear();
		
		std::set<ImpFuncCall *>::iterator vec_ifc_i;
		for (vec_ifc_i = calls.begin(); vec_ifc_i != calls.end(); vec_ifc_i++)
			delete (*vec_ifc_i);
			
		calls.clear();
			
	}
  
    void addFuncCall(FuncCall *fc) //Moved Implementation from ..Graph.cpp
    {
        std::set<FuncCall*>::iterator vec_fc_i;
  
        for (vec_fc_i = funcCalls.begin(); vec_fc_i != funcCalls.end(); vec_fc_i++){
		    if ((*vec_fc_i)->funcName == fc->funcName && (*vec_fc_i)->paramNumber == fc->paramNumber) {
			return;
		    }
	    }
	    //std::cout<<"Pushing back call to "<<fc->funcName<<" param "<<fc->paramNumber<<" for vertex "<<name<<std::endl;
        funcCalls.insert(fc);
    }

 
};

#endif
