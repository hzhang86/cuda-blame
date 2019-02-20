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

#ifndef VERTEX_PROPS_DEC
#define VERTEX_PROPS_DEC


#include <set>
#include <string>
#include <vector>

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
#include <unordered_map>

#define NODE_PROPS_SIZE 20


#define ANY_EXIT 0
#define EXIT_VAR 1 
#define EXIT_PROGG 2
#define EXIT_OUTPP 3

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



#define NO_EXIT          0
#define EXIT_PROG        1
#define EXIT_OUTP        2
#define EXIT_VAR_GLOBAL  3
#define EXIT_VAR_RETURN  4
#define EXIT_VAR_PARAM  5 //changed by Hui 03/15/16

//added for debug purpose//

//#define DEBUG_BLAMEES
//#define DEBUG_BLAMED_EXITS
//#define DEBUG_DETER_BH
//#define DEBUG_RESOLVE_LN
//#define DEBUG_GFSN0
//#define DEBUG_ATFB
//#define DEBUG_CALCPARAM_INFO
//#define DEBUG ADD_TEMP_FIELDBLAMEES

//#define DEBUG_SELINES
//#define DEBUG_GFSN
//newly added by Hui---//
//#define SUPPORT_MULTILOCALE
#define CHECK_PARAM_WRITTEN_IN_CALL
//#define DEBUG_TRANSFER_FUNC
//---------------------//

using namespace std;
class BlameFunction;
//struct BlameField;

struct StructBlame;
struct StructField;
struct SideEffectParam;

class VertexProps;

typedef std::unordered_map<int, int> LineReadHash;

/*
  For "this" node: set<FuncCall*> calls are all nodes that are related to "this" in
  function calls, e.g, a function call: a = call myfunc(b,c,d), then:
  if "this" == a/b/c/d:
    "calls" includes all the function callnodes that uses "this" as their params or
    return values, like here, calls={myfunc,..}
  else if "this" == myfunc(which is a function callnode):
    "calls" includes all the params and return value of this func call, like here,
    calls={a,b,c,d}

  the members of a FuncCall (paramNumber, Node) are corresponding attributes of
  what the FuncCall refers to as described above :))
*/

struct FuncCall 
{
	int paramNumber;
	
	// If we're looking at the VP for
	//  any of the params, this is the 
	// call node, but if we're looking
	// at the VP for the call node, this
	// is the param node
    // Which means: if it's one of calls that refers to a param, then it's the param
    // if it refers to a callNode, then it's the callNode, so we better rename it
	VertexProps *Node; //TODO: should it be better named just "Node" ?
	
	FuncCall(int pn, VertexProps * cn)
	{
		paramNumber = pn;
		Node = cn;
	}
	
};

class VertexProps {
 
 public:

	// Name of vertex
	std::string name;
    // Real name of a global/shared variable
    std::string realName;
	
	// the full struct name complete with all of the struct field parents recursively up to root
	std::string fsName;
	
	// These are created fields for bookkeeping purposes, they should be deleted after 
	//  every frame that is parsed
	bool isDerived;
	
	// Line vertex operation came from
	int declaredLine;
	
	// Vertex status (everything but exit variable status)
	int nStatus[NODE_PROPS_SIZE];
	
	// Exit variable status
	int eStatus;
 
	// Line Number for just that node (not its children)
	std::set<int> lineNumbers;
	
	// Line Number for line and all its descendants
	std::set<int> descLineNumbers;
	
	std::set<int> seLineNumbers;
	
	// Line numbers dominated by the vertex
	std::set<int> domLineNumbers;
	
	std::set<VertexProps *> descParams;
	std::set<VertexProps *> aliasParams;
	
	// Was this vertex or any of its descendants written to (or just read)
	bool isWritten;
	
	// Up Pointer to the parent Blame Function 
	BlameFunction * BF;
	
	// All the calls this vertex is associatied with
	std::vector<FuncCall *> calls;

	// DATA FLOW/POINTER RELATIONS
	std::set<VertexProps *> parents; // data flow parents
	std::set<VertexProps *> children; // data flow children
	std::set<VertexProps *> aliases;  // aliases
	
	
	VertexProps * aliasUpPtr; // only used when there is an Exit Variable (which has
	// slightly different alias rules) has an alias of a local variable 
	//  local var --- aliasUpPtr---> exitVariable
	
	std::set<VertexProps *> dfAliases; // aliases dictated by data flow
	VertexProps * dfaUpPtr;
	
	
	std::set<VertexProps *> dfChildren;
	
	std::set<VertexProps *> dataPtrs; // vertices in data space
	VertexProps * dpUpPtr;
	
	std::set<VertexProps *> fields; // fields off of this node
	
	// The list of nodes that resolves to the VP through a RESOLVED_LS_OP
	set<VertexProps *> resolvedLS;
	
	// The list of nodes that are resolved from the VP through a R_LS
	set<VertexProps *> resolvedLSFrom;
	
	// A subset of the resolvedLS nodes that write to the data range
	//  thus causing potential side effects to all the other nodes that
	//  interact through the R_L_S
	set<VertexProps *> resolvedLSSideEffects;
	
	
	VertexProps * storeFrom;
	set<VertexProps *> storesTo;
	
	
	std::set<VertexProps *> params; // params in the case this node is a call
	
	
	// Only for EVs (and their fields)
	set<VertexProps *> readableVals;
	set<VertexProps *> writableVals;

	
	
	bool calcAgg;
	bool calcAggCall;
	
	
	int calleePar; //the paramNum of "this" when dealt as callee func
	set<int> callerPars;//the paramNums of "this" when seen as from caller func
	/***************** TEMP VARIABLES  *******************/
	
	// taken out in clearPastData()
	// (tempParents, tempChildren, tempSELines, tempIsWritten, tempLine,
	//   calcAggSE, tempIsWritten)
	
	// not taken out in clearPastData()
	// TODO: check to see why this is
	// ( tempAliases)
	
	// set if we have already calculated the side effects for a given function
	bool calcAggSE;
	
	// For dealing with transfer functions, based on the blamed node in the
	//   transfer function these values may change
	std::set<VertexProps *> tempParents;
	std::set<VertexProps *> tempChildren;
	
	int tempLine;  // for temporary call nodes, it's the declared line of the call node
	
	// We need to set this because of rules we have about propagating blame
	// through read only data values, if we say it's written (which it is somewhere
	// in the call) then we allow the line number of the statement to be propagated up
	bool tempIsWritten;
    //added by Hui 04/18/16: not sure if it's duplicate with the above
    //but currently we need it to know whether this param is blamed for the call
    bool paramIsBlamedForTheCall;
	
	
	std::set<int> tempSELines;  // for the line numbers of calls involving side effects
	
	
	std::set<SideEffectParam *> tempAliases;
	
	// Side effect relations will change depending on the calling context
	//  so we clear them out after every instance
	std::set<VertexProps *> tempRelations; // for side effects
	std::set<VertexProps *> tempRelationsParent;  // same thing, reverse mapping
	
	
	// NON TRANSFER FUNCTION/SIDE EFFECT VARIABLES
	// The following values are for creating temporary fake vertices for propagating blame
	// up fields.  We don't techincally need to create these, but it makes bookkeeping much 
	// easier, we also need to make sure to free all the memory for these vertices
	std::set<VertexProps *>  tempFields;
	
	
	// These are set up in the case where an exit variable has an alias of a local variable,
	// this stores all the immediate fields of the local variable
	std::set<VertexProps *> temptempFields;
	
	
	
	// changes every time(so temp) but always explicitly set and only used for debugging output
	//  so no need to worry about resetting the variable
	short addedFromWhere;
	
	/************* END TEMP VARIABLES  *******************/
	
	/*** SPECIAL ALIAS TEMP VARIABlE ******************/
	float weight;
	/********************************************************/
	
	
	
	// In case this is a field, this data structure contains relevant information
	//BlameField * bf;
	
	StructBlame * bs; //STRUCTTYPE
	StructBlame * sType; //STRUCTPARENT
	StructField * sField;
	VertexProps * fieldUpPtr;
	
	VertexProps * fieldAlias;
	
	// General type of this vertex
	std::string genType;

    // new added for multi-locale Chapel
    bool isPid;
    bool isObj;
    VertexProps * myPid;
    VertexProps * myObj;
	std::set<VertexProps *> pidAliases;
    std::set<VertexProps *> objAliases;
    std::set<VertexProps *> PPAs;

	VertexProps(string na)
	{
		name = na;
		//aliasedFrom = NULL;
		//bf = NULL;
		bs = NULL;
        sType = NULL;
		sField = NULL;
		eStatus = NO_EXIT;
        genType = "NULL";

        //added 03/27/17 for multi-locale Chapel
        isPid = false;
        isObj = false;
        myPid = NULL;
        myObj = NULL;

        BF = NULL; //added by Hui 08/08/16
		
		weight = 1.0;
		
		calleePar = -99; //Init impossible as paramNum
                        //when it's func seen as callee
		addedFromWhere = -1;
		
		isDerived = false;
		
		calcAgg = false;
		calcAggCall = false;
		
		calcAggSE = false;
		
		storeFrom = NULL;
		isWritten = false;
		tempIsWritten = false;
		paramIsBlamedForTheCall = false; //added by Hui 04/18/16

		fieldUpPtr = NULL;
		dpUpPtr = NULL;
		dfaUpPtr = NULL;
		aliasUpPtr = NULL; //added by Hui 08/08/16

		fieldAlias = NULL;
		
		for (int a = 0; a < NODE_PROPS_SIZE; a++)
			nStatus[a] = false;
		
		tempLine = 0;
		declaredLine = 0;
	}

	void propagateTempLineUp(std::set<VertexProps *> & visited, int lineNum);
	int findBlamedExits(std::set<VertexProps *> & visited, int lineNum);
	
	void populateTFVertices(std::set<VertexProps *> & visited);

	
	void findSEExits(std::set<VertexProps *> & blamees);
	void populateSERBlamees(std::set<VertexProps *> & visited, std::set<VertexProps *> & blamees);


 
	void parseVertex(ifstream & bI, BlameFunction * bf);
	void adjustVertex();
	void printParsed(std::ostream & O);
	
	
	////// ALIAS ONLY FUNCTIONS/VARIABLES  /////////
	void parseVertex_OA(ifstream & bI, BlameFunction * bf);	
	int findBlamedExits_OAR(std::set<VertexProps *> & visited, int lineNum);

	std::set<int> lineReadNumbers;
	LineReadHash readLines;
	//////////////////////////////////////////
	

};

#endif
