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

#include "FunctionBFC.h"
#include <iostream>
#include <map>
#include <time.h>
#include <ctype.h> //for the call of "isdigit"

using namespace std;


void FunctionBFC::addFuncCalls(FuncCall *fc)
{
	
    set<FuncCall*>::iterator vec_fc_i;
    for (vec_fc_i = funcCalls.begin(); vec_fc_i != funcCalls.end(); vec_fc_i++) {
		if ((*vec_fc_i)->funcName == fc->funcName && (*vec_fc_i)->paramNumber == fc->paramNumber) {
			return;
		}
	}
	
    funcCalls.insert(fc);
}

ExitProgram *FunctionBFC::findOrCreateExitProgram(string &name)
{
  for (vector<ExitProgram *>::iterator ev_i = exitPrograms.begin(); 
       ev_i != exitPrograms.end();  ev_i++) {
		if ( (*ev_i)->realName == name) {
			return *ev_i;
		}
	}
	
  ExitProgram *ep = new ExitProgram(name, BLAME_HOLDER);
  addExitProg(ep);
  return ep;
}

void FunctionBFC::LLVMtoOutputVar(NodeProps * v)
{
	exitOutput->addVertex(v);
	v->eStatus = EXIT_OUTP;
	v->nStatus[CALL_NODE] = false;
}


int FunctionBFC::checkForUnreadReturn(NodeProps *v, int v_index)
{
	boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
	
    e_beg = boost::out_edges(v_index, G).first;		
    e_end = boost::out_edges(v_index, G).second;
	
    int returnVal = 0;
    int out_d = out_degree(v_index, G);
	//added by Hui 03/26/17
    int in_d = in_degree(v_index,G);
	// Iterate over all of the outgoing edges from the vertex
    for(; e_beg != e_end; ++e_beg) {
		int opCode = get(get(edge_iore, G),*e_beg);
		bool funcName = false;
    //Early on we attached "--" along with line number information to functions
		string::size_type loc = v->name.find( "--", 0 );
		
		// if we find that string we are dealing with a function
		if( loc != string::npos ) {
			funcName = true;   
		}
		
///////////////////////////TO BE DELETED/////////////////////////////////
        //These are both for cases where
/*		 
		 tmp19  -->  |  Call to Foo | 
		 
		 where tmp19 is never read.  Common in cases such as malloc or printf
		 where there is technically a return but sometimes not read
		 
		 opCode - has to equal Call or Invoke
		 funcName - if funcName is true, then we are dealing with the call, not the return
		 out_d   -  if out_d is greater than 1, it's not an unused variable
*/
///////////////////////////////////////////////////////////////////////////
		
		if ((opCode == Instruction::Call || opCode == Instruction::Invoke) 
                && !funcName && out_d == 1 && in_d == 0)
			returnVal = 1;
		
		// If it's a local var we don't care if it's an unread return, we deal with it specially anyway 
		if (v->isLocalVar == true)
			returnVal = 0;
	}

	return returnVal;
}


// TODO 7/6/2010 Either make this much more efficient or get rid of it completely
void FunctionBFC::goThroughAllAliases(NodeProps *oP, NodeProps *tP, set<NodeProps *> &visited)
{
	if (visited.count(tP) > 0)
		return;
	
	visited.insert(tP);
	set<NodeProps *>::iterator vec_vp_i;
	for(vec_vp_i = tP->dataPtrs.begin(); vec_vp_i != tP->dataPtrs.end(); vec_vp_i++){
		NodeProps *tsDP = (*vec_vp_i);
#ifdef DEBUG_RECURSIVE_EX_CHILDREN
		blame_info<<"Checking oP "<<oP->name<<" against tsDP(2) "<<tsDP->name<<endl;
#endif
		if (cfg->controlDep(tsDP, oP, blame_info)) {
			oP->dfChildren.insert(tsDP);
			tsDP->dfParents.insert(oP);
		}
	}
	
	for (vec_vp_i = tP->dfAliases.begin(); vec_vp_i != tP->dfAliases.end(); vec_vp_i++) {
		goThroughAllAliases(oP, (*vec_vp_i), visited);
	}
	
	for (vec_vp_i = tP->aliases.begin(); vec_vp_i != tP->aliases.end(); vec_vp_i++) {
		goThroughAllAliases(oP, (*vec_vp_i), visited);
	}
}


void FunctionBFC::addControlFlowChildren(NodeProps * oP, NodeProps * tP)
{		
	NodeProps * tPSource = tP->dpUpPtr;
	if (tPSource == NULL)
		return;
	
	set<NodeProps *>::iterator vec_vp_i;
	for (vec_vp_i = tPSource->dataPtrs.begin(); vec_vp_i != tPSource->dataPtrs.end(); vec_vp_i++)
	{
		NodeProps * tsDP = (*vec_vp_i);
#ifdef DEBUG_RECURSIVE_EX_CHILDREN
		blame_info<<"Checking oP "<<oP->name<<" against tsDP "<<tsDP->name<<endl;
#endif
		
		if ( cfg->controlDep(tsDP, oP, blame_info) && tP != tsDP )
		{
			oP->dfChildren.insert(tsDP);
			tsDP->dfParents.insert(oP);
		}
	}
}


void FunctionBFC::recursiveExamineChildren(NodeProps *v, NodeProps *origVP, set<int> & visited, int preOpcode)
{
	int v_index = v->number;
	if (visited.count(v_index) > 0)
	  return;
	
#ifdef DEBUG_RECURSIVE_EX_CHILDREN
	blame_info<<"Calling recursiveExamineChildren on "<<v->name<<" for "<<origVP->name<<endl;
#endif
	visited.insert(v_index);
	
	// No edges, just get the line numbers and get out
	int out_d = out_degree(v_index, G);
	if (out_d == 0) {
	  origVP->lineNumbers.insert(v->line_num);
#ifdef DEBUG_LINE_NUMS
	  blame_info<<"Inserting line number(6) "<<v->line_num<<" to "<<origVP->name<<endl;
#endif		
	  set<int>::iterator ln_i = v->lineNumbers.begin();
	  for (; ln_i != v->lineNumbers.end(); ln_i++) {
#ifdef DEBUG_LINE_NUMS		
	    blame_info<<"Inserting line number(7) "<<*ln_i<<" to "<<origVP->name<<endl;
#endif
		origVP->lineNumbers.insert(*ln_i);
	  }
	}

	boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
	e_beg = boost::out_edges(v_index, G).first;		// edge iterator begin
	e_end = boost::out_edges(v_index, G).second;  // edge iterator end
	
	// iterate through the edges to find matching opcode
	for (; e_beg != e_end; ++e_beg) {
	  NodeProps *targetVP = get(get(vertex_props, G), target(*e_beg, G));
	  int opCode = get(get(edge_iore, G),*e_beg);
	  //int in_deg = in_degree(targetVP->number, G);		
	  // We need a store that has went through our CFG parsing and has
      //  a RESOLVED_L_S instruction as an input
	  /*if (opCode == Instruction::Store && (targetVP->storeLines.size() > 0 && in_deg != 1)) {	
	    // TODO: Investigate if this matters for local variables, 
		//  it would react the same as commenting out the next line
		//  for staticTest1
		if (v->eStatus != EXIT_VAR_GLOBAL)																
			continue;
	  }*/ 
      //above deleted by Hui 04/12/16: Not sure why we need to skip these kinda nodes
      //since it'll matter local vars like: store %1, a; %2=load a; then 'a' will lose child %1 in this case
	
      if (!targetVP) continue;
#ifdef DEBUG_RECURSIVE_EX_CHILDREN
	  blame_info<<"Looking at target "<<targetVP->name<<" from "<<v->name<<endl;
	  blame_info<<"Node Props for "<<targetVP->name<<": ";
	  for (int a = 0; a < NODE_PROPS_SIZE; a++)
		blame_info<<targetVP->nStatus[a]<<" ";
		
	  blame_info<<endl;	
      blame_info<<"Edge's opCode/edge_type="<<opCode<<endl;
#endif 
		
	  if (targetVP->eStatus > 0 || targetVP->nStatus[ANY_EXIT]) {
#ifdef DEBUG_RECURSIVE_EX_CHILDREN
		if (targetVP->exitV)
	      blame_info<<"TargetV->exitV - "<<targetVP->exitV->name<<endl;
        else
          blame_info<<"TargetV->exitV - NULL"<<endl;
		if (origVP->exitV)
		  blame_info<<"OrigVP->exitV - "<<origVP->exitV->name<<endl;
        else
          blame_info<<"OrigVP->exitV - NULL"<<endl;
		
		if (targetVP->pointsTo)
		  blame_info<<"TargetV->pointsTo - "<<targetVP->pointsTo->name<<endl;
        else
          blame_info<<"TargetV->pointsTo - NULL"<<endl;
	    if (origVP->pointsTo)
		  blame_info<<"OrigVP->pointsTo - "<<origVP->pointsTo->name<<endl;
        else
          blame_info<<"TargetV->pointsTo - NULL"<<endl;
			
        if (targetVP->dpUpPtr)
          blame_info<<"TargetV->dpUpPtr - "<<targetVP->dpUpPtr->name<<endl;
        else
          blame_info<<"TargetV->dpUpPtr - NULL"<<endl;
		if (origVP->dpUpPtr)
		  blame_info<<"OrigVP->dpUpPtr - "<<origVP->dpUpPtr->name<<endl;
        else
          blame_info<<"TargetV->dpUpPtr - NULL"<<endl;
#endif 			
			
        //For v-GEP_BASE_OP->targetV, like origV->v; v=GEP targetV,...; we don't add C/P relationship between "v-targetV" 
        //and "origV-targetV" because field shouldn't have all blamed lines as the structure, so shouldn't anyone who only 
        //depends on the field. So we added last Cond:opCode!=GEP.. EXCEPT: v is a param of previous edge
        if (opCode==GEP_BASE_OP && preOpcode!=RESOLVED_EXTERN_OP && 
            preOpcode!=Instruction::Call && preOpcode!=Instruction::Invoke) {
          origVP->lineNumbers.insert(targetVP->line_num);
          continue;
        }

        // here includes the case where opCode==GEP_BASE_OP and orig->...call(v), v->GEP_BASE_OP->targetV, 
        // in which case, we should add the GEP base to the Children of origV
        else if ((targetVP->pointsTo != origVP->pointsTo || 
                 (!targetVP->pointsTo && !origVP->pointsTo)) && 
                  targetVP->dpUpPtr != origVP->dpUpPtr) {
#ifdef DEBUG_RECURSIVE_EX_CHILDREN			
	      blame_info<<"Adding Child/Parent relation between "<<targetVP->name<<" and "<<origVP->name<<endl;
#endif
		  origVP->children.insert(targetVP);
		  targetVP->parents.insert(origVP);
				
          if ((origVP->nStatus[EXIT_VAR_PTR] || origVP->nStatus[LOCAL_VAR_PTR]) && 
              (targetVP->nStatus[EXIT_VAR_PTR] || targetVP->nStatus[LOCAL_VAR_PTR])) {
			//addControlFlowChildren(origVP, targetVP);
			set<NodeProps *> visited;
			//visited.insert(targetVP->dpUpPtr);
					
			// If statement added 7/7/2010, don't think we really need to care about the data writes data flow ancestors
			if (targetVP->isWritten == false)
			  goThroughAllAliases(origVP, targetVP->dpUpPtr, visited);	
	      }
		}
	  }
		
      else if (targetVP->eStatus >= EXIT_VAR_PARAM && !targetVP->isWritten) {
		 //cout<<"UNWRITTEN EV "<<targetVP->name<<" is sucked in by "<<origVP->name<<endl;
#ifdef DEBUG_SIDE_EFFECTS
		blame_info<<"UNWRITTEN EV "<<targetVP->name<<" is sucked in by "<<origVP->name<<endl;
#endif
		origVP->suckedInEVs.insert(targetVP);
	  }
	  //we should stop finding children on this line if we met a GEP base(father)
      else {
		origVP->lineNumbers.insert(targetVP->line_num);
#ifdef DEBUG_LINE_NUMS			
		blame_info<<"Inserting line number(8) "<<targetVP->line_num<<" to "<<origVP->name<<endl;
#endif 			
		set<int>::iterator ln_i = targetVP->lineNumbers.begin();
		for (; ln_i != targetVP->lineNumbers.end(); ln_i++) {
#ifdef DEBUG_LINE_NUMS			
	      blame_info<<"Inserting line number(9) "<<*ln_i<<" to "<<origVP->name<<endl;
#endif 				
		  origVP->lineNumbers.insert(*ln_i);
		}	
#ifdef DEBUG_RECURSIVE_EX_CHILDREN
        blame_info<<"Start recursion of recursiveExamineChildren on "<<targetVP->name<<endl;
#endif
		recursiveExamineChildren(targetVP, origVP, visited, opCode);
	  }
	}
	
#ifdef DEBUG_RECURSIVE_EX_CHILDREN	
	blame_info<<"Line nums for "<<v->name<<" (E) ";
	set<int>::iterator s_i = v->lineNumbers.begin();
	for (; s_i != v->lineNumbers.end(); s_i++) {
	  blame_info<<" "<<*s_i;
	}
	
	blame_info<<endl;
#endif
	
}



void FunctionBFC::resolveIVCalls()
{
	set<NodeProps *>::iterator ivh_i;
	
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
      NodeProps *ivp = (*ivh_i);
	  if (!ivp) continue;	
	  //clearAfter.clear();
	  if (ivp->nStatus[CALL_NODE]) {
#ifdef DEBUG_IMPORTANT_VERTICES
		blame_info<<"For CALL_NODE "<<ivp->name<<endl;
#endif		
		set<NodeProps *>::iterator ivp_i;
		for (ivp_i = ivp->parents.begin(); ivp_i != ivp->parents.end(); ivp_i++){
          NodeProps *ivpParent = (*ivp_i);
		  if (ivpParent == NULL) {// || ivpParent->name.find("GLOBAL") != string::npos)
			continue;
		  }
#ifdef DEBUG_IMPORTANT_VERTICES				
		  blame_info<<"Parent "<<ivpParent->name<<endl;
#endif
		  set<FuncCall *>::iterator vfc_i;
		  for (vfc_i = ivpParent->funcCalls.begin(); vfc_i != ivpParent->funcCalls.end(); vfc_i++) {
#ifdef DEBUG_IMPORTANT_VERTICES
			blame_info<<"Func name "<<(*vfc_i)->funcName<<", pn "<<(*vfc_i)->paramNumber<<endl;
#endif
		    if ((*vfc_i)->funcName == ivp->name) {
			  ImpFuncCall *ifcCN = new ImpFuncCall((*vfc_i)->paramNumber, ivpParent);
			  ivp->calls.insert(ifcCN);
			  ImpFuncCall *ifcCR = new ImpFuncCall((*vfc_i)->paramNumber, ivp);
			  ivpParent->calls.insert(ifcCR);
			  //clearAfter.push_back(ivpParent);
			}
		  }
		}
	  }
	}
}



void FunctionBFC::populateImportantVertices()
{
    property_map<MyGraphType, vertex_props_t>::type props = get(vertex_props, G);	
    graph_traits<MyGraphType>::vertex_iterator i, v_end;
	
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
	  NodeProps *targetVP = get(get(vertex_props, G),*i);
      if (!targetVP) continue;
	  //string na("GLOBAL");
	  //ImpNodeProps *ivpParent = new ImpNodeProps(na);
	  targetVP->nStatus[ANY_EXIT] = false; //added by Hui 03/22/16: initialize

	  short anyImp = 0;
	  for (int a = 0; a < NODE_PROPS_SIZE; a++) 
	    anyImp += targetVP->nStatus[a];
	
	  if (anyImp) 
	    targetVP->nStatus[ANY_EXIT] = true;
      //added by Hui 03/22/16: we need to keep regs like %1, %2 in 'store %1, %2'
	  else { 
	    int v_index = get(get(vertex_index, G),*i);
        boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
        e_beg = boost::out_edges(v_index, G).first;		// edge iterator begin
        e_end = boost::out_edges(v_index, G).second;    // edge iterator end	
        for (; e_beg != e_end; ++e_beg) {
          int opCode = get(get(edge_iore, G), *e_beg);
          //always keep the operands of GEP and store
          if (opCode == Instruction::Store || opCode == GEP_BASE_OP) {  
	        NodeProps *outV = get(get(vertex_props,G), target(*e_beg,G));
            if (!outV) continue;
            targetVP->nStatus[ANY_EXIT] = true;
            outV->nStatus[ANY_EXIT] = true;

            targetVP->nStatus[IMP_REG] = true;
            outV->nStatus[IMP_REG] = true;
          }
        }

#ifdef NEW_FOR_PARAM1
        //for cases like store %1, var; %1 is nothing, but we need this reg
        //Also for GEP base, we should always keep it
        boost::graph_traits<MyGraphType>::in_edge_iterator ei_beg, ei_end;
        ei_beg = boost::in_edges(v_index, G).first;		
        ei_end = boost::in_edges(v_index, G).second;    
        for (; ei_beg != ei_end; ++ei_beg) {
          int opCode = get(get(edge_iore, G), *ei_beg);
          if (opCode == Instruction::Store || opCode == GEP_BASE_OP) { //when the register is 
            //NodeProps *storeValue = get(get(vertex_props,G), target(*ei_beg,G));
            targetVP->nStatus[ANY_EXIT] = true;
            //storeValue->nStatus[ANY_EXIT] = true;
            targetVP->nStatus[IMP_REG] = true;
            //storeValue->nStatus[IMP_REG] = true;
          }
        }
#endif
	  }
    }
	
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
	  //int v_index = get(get(vertex_index, G),*i);
	  NodeProps *targetVP = get(get(vertex_props, G),*i);
      if (!targetVP) continue;
	  if (targetVP->nStatus[ANY_EXIT] || targetVP->eStatus > NO_EXIT) {
	 	//populateImportantVertex(targetVP);
		set<int> visited;
#ifdef DEBUG_IMPORTANT_VERTICES
		blame_info<<"In populateImportantVertices for "<<targetVP->name<<endl;
#endif
        //original call. preOpcode=0 since there's no edge before this call
        recursiveExamineChildren(targetVP, targetVP, visited, 0); 
#ifdef DEBUG_IMPORTANT_VERTICES
		blame_info<<"Finished Calling recursiveExamineChildren for "<<targetVP->name<<endl;
#endif
		impVertices.insert(targetVP);
	  }
	}

	impVertCount = impVertices.size();
}


//Destructor
FunctionBFC::~FunctionBFC()
{
	vector<ExitVariable *>::iterator vec_ev_i;
	for (vec_ev_i = exitVariables.begin();  vec_ev_i != exitVariables.end(); vec_ev_i++)
	    delete (*vec_ev_i);
	
	vector<ExitProgram *>::iterator vec_ep_i;
	for (vec_ep_i = exitPrograms.begin(); vec_ep_i != exitPrograms.end(); vec_ep_i++)
		delete (*vec_ep_i);
	
	delete exitOutput;
	
	pointers.clear();

	vector<LocalVar *>::iterator vec_lv_i;
	for (vec_lv_i = localVars.begin(); vec_lv_i != localVars.end(); vec_lv_i++)
		delete (*vec_lv_i);
	
	localVars.clear();
	
	impVertices.clear();
	//set<NodeProps *> vprops;
	graph_traits<MyGraphType>::vertex_iterator i, v_end;
    for(tie(i,v_end) = vertices(G); i != v_end; ++i) {
		NodeProps *v = get(get(vertex_props, G),*i);
		if (v == NULL)
			continue;
		
		delete v;
	}
	
	set<FuncCall *>::iterator vec_fc_i;
	for (vec_fc_i = funcCalls.begin(); vec_fc_i != funcCalls.end(); vec_fc_i++) 
		delete (*vec_fc_i);

	funcCalls.clear();
	variables.clear();
	iReg.clear();
	G.clear();
	G_trunc.clear();

    // free space for some new members
    blamedArgs.clear();
    vector<pair<NodeProps*, NodeProps*>>().swap(distObjs);
	vector<pair<NodeProps*, NodeProps*>>().swap(seAliases);
	vector<pair<NodeProps*, NodeProps*>>().swap(seRelations);
	vector<FuncCallSE *>().swap(seCalls);
    knownFuncsInfo.clear();
}


bool FunctionBFC::isTargetNode(NodeProps *ivp)
{
	if (ivp->nStatus[EXIT_VAR] || 
			ivp->eStatus >= EXIT_VAR_GLOBAL || ivp->eStatus == EXIT_OUTP ||
			(ivp->nStatus[LOCAL_VAR]) || 
			ivp->nStatus[EXIT_VAR_FIELD] || ivp->nStatus[LOCAL_VAR_FIELD] ||
			ivp->nStatus[EXIT_VAR_FIELD_ALIAS] || ivp->nStatus[LOCAL_VAR_FIELD_ALIAS] ||
			ivp->nStatus[CALL_PARAM] || 
			ivp->nStatus[CALL_RETURN] || ivp->nStatus[CALL_NODE] )
	{
		return true;
	}

	else
		return false;
}




void FunctionBFC::makeNewTruncGraph()
{
#ifdef DEBUG_GRAPH_TRUNC
	blame_info<<"Making trunc graph for "<<getSourceFuncName()<<" "<<impVertices.size()<<endl;
#endif
	MyTruncGraphType G_new(impVertCount);
	G_trunc.swap(G_new);
	
	MyTruncGraphType G_new2(impVertCount);
	G_abbr.swap(G_new2);
	
	bool inserted;
    graph_traits < MyTruncGraphType >::edge_descriptor ed;
	
	// Create property map for vertex properies for each register
    property_map<MyTruncGraphType, vertex_props_t>::type props = get(vertex_props, G_trunc);	
	property_map<MyTruncGraphType, vertex_props_t>::type props2 = get(vertex_props, G_abbr);	
	
  
    // Create property map for edge properties for each blame relationship
    property_map<MyTruncGraphType, edge_iore_t>::type edge_type = get(edge_iore, G_trunc);
	property_map<MyTruncGraphType, edge_iore_t>::type edge_type2 = get(edge_iore, G_abbr);
	
	//ImpVertexHash::iterator ivh_i;	
	set<NodeProps *>::iterator ivh_i;
	int num = 0;
	// Populate Nodes
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	  if (!ivp) continue;

      if (ivp->isExported) {
		put(props, num, ivp);
		put(props2, num, ivp);
		ivp->impNumber = num;
#ifdef DEBUG_GRAPH_TRUNC			
		blame_info<<"Putting IVP node "<<ivp->impNumber<<" ("<<ivp->name<<") into graph "<<ivp->eStatus<<endl;
#endif			
		num++;
	  }
	}
	// Populate Edges for G_trunc
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	  if (!ivp) continue;

      if (!ivp->isExported)
			continue;
		
	  set<NodeProps *>::iterator ivp_i;
	  set<NodeProps *>::iterator set_ivp_i;
			
	  // PARENTS
	  for (set_ivp_i = ivp->parents.begin(); set_ivp_i != ivp->parents.end(); set_ivp_i++) {
		NodeProps *ivpParent = (*set_ivp_i);
		if (!ivpParent->isExported)
	      continue;
			//if (ivpParent->name.find("GLOBAL") != string::npos)
			//	continue;
			
		if (ivp->name.find("--") != string::npos)
		  continue;
			
		if (ivp->eStatus > EXIT_VAR_GLOBAL)
		  continue;
#ifdef DEBUG_GRAPH_TRUNC			
		blame_info<<"Adding IVP P edge ("<<ivpParent->name<<","<<ivpParent->impNumber<<") to ("<<ivp->name<<","<<ivp->impNumber<<")"<<endl;
#endif
		tie(ed, inserted) = add_edge(ivpParent->impNumber, ivp->impNumber, G_trunc);
		if (inserted)
		  edge_type[ed] = PARENT_EDGE;
	  }
	
      // ALIASES
	  for (ivp_i = (*ivh_i)->aliases.begin(); ivp_i != (*ivh_i)->aliases.end(); ivp_i++) {
		NodeProps *ivpAlias = (*ivp_i);
		if (!ivpAlias) continue;

		if (!ivpAlias->isExported)
		  continue;
			
		if (ivpAlias->impNumber < 0 || ivp->impNumber < 0) {
		  //cerr<<"-1 Val A for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
#ifdef DEBUG_GRAPH_TRUNC				
		  blame_info<<"-1 Val A for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
		  blame_info<<ivpAlias->impNumber<<" "<<ivp->impNumber<<endl;
#endif				
		  continue;
		}
			
#ifdef DEBUG_GRAPH_TRUNC			
		blame_info<<"Adding IVP A edge "<<ivp->impNumber<<" to "<<ivpAlias->impNumber<<endl;
#endif
		tie(ed, inserted) = add_edge(ivp->impNumber, ivpAlias->impNumber, G_trunc);
		if (inserted)
		  edge_type[ed] = ALIAS_EDGE;
	  }
	
      // DATA
	  for (ivp_i = (*ivh_i)->dataPtrs.begin(); ivp_i != (*ivh_i)->dataPtrs.end(); ivp_i++) {
		NodeProps *ivpAlias = (*ivp_i);
		if (!ivpAlias) continue;
			
		if (!ivpAlias->isExported)
	      continue;			
			
		if (ivpAlias->impNumber < 0 || ivp->impNumber < 0) {
		  //cerr<<"-1 Val D for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
#ifdef DEBUG_GRAPH_TRUNC				
		  blame_info<<"-1 Val D for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
		  blame_info<<ivpAlias->impNumber<<" "<<ivp->impNumber<<endl;
#endif				
		  continue;
		}	
			
#ifdef DEBUG_GRAPH_TRUNC			
		blame_info<<"Adding IVP D edge "<<ivp->impNumber<<" to "<<ivpAlias->impNumber<<endl;
#endif
		tie(ed, inserted) = add_edge(ivp->impNumber, ivpAlias->impNumber, G_trunc);
		if (inserted)
		  edge_type[ed] = DATA_EDGE;
	  }
		
	  // DATA FLOW ALIAS
	  for (ivp_i = (*ivh_i)->dfAliases.begin(); ivp_i != (*ivh_i)->dfAliases.end(); ivp_i++) {
		NodeProps *ivpAlias = (*ivp_i);
		if (!ivpAlias) continue;
		
        if (!ivpAlias->isExported)
	      continue;			
			
		if (ivpAlias->impNumber < 0 || ivp->impNumber < 0) {
		  //cerr<<"-1 Val DFA for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
#ifdef DEBUG_GRAPH_TRUNC
		  blame_info<<"-1 Val DFA for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
		  blame_info<<ivpAlias->impNumber<<" "<<ivp->impNumber<<endl;
#endif				
		  continue;
		}	
			
#ifdef DEBUG_GRAPH_TRUNC
		blame_info<<"Adding IVP DFA edge "<<ivp->impNumber<<" to "<<ivpAlias->impNumber<<endl;
#endif
		tie(ed, inserted) = add_edge(ivp->impNumber, ivpAlias->impNumber, G_trunc);
		if (inserted)
		  edge_type[ed] = DF_ALIAS_EDGE;
	  }
		
	  // DATA FLOW CHILDREN
	  for (set_ivp_i = (*ivh_i)->dfChildren.begin(); set_ivp_i != (*ivh_i)->dfChildren.end(); set_ivp_i++) {
		NodeProps *ivpAlias = (*set_ivp_i);
	    if (!ivpAlias) continue;
			
		if (!ivpAlias->isExported)
	      continue;			
			
		if (ivpAlias->impNumber < 0 || ivp->impNumber < 0) {
		  //cerr<<"-1 Val DFC for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
#ifdef DEBUG_GRAPH_TRUNC				
		  blame_info<<"-1 Val DFC for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
		  blame_info<<ivpAlias->impNumber<<" "<<ivp->impNumber<<endl;
#endif			
		  continue;
		}
#ifdef DEBUG_GRAPH_TRUNC
		blame_info<<"Adding IVP DFC edge "<<ivp->impNumber<<" to "<<ivpAlias->impNumber<<endl;
#endif
		tie(ed, inserted) = add_edge(ivp->impNumber, ivpAlias->impNumber, G_trunc);
		if (inserted)
		  edge_type[ed] = DF_CHILD_EDGE;
	  }
	  // DATA FLOW INSTANTIATION
	  for (set_ivp_i = (*ivh_i)->storesTo.begin(); set_ivp_i != (*ivh_i)->storesTo.end(); set_ivp_i++) {
		NodeProps *ivpAlias = (*set_ivp_i);
	    if (!ivpAlias) continue;
			
		if (!ivpAlias->isExported)
	      continue;			
			
		if (ivpAlias->impNumber < 0 || ivp->impNumber < 0) {
		  //cerr<<"-1 Val DFI for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
#ifdef DEBUG_GRAPH_TRUNC
		  blame_info<<"-1 Val DFI for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
		  blame_info<<ivpAlias->impNumber<<" "<<ivp->impNumber<<endl;
#endif				
		  continue;
		}		
			
		if (ivpAlias->impNumber == -1 || ivp->impNumber == -1)
		  continue;
#ifdef DEBUG_GRAPH_TRUNC			
	    blame_info<<"Adding IVP DFI edge "<<ivp->impNumber<<" to "<<ivpAlias->impNumber<<endl;
#endif
	    tie(ed, inserted) = add_edge(ivp->impNumber, ivpAlias->impNumber, G_trunc);
	    if (inserted)
	      edge_type[ed] = DF_INST_EDGE;
	  }
				
	  // FIELDS
	  for (ivp_i = (*ivh_i)->fields.begin(); ivp_i != (*ivh_i)->fields.end(); ivp_i++) {
		NodeProps *ivpAlias = (*ivp_i);
		if (!ivpAlias) continue;

		if (!ivpAlias->isExported)
		  continue;			
			
		if (ivpAlias->impNumber < 0 || ivp->impNumber < 0) {
		  //cerr<<"-1 Val F for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
#ifdef DEBUG_GRAPH_TRUNC
		  blame_info<<"-1 Val F for Imp Vertex "<<ivpAlias->name<<" "<<ivp->name<<endl;
		  blame_info<<ivpAlias->impNumber<<" "<<ivp->impNumber<<endl;
#endif				
		  continue;
		}
			
#ifdef DEBUG_GRAPH_TRUNC			
		blame_info<<"Adding IVP F edge "<<ivp->impNumber<<" to "<<ivpAlias->impNumber<<endl;
#endif
		tie(ed, inserted) = add_edge(ivp->impNumber, ivpAlias->impNumber, G_trunc);
		if (inserted)
		  edge_type[ed] = FIELD_EDGE;
	  }
			
	  if (ivp->nStatus[CALL_NODE])
		continue;
		
	  // CALLS
	  set<ImpFuncCall *>::iterator ifc_i;
	  for (ifc_i = (*ivh_i)->calls.begin(); ifc_i != (*ivh_i)->calls.end(); ifc_i++) {
		ImpFuncCall *iFunc = (*ifc_i);
#ifdef DEBUG_GRAPH_TRUNC
		blame_info<<"Adding IVP C edge "<<ivp->impNumber<<" to "<<iFunc->callNode->impNumber<<endl;
#endif
		tie(ed, inserted) = add_edge(ivp->impNumber, iFunc->callNode->impNumber, G_trunc);
		if (inserted)
	      edge_type[ed] = CALL_EDGE + iFunc->paramNumber;						
	  }
	}
	
	// Populate Edges for G_abbr
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
      if (!ivp) continue;
#ifdef DEBUG_GRAPH_TRUNC
	  blame_info<<"G_abbr populate edges for "<<*ivh_i<<endl;
#endif		
	  if (!ivp->isExported)
		continue;
		
	  if (!isTargetNode(ivp))
		continue;
		
#ifdef DEBUG_GRAPH_TRUNC		
	  blame_info<<"G_abbr STILL populate edges for "<<*ivh_i<<endl;
#endif		
		
	  set<NodeProps *>::iterator ivp_i;
	  set<NodeProps *>::iterator set_ivp_i;
	  // FIELDS
	  for (ivp_i = (*ivh_i)->fields.begin(); ivp_i != (*ivh_i)->fields.end(); ivp_i++) {
		NodeProps *ivpAlias = (*ivp_i);
        if (!ivpAlias) continue;

		if (!ivpAlias->isExported)
	      continue;			
			
		if (ivpAlias->impNumber < 0 || ivp->impNumber < 0) {
		  continue;
			}
			
#ifdef DEBUG_GRAPH_TRUNC			
			blame_info<<"Adding IVP F edge(2) "<<ivp->impNumber<<" to "<<ivpAlias->impNumber<<endl;
#endif	
			tie(ed, inserted) = add_edge(ivp->impNumber, ivpAlias->impNumber, G_abbr);
			if (inserted)
				edge_type2[ed] = FIELD_EDGE;
		}
			
		if (ivp->nStatus[CALL_NODE])
			continue;
	
		// CALLS
		
		set<ImpFuncCall *>::iterator ifc_i;
		for (ifc_i = (*ivh_i)->calls.begin(); ifc_i != (*ivh_i)->calls.end(); ifc_i++) {
			ImpFuncCall *iFunc = (*ifc_i);
#ifdef DEBUG_GRAPH_TRUNC					
			blame_info<<"Adding IVP C edge(2) "<<ivp->impNumber<<" to "<<iFunc->callNode->impNumber<<endl;
#endif
			tie(ed, inserted) = add_edge(ivp->impNumber, iFunc->callNode->impNumber, G_abbr);
			if (inserted)
				edge_type2[ed] = CALL_EDGE + iFunc->paramNumber;						
		}
	
		// PARAMS
		for (set_ivp_i = ivp->descParams.begin(); set_ivp_i != ivp->descParams.end(); set_ivp_i++) {
			NodeProps *ivpAlias = (*set_ivp_i);
            if (!ivpAlias) continue;

			if (!ivpAlias->isExported)
				continue;			
			
			if (ivpAlias->impNumber < 0 || ivp->impNumber < 0) 
				continue;
			
			if (ivpAlias == ivp)
				continue;
			
#ifdef DEBUG_GRAPH_TRUNC			
			blame_info<<"Adding IVP CP edge(2) "<<ivp->impNumber<<" to "<<ivpAlias->impNumber<<endl;
#endif
			
			tie(ed, inserted) = add_edge(ivp->impNumber, ivpAlias->impNumber, G_abbr);
			if (inserted)
				edge_type2[ed] = CALL_PARAM_EDGE;
			
		}
	}
	
	string ext("_REG");
	printFinalDot(true, ext);
	printFinalDotPretty(true, ext);
	printFinalDotAbbr(ext);
}

void FunctionBFC::calcAggCallRecursive(NodeProps *ivp, set<NodeProps *> &vStack_call, set<NodeProps *> &vRevisit_call)
{

#ifdef DEBUG_CALC_RECURSIVE
	blame_info<<"Entering calcAggCallRecursive for "<<ivp->name<<endl;
#endif

	if (ivp->calcAggCall) {
#ifdef DEBUG_CALC_RECURSIVE	
	  blame_info<<"Exiting calcAggCallRecursive(1) for "<<ivp->name<<endl;
#endif		
	  return;
	}
	
	ivp->calcAggCall = true;
    vStack_call.insert(ivp); //added by Hui 03/15/16

	if (ivp->nStatus[CALL_NODE]) {
#ifdef DEBUG_CALC_RECURSIVE	
	  blame_info<<"Exiting calcAggCallRecursive(2) for "<<ivp->name<<endl;
#endif
	  return;
	}
	//ivp->descCalls.insert(ivp->calls.begin(), ivp->calls.end());
	
	
	if (ivp->nStatus[CALL_PARAM]) {	
	  ivp->descParams.insert(ivp);
	  ivp->aliasParams.insert(ivp);
#ifdef DEBUG_CALC_RECURSIVE		
	  blame_info<<"Exiting calcAggCallRecursive(3) for "<<ivp->name<<endl;
#endif
		//return;
	}
	
	if ( ivp->nStatus[CALL_RETURN]) {
	  ivp->descParams.insert(ivp);
	  ivp->aliasParams.insert(ivp);
#ifdef DEBUG_CALC_RECURSIVE		
	  blame_info<<"Exiting calcAggCallRecursive(4) for "<<ivp->name<<endl;
#endif 		
		//return;
	}

	set<NodeProps *>::iterator s_vp_i;
	set<NodeProps *>::iterator v_vp_i;
	
    // Add descParams for Pids from loadForCalls 04/06/17
    if (ivp->isPid) {
      for (s_vp_i = ivp->loadForCalls.begin(); s_vp_i != ivp->loadForCalls.end(); s_vp_i++) {
        NodeProps *child = *s_vp_i;
        if (child->calcAggCall == false)
          calcAggCallRecursive(child, vStack_call, vRevisit_call);

        if (vStack_call.count(child)) {
          vRevisit_call.insert(child);
          vRevisit_call.insert(ivp);
        }
#ifdef DEBUG_CALC_RECURSIVE		
		blame_info<<"Inserting descParams(0) for "<<ivp->name<<" from "<<child->name<<endl;
#endif 		
		ivp->descParams.insert(child->descParams.begin(), child->descParams.end());	
		//ivp->descCalls.insert(child->descCalls.begin(), child->descCalls.end());
      }
	}
      

	for (s_vp_i = ivp->children.begin(); s_vp_i != ivp->children.end(); s_vp_i++) {
	  NodeProps * child = *s_vp_i;
	  if (child->calcAggCall == false)
		calcAggCallRecursive(child, vStack_call, vRevisit_call);
		
	  if (vStack_call.count(child)) {
	  	vRevisit_call.insert(child);
		vRevisit_call.insert(ivp);
	  }
#ifdef DEBUG_CALC_RECURSIVE		
	  blame_info<<"Inserting descParams(1) for "<<ivp->name<<" from "<<child->name<<endl;
#endif 		
	  ivp->descParams.insert(child->descParams.begin(), child->descParams.end());	
	  //ivp->descCalls.insert(child->descCalls.begin(), child->descCalls.end());
	}
	
	for (s_vp_i = ivp->storesTo.begin(); s_vp_i != ivp->storesTo.end(); s_vp_i++) {
	  NodeProps *child = *s_vp_i;
	  if (child->calcAggCall == false)
		calcAggCallRecursive(child, vStack_call, vRevisit_call);
		
	  if (vStack_call.count(child)) {
	 	vRevisit_call.insert(child);
		vRevisit_call.insert(ivp);
	  }
	  if (child->nStatus[CALL_RETURN]) {			
#ifdef DEBUG_CALC_RECURSIVE		
		blame_info<<"Inserting descParams(2) for "<<ivp->name<<" from "<<child->name<<endl;
#endif 				
		ivp->descParams.insert(child->descParams.begin(), child->descParams.end());
	  }	
	}
	
	for (v_vp_i = ivp->dataPtrs.begin(); v_vp_i != ivp->dataPtrs.end(); v_vp_i++) {
	  NodeProps *child = *v_vp_i;
	  if (child->calcAggCall == false)
		calcAggCallRecursive(child, vStack_call, vRevisit_call);
		
	  if (vStack_call.count(child)) {
		vRevisit_call.insert(child);
		vRevisit_call.insert(ivp);
	  }
#ifdef DEBUG_CALC_RECURSIVE		
	  blame_info<<"Inserting descParams(3) for "<<ivp->name<<" from "<<child->name<<endl;
#endif 	
	  ivp->descParams.insert(child->descParams.begin(), child->descParams.end());
	  ivp->aliasParams.insert(child->descParams.begin(), child->descParams.end());	
	  //ivp->descCalls.insert(child->descCalls.begin(), child->descCalls.end());
	}
	
	for (v_vp_i = ivp->aliases.begin(); v_vp_i != ivp->aliases.end(); v_vp_i++) {
	  NodeProps *child = *v_vp_i;
	  if (child->calcAggCall == false)
		calcAggCallRecursive(child, vStack_call, vRevisit_call);

	  if (vStack_call.count(child)) {
		vRevisit_call.insert(child);
		vRevisit_call.insert(ivp);
	  }
#ifdef DEBUG_CALC_RECURSIVE		
	  blame_info<<"Inserting descParams(4) for "<<ivp->name<<" from "<<child->name<<endl;
#endif 				
	  ivp->descParams.insert(child->descParams.begin(), child->descParams.end());
	  ivp->aliasParams.insert(child->descParams.begin(), child->descParams.end());
	  //ivp->descCalls.insert(child->descCalls.begin(), child->descCalls.end());
	}
	
	for (v_vp_i = ivp->dfAliases.begin(); v_vp_i != ivp->dfAliases.end(); v_vp_i++) {
	  NodeProps *child = *v_vp_i;
	  if (child->calcAggCall == false)
		calcAggCallRecursive(child, vStack_call, vRevisit_call);

	  if (vStack_call.count(child)) {
		vRevisit_call.insert(child);
		vRevisit_call.insert(ivp);
	  }
#ifdef DEBUG_CALC_RECURSIVE		
	  blame_info<<"Inserting descParams(5) for "<<ivp->name<<" from "<<child->name<<endl;
#endif 				
	  ivp->descParams.insert(child->descParams.begin(), child->descParams.end());
	  ivp->aliasParams.insert(child->descParams.begin(), child->descParams.end());
	}

#ifdef PARAMS_CONTRIBUTOR_FIELDS
    for (v_vp_i = ivp->fields.begin(); v_vp_i != ivp->fields.end(); v_vp_i++) {
      NodeProps *child = *v_vp_i;
      if (child->calcAggCall == false)
        calcAggCallRecursive(child, vStack_call, vRevisit_call);

	  if (vStack_call.count(child)) {
		vRevisit_call.insert(child);
		vRevisit_call.insert(ivp);
	  }
#ifdef DEBUG_CALC_RECURSIVE		
	  blame_info<<"Inserting descParams(6) for "<<ivp->name<<" from "<<child->name<<endl;
#endif 				
	  ivp->descParams.insert(child->descParams.begin(), child->descParams.end());
	  ivp->aliasParams.insert(child->descParams.begin(), child->descParams.end());
	}
#endif

#ifdef DEBUG_CALC_RECURSIVE
	blame_info<<"Exiting calcAggCallRecursive(N) for "<<ivp->name<<endl;
#endif 		

    vStack_call.erase(ivp);

}


void FunctionBFC::debugPrintLineNumbers(NodeProps *ivp, NodeProps *target,int locale)
{
#ifdef DEBUG_PRINT_LINE_NUMS
	blame_info<<"Inserting Line Numbers("<<locale<<") from "<<target->name<<" to "<<ivp->name<<endl;
	set<int>::iterator set_i_i;
	
	for (set_i_i = target->descLineNumbers.begin(); set_i_i != target->descLineNumbers.end(); set_i_i++) {
		blame_info<<*set_i_i<<" ";
	}
	
	blame_info<<endl;
#endif
}



void FunctionBFC::calcAggregateLNRecursive(NodeProps *ivp, set<NodeProps *> &vStack, set<NodeProps *> &vRevisit)
{
	if (ivp->calcAgg)
	  return;

	ivp->calcAgg = true;
	vStack.insert(ivp);
	set<int>::iterator set_i_i;

#ifdef DEBUG_PRINT_LINE_NUMS
	blame_info<<endl;
	blame_info<<"Entering calcAggregateLNRecursive for "<<ivp->name<<endl;
	blame_info<<"Starting line number tally for "<<ivp->name<<" originally:"<<endl;
	
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}
    blame_info<<endl;
#endif
	
	ivp->descLineNumbers.insert(ivp->line_num);
#ifdef DEBUG_PRINT_LINE_NUMS
	blame_info<<"After insert line_num of "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif 

    //added by Hui 04/02/16: Constants shouldn't have extra lines from its 'related nodes'
    if (ivp->name.find("Constant+") != string::npos) {
      blame_info<<ivp->name<<" is not needed in calcAggregateLNRecursive"<<endl;
      return;
    }

	ivp->descLineNumbers.insert(ivp->lineNumbers.begin(), ivp->lineNumbers.end());
#ifdef DEBUG_PRINT_LINE_NUMS
	blame_info<<"After insert lineNumbers(-1) from baseline for "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif
    // added on 08/22/17: stop here if we are using exclusive blame
    if (exclusive_blame)
      return;

	// TODO: DETAILS BELOW
	//7/12/2010  INVESTIGATE FURTHER, do we need this still and why
    //commented out on 11/03/17, not necessary for CUDA right now
/*
#ifdef ONLY_FOR_PARAM1
    if (ivp->storeFrom && (ivp->isPtr || ivp->nStatus[LOCAL_VAR_PTR] || ivp->nStatus[EXIT_VAR_PTR])) {//changed by Hui on 08/03/15
		//ivp->descLineNumbers.insert(ivp->storeFrom->line_num);
        blame_info<<"ivp->eStatus="<<ivp->eStatus<<endl;
        blame_info<<"ivp->nStatus:  ";
        for(int a=0; a< NODE_PROPS_SIZE; a++)
            blame_info<<ivp->nStatus[a]<<" ";
        blame_info<<endl;
        NodeProps *child = ivp->storeFrom;
		
        if (child->calcAgg == false)
			calcAggregateLNRecursive(child, vStack, vRevisit);
		
		if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
			blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
			vRevisit.insert(child);
			vRevisit.insert(ivp);
		}
        
        ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
#ifdef DEBUG_PRINT_LINE_NUMS
	    blame_info<<"After storeFrom: "<<ivp->name<<endl;
	    for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
		    blame_info<<*set_i_i<<" ";
	    }
	
	    blame_info<<endl;
#endif 
    }
#endif
*/
	set<NodeProps *>::iterator s_vp_i;
	set<NodeProps *>::iterator s_vp_i2;
	
	set<NodeProps *>::iterator v_vp_i;
	set<NodeProps *>::iterator v_vp_i2;

    //added by Hui 04/04/16: intuitively, we should add-in storesTo, not storeFrom
    for (s_vp_i = ivp->storesTo.begin(); s_vp_i != ivp->storesTo.end(); s_vp_i++) {
      NodeProps *child = *s_vp_i;
      if (child->calcAgg == false)
	    calcAggregateLNRecursive(child, vStack, vRevisit);
	
	  if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
		blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
    	vRevisit.insert(child);
		vRevisit.insert(ivp);
	  }

      ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
	  debugPrintLineNumbers(ivp, child, 0);
	}
#ifdef DEBUG_PRINT_LINE_NUMS
    blame_info<<"After storesTo: "<<ivp->name<<endl;
    for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
      blame_info<<*set_i_i<<" ";
    }
    blame_info<<endl;
#endif 

	if ((ivp->nStatus[LOCAL_VAR_PTR] || ivp->nStatus[EXIT_VAR_PTR]) && ivp->isWritten == false) {		
	  if (ivp->dpUpPtr != ivp && ivp->dpUpPtr != NULL) {
		for (s_vp_i = ivp->dpUpPtr->dataPtrs.begin(); s_vp_i != ivp->dpUpPtr->dataPtrs.end(); s_vp_i++) {
	      NodeProps *cand = *s_vp_i;
		  if (cand == NULL) {
#ifdef DEBUG_ERROR
			blame_info<<"Why is this NULL?"<<endl;
#endif
			continue;
  		  }
				
		  if (cand == ivp)
			continue;
			
		  if (cfg->controlDep(cand, ivp, blame_info)) {
			//if (cand->storePTR_Lines.count(ivp->line_num))
#ifdef DEBUG_LINE_NUMS
			blame_info<<"Adding lines from the dominating data write for "<<cand->name<<endl;
#endif
			ivp->dataWritesFrom.insert(cand);				
			NodeProps *child = cand;
		    if (child->calcAgg == false)
			  calcAggregateLNRecursive(child, vStack, vRevisit);
					
			if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
			  blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif 
              vRevisit.insert(child);
			  vRevisit.insert(ivp);
			}
					
		    if (child->isWritten) {
			  ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
			  debugPrintLineNumbers(ivp, child, 1);
			}
		  }
		}
			
		for (v_vp_i = ivp->dpUpPtr->aliasesIn.begin(); v_vp_i != ivp->dpUpPtr->aliasesIn.end(); v_vp_i++) {			
          NodeProps *alias = *v_vp_i;
#ifdef DEBUG_LINE_NUMS
		  blame_info<<"Examining DPs for alias "<<alias->name<<endl;
#endif
		  for (s_vp_i = alias->dataPtrs.begin(); s_vp_i != alias->dataPtrs.end(); s_vp_i++) {
		    NodeProps *cand = *s_vp_i;
#ifdef DEBUG_LINE_NUMS
			blame_info<<"Examining candidate "<<cand->name<<endl;
#endif
      	   	if (cand == NULL) {
#ifdef DEBUG_ERROR
              blame_info<<"Why is this NULL?"<<endl;
#endif
              continue;
            }
					
			if (cand == ivp)
			  continue;
					
			if (cfg->controlDep(cand, ivp, blame_info)) {
			  //if (cand->storePTR_Lines.count(ivp->line_num))
#ifdef DEBUG_LINE_NUMS
  			  blame_info<<"Adding lines from the dominating data write for(2) "<<cand->name<<endl;
#endif
              ivp->dataWritesFrom.insert(cand);
			  NodeProps *child = cand;
			  if (child->calcAgg == false)
				calcAggregateLNRecursive(child, vStack, vRevisit);
						
			  if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
				blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
				vRevisit.insert(child);
				vRevisit.insert(ivp);
  			  }
						
			  if (child->isWritten) {
			    ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
				debugPrintLineNumbers(ivp, child, 1);		
			  }
			}
		  }
		}
		//Commented out by Hui 11/03/17, why do we need to add this part after we've had the previous one: aliasIn	
        /*
        for (s_vp_i2 = ivp->dpUpPtr->aliases.begin(); s_vp_i2 != ivp->dpUpPtr->aliases.end(); s_vp_i2++) {
            NodeProps *alias = *s_vp_i2;
            
        #ifdef DEBUG_LINE_NUMS
            blame_info<<"Examining DPs(2) for alias "<<alias->name<<endl;
        #endif
            for (s_vp_i = alias->dataPtrs.begin(); s_vp_i != alias->dataPtrs.end(); s_vp_i++) {
                NodeProps *cand = *s_vp_i;
            #ifdef DEBUG_LINE_NUMS
                blame_info<<"Examining(2) candidate "<<cand->name<<endl;
            #endif
                if (cand == NULL) {
                #ifdef DEBUG_ERROR
                    blame_info<<"Why is this NULL?"<<endl;
                #endif
                    continue;
                }
                
                if (cand == ivp)
                    continue;
                
                if (cfg->controlDep(cand, ivp, blame_info)) {
                    //if (cand->storePTR_Lines.count(ivp->line_num))
                #ifdef DEBUG_LINE_NUMS
                    blame_info<<"Adding lines from the dominating data write for(3) "<<cand->name<<endl;
                #endif
                    ivp->dataWritesFrom.insert(cand);
                    NodeProps *child = cand;
                    if (child->calcAgg == false)
                        calcAggregateLNRecursive(child, vStack, vRevisit);
                    
                    if (vStack.count(child)) {
                    #ifdef DEBUG_LINE_NUMS
                        blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
                    #endif
                        vRevisit.insert(child);
                        vRevisit.insert(ivp);
                    }						
                    if (child->isWritten) {
                        ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
                        debugPrintLineNumbers(ivp, child, 1);
                        
                    }
                }
            }
        }
        */
	  }
	}
	
#ifdef DEBUG_LINE_NUMS
	blame_info<<"After DF Children "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
		blame_info<<*set_i_i<<" ";
	}

	blame_info<<endl;
#endif

	for (s_vp_i = ivp->children.begin(); s_vp_i != ivp->children.end(); s_vp_i++) {
	  NodeProps *child = *s_vp_i;
	  if (child->calcAgg == false)
		calcAggregateLNRecursive(child, vStack, vRevisit);
		
	  if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
		blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
    	vRevisit.insert(child);
		vRevisit.insert(ivp);
	  }

	  if (!(child->nStatus[LOCAL_VAR] && child->storesTo.size() > 1) && 
          !(child->nStatus[CALL_NODE]) && !(child->nStatus[LOCAL_VAR] && child->nStatus[CALL_PARAM])) {
		ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
		debugPrintLineNumbers(ivp, child, 2);
	  }
	}
	
#ifdef DEBUG_LINE_NUMS
	blame_info<<"After Children "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif
		
	for (v_vp_i = ivp->dataPtrs.begin(); v_vp_i != ivp->dataPtrs.end(); v_vp_i++) {
	  NodeProps *child = *v_vp_i;
	  if (child->calcAgg == false)
	    calcAggregateLNRecursive(child, vStack, vRevisit);
		
	  if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
		blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
   	    vRevisit.insert(child);
		vRevisit.insert(ivp);
	  }
		
	  if (child->isWritten) {
#ifdef ONLY_FOR_PARAM1
        //added by Hui 04/12/16: for miss added cases like:
        //a = GEP b..; store a, c; then c is a DP of b
        //but if !a->isWritten, then b shouldn't include line#s of c
        set<NodeProps *>::iterator set_np_i;
        bool skipThisDP = false;
        bool existed;
        graph_traits < MyGraphType >::edge_descriptor Edge;
        for (set_np_i = child->storesTo.begin(); set_np_i != child->storesTo.end(); set_np_i++) {
          NodeProps *c_st = (*set_np_i);
          tie(Edge, existed) = edge(c_st->number, ivp->number, G);
          if (existed && get(get(edge_iore, G),Edge)==GEP_BASE_OP && !(c_st->isWritten)) { 
	        skipThisDP = true;
            break;
          }
        }
            
        if (skipThisDP)
          continue;
#endif
                    
        ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
	    debugPrintLineNumbers(ivp, child, 3);
	  }
	}

#ifdef DEBUG_LINE_NUMS
	blame_info<<"After Data Ptrs "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif

#ifdef ONLY_FOR_MINIMD_LINE_FROM_FIELDS2
    for (v_vp_i = ivp->fields.begin(); v_vp_i != ivp->fields.end(); v_vp_i++) {
      NodeProps *child = *v_vp_i;
      if (child->calcAgg == false)
        calcAggregateLNRecursive(child, vStack, vRevisit);
        
      if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
        blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
        vRevisit.insert(child);
        vRevisit.insert(ivp);
      }
        
      if (child->isWritten && child != ivp) {
        ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
        debugPrintLineNumbers(ivp, child, 33);
      }
    }

#ifdef DEBUG_LINE_NUMS
	blame_info<<"After fields "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif
#endif
	// TOCHECK: 11/04/17  SHALL we only addin the earlier aliases(aliasIn)' lines
    for (v_vp_i = ivp->aliases.begin(); v_vp_i != ivp->aliases.end(); v_vp_i++) {
	  NodeProps *child = *v_vp_i;
		 
	  NodeProps *childFieldParent = child->fieldUpPtr;
	  if (childFieldParent != NULL) { //"parm" only used for Fortran
		if (childFieldParent->name.find("parm") != string::npos) {
		  ivp->descLineNumbers.insert(childFieldParent->externCallLineNumbers.begin(), childFieldParent->externCallLineNumbers.end());
#ifdef DEBUG_LINE_NUMS
          blame_info<<"Inserting some line numbers.  Booyah!"<<endl;
#endif									
		}
      }

	  for (v_vp_i2 = child->dataPtrs.begin(); v_vp_i2 != child->dataPtrs.end(); v_vp_i2++) {
		NodeProps *child2 = *v_vp_i2;
		if (child2->calcAgg == false)
		  calcAggregateLNRecursive(child2, vStack, vRevisit);
			
		if (vStack.count(child2)) {
#ifdef DEBUG_LINE_NUMS
          blame_info<<"Conflict in LNRecursive. Need to revisit "<<child2->name<<" and "<<ivp->name<<endl;
#endif
          vRevisit.insert(child2);
		  vRevisit.insert(ivp);
		}
			
		if (child2->isWritten && child2 != ivp) {
	      ivp->descLineNumbers.insert(child2->descLineNumbers.begin(), child2->descLineNumbers.end());
		  debugPrintLineNumbers(ivp, child2, 4);
		}
	  }

#ifdef ONLY_FOR_MINIMD_LINE_FROM_FIELDS
	  for (v_vp_i2 = child->fields.begin(); v_vp_i2 != child->fields.end(); v_vp_i2++) {
		NodeProps *child3 = *v_vp_i2;
		if (child3->calcAgg == false)
	      calcAggregateLNRecursive(child3, vStack, vRevisit);
			
		if (vStack.count(child3)) {
#ifdef DEBUG_LINE_NUMS
		  blame_info<<"Conflict in LNRecursive. Need to revisit "<<child3->name<<" and "<<ivp->name<<endl;
#endif
          vRevisit.insert(child3);
		  vRevisit.insert(ivp);
		}
			
		if (child3->isWritten && child3 != ivp) {
		  ivp->descLineNumbers.insert(child3->descLineNumbers.begin(), child3->descLineNumbers.end());
		  debugPrintLineNumbers(ivp, child3, 44);
		}
	  }
#endif
	}

#ifdef DEBUG_LINE_NUMS
	blame_info<<"After aliases "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}	
	blame_info<<endl;
#endif

	for (v_vp_i = ivp->dfAliases.begin(); v_vp_i != ivp->dfAliases.end(); v_vp_i++) {
	  NodeProps *child = *v_vp_i;
	  if (child->calcAgg == false)
		calcAggregateLNRecursive(child, vStack, vRevisit);
		
	  if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
		blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
		vRevisit.insert(child);
		vRevisit.insert(ivp);
      }
		
	  ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
	  debugPrintLineNumbers(ivp, child, 5);
	}
	
#ifdef DEBUG_LINE_NUMS
	blame_info<<"After DF Aliases "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif

	for (v_vp_i = ivp->resolvedLSFrom.begin(); v_vp_i != ivp->resolvedLSFrom.end(); v_vp_i++) {
	  NodeProps *child = *v_vp_i;
	  if (child->calcAgg == false)
		calcAggregateLNRecursive(child, vStack, vRevisit);
		
	  if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
	    blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
		vRevisit.insert(child);
		vRevisit.insert(ivp);
      }
		
      for (v_vp_i2 = child->resolvedLSSideEffects.begin(); 
	    v_vp_i2 != child->resolvedLSSideEffects.end(); v_vp_i2++) {
		NodeProps *sECause = *v_vp_i2;
		if (sECause->calcAgg == false)
	      calcAggregateLNRecursive(sECause, vStack,  vRevisit);	
#ifdef DEBUG_IMPORTANT_VERTICES
		blame_info<<"ivp - "<<ivp->name<<" adding RLSE lines from "<<sECause->name<<endl;	
#endif 			
		// TODO:  This technically only happens if sECause dominates ivp
		ivp->descLineNumbers.insert(sECause->descLineNumbers.begin(), sECause->descLineNumbers.end());
		debugPrintLineNumbers(ivp, child, 6);
	  }
	}
	
#ifdef DEBUG_LINE_NUMS
	blame_info<<"After Resolved LS From "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
		blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif

#ifdef ONLY_FOR_MINIMD_LINE_FROM_LOADFORCALLS //this can also be retrieved in postmortem phase
    for (v_vp_i = ivp->loadForCalls.begin(); v_vp_i != ivp->loadForCalls.end(); v_vp_i++) {
      NodeProps *child = *v_vp_i;
      if (child->calcAgg == false)
        calcAggregateLNRecursive(child, vStack, vRevisit);
        
      if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
        blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
        vRevisit.insert(child);
        vRevisit.insert(ivp);
      }
        
      if (child->isWritten && child != ivp) {
        ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
        debugPrintLineNumbers(ivp, child, 7);
      }
      else if (ivp->isPid && child != ivp) {
        ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
        debugPrintLineNumbers(ivp, child, 13);
      }
    }
#ifdef DEBUG_LINE_NUMS
	blame_info<<"After loadForCalls "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif
#endif
/* commented out by Hui 11/04/17 for cuda, we don't add in this part for now
#ifdef ONLY_FOR_MINIMD_LINE_FROM_ALIASESOUT
    for (v_vp_i = ivp->aliasesOut.begin(); v_vp_i != ivp->aliasesOut.end(); v_vp_i++) {
      NodeProps *child = *v_vp_i;
      if (child->calcAgg == false)
        calcAggregateLNRecursive(child, vStack, vRevisit);
        
      if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
        blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
        vRevisit.insert(child);
        vRevisit.insert(ivp);
      }
        
      if (child->isWritten && ivp->isWritten && child != ivp) {
        ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
        debugPrintLineNumbers(ivp, child, 8);
      }
    }
#endif

#ifdef DEBUG_LINE_NUMS
	blame_info<<"After aliasesOut "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
		blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif
*/
/* not needed by cuda 11/04/17
#ifdef ADD_MULTI_LOCALE
    if (ivp->isObj) {
      for (v_vp_i = ivp->objAliases.begin(); v_vp_i != ivp->objAliases.end(); v_vp_i++) {
        NodeProps *child = *v_vp_i;
        if (child->calcAgg == false)
            calcAggregateLNRecursive(child, vStack, vRevisit);
        
        if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
            blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
            vRevisit.insert(child);
            vRevisit.insert(ivp);
        }
        
        if (child->isWritten && ivp->isWritten && child != ivp) {
            ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
            debugPrintLineNumbers(ivp, child, 9);
        }
      }
#ifdef DEBUG_LINE_NUMS
	  blame_info<<"After objAliases "<<ivp->name<<endl;
	  for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
		blame_info<<*set_i_i<<" ";
	  }
	  blame_info<<endl;
#endif
    }

    if (ivp->isPid) {
      //first add myObj if it exists
      if (ivp->myObj) {
        NodeProps *child = ivp->myObj;
        if (child->calcAgg == false)
            calcAggregateLNRecursive(child, vStack, vRevisit);
        
        if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
            blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
            vRevisit.insert(child);
            vRevisit.insert(ivp);
        }
        
        if (child->isWritten && ivp->isWritten && child != ivp) {
            ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
            debugPrintLineNumbers(ivp, child, 10);
        }

#ifdef DEBUG_LINE_NUMS
	    blame_info<<"After myObj "<<ivp->name<<endl;
	    for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
		    blame_info<<*set_i_i<<" ";
	    }
	    blame_info<<endl;
#endif
      }
      //Now add pidAliases
      for (v_vp_i = ivp->pidAliasesOut.begin(); v_vp_i != ivp->pidAliasesOut.end(); v_vp_i++) {
        NodeProps *child = *v_vp_i;
        if (child->calcAgg == false)
            calcAggregateLNRecursive(child, vStack, vRevisit);
        
        if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
            blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
            vRevisit.insert(child);
            vRevisit.insert(ivp);
        }
        
        if (child->isWritten && ivp->isWritten && child != ivp) {
            ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
            debugPrintLineNumbers(ivp, child, 11);
        }
      }
#ifdef DEBUG_LINE_NUMS
	  blame_info<<"After pidAliasesOut "<<ivp->name<<endl;
	  for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	    blame_info<<*set_i_i<<" ";
	  }
	  blame_info<<endl;
#endif
    }
    
    //The following is added only if the child is a pid
    for (v_vp_i = ivp->GEPs.begin(); v_vp_i != ivp->GEPs.end(); v_vp_i++) {
      NodeProps *child = *v_vp_i;
      if (child->isPid) { 
        if (child->calcAgg == false)
            calcAggregateLNRecursive(child, vStack, vRevisit);
        
        if (vStack.count(child)) {
        #ifdef DEBUG_LINE_NUMS
            blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
        #endif
            vRevisit.insert(child);
            vRevisit.insert(ivp);
        }
        
        if (child->isWritten && child != ivp) {
            ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
            debugPrintLineNumbers(ivp, child, 12);
        }
      }
    }
#ifdef DEBUG_LINE_NUMS
	blame_info<<"After pid GEPs "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
		blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif
#endif //ADD_MULTI_LOCALE
*/

    //The following for GEP base line# propagation
    for (v_vp_i = ivp->GEPChildren.begin(); v_vp_i != ivp->GEPChildren.end(); v_vp_i++) {
      NodeProps *child = *v_vp_i;
      if (child->calcAgg == false)
        calcAggregateLNRecursive(child, vStack, vRevisit);
        
      if (vStack.count(child)) {
#ifdef DEBUG_LINE_NUMS
        blame_info<<"Conflict in LNRecursive. Need to revisit "<<child->name<<" and "<<ivp->name<<endl;
#endif
        vRevisit.insert(child);
        vRevisit.insert(ivp);
      }
        
      if (child->isWritten && child != ivp) {
        ivp->descLineNumbers.insert(child->descLineNumbers.begin(), child->descLineNumbers.end());
        debugPrintLineNumbers(ivp, child, 13);
      }
    }
#ifdef DEBUG_LINE_NUMS
	blame_info<<"After GEPChildren "<<ivp->name<<endl;
	for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++) {
	  blame_info<<*set_i_i<<" ";
	}
	blame_info<<endl;
#endif

#ifdef DEBUG_LINE_NUMS
	blame_info<<"Exiting calcAggregateLNRecursive for "<<ivp->name<<endl;
#endif
	
	vStack.erase(ivp);
	
}

void FunctionBFC::calcAggregateLN()
{
	set<NodeProps *>::iterator ivh_i;
	set<NodeProps *> vStack;
	set<NodeProps *> vRevisit;

	set<NodeProps *> vStack_call;
	set<NodeProps *> vRevisit_call;
	// Populate Edges
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	  if (ivp->calcAgg == false)
		calcAggregateLNRecursive(ivp, vStack, vRevisit);
		
	  if (ivp->calcAggCall == false)
		calcAggCallRecursive(ivp, vStack_call, vRevisit_call);
	}
	
	// Revisit the ones that had conflicts for calcAggregateLNRecursive
#ifdef DEBUG_LINE_NUMS
	blame_info<<"Revisiting some of the conflicted agg LN values."<<endl;
#endif
	set<NodeProps *>::iterator set_vp_i;
	for (set_vp_i = vRevisit.begin(); set_vp_i != vRevisit.end(); set_vp_i++)
      (*set_vp_i)->calcAgg = false;
		
	for (set_vp_i = vRevisit.begin(); set_vp_i != vRevisit.end(); set_vp_i++)
	  calcAggregateLNRecursive(*set_vp_i, vStack, vRevisit);
		
	// Reverse-visit them again to take care of cases when "early nodes" needs "later" nodes' blame lines		
    set<NodeProps *>::reverse_iterator r_si;
    for (r_si = impVertices.rbegin(); r_si != impVertices.rend(); ++r_si)
	  (*r_si)->calcAgg = false;
		
	for (r_si = impVertices.rbegin(); r_si != impVertices.rend(); ++r_si) {
	  NodeProps *ivp = (*r_si);
	  if (ivp->calcAgg == false)
		calcAggregateLNRecursive(ivp, vStack, vRevisit);
	}

//mainly because of the aliasesOut line# insertion, it'll have to wait for the later nodes
//to finish before the earlier nodes to be complete
#ifdef TEMP_FOR_MINIMD
/*    set<NodeProps *>::reverse_iterator r_si;
    cerr<<"\nReverse order:"<<endl;
    for (r_si = impVertices.rbegin(); r_si != impVertices.rend(); ++r_si)
		cerr<<(*r_si)->name<<" ";
    cerr<<"\nNormal order:"<<endl;
		
	// Revisit them all to take care of scragglers		
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++)
		cerr<<(*ivh_i)->name<<" ";
    cerr<<endl;
*/
		// Revisit them all to take care of scragglers		
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++)
      (*ivh_i)->calcAgg = false;
		
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	  if (ivp->calcAgg == false)
		calcAggregateLNRecursive(ivp, vStack, vRevisit);
	}

		// Revisit them all to take care of scragglers		
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++)
	  (*ivh_i)->calcAgg = false;
		
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	  if (ivp->calcAgg == false)
		calcAggregateLNRecursive(ivp, vStack, vRevisit);
	}
/*
	// Revisit them all to take care of scragglers		
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++)
		(*ivh_i)->calcAgg = false;
		
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
		NodeProps *ivp = (*ivh_i);
		if (ivp->calcAgg == false)
			calcAggregateLNRecursive(ivp, vStack, vRevisit);
	}
*/

#endif
	// Revisit the ones that had conflicts for calcAggCallRecursive, added by Hui
#ifdef DEBUG_LINE_NUMS
	blame_info<<"Revisiting some of the conflicted agg Call values."<<endl;
#endif
	for (set_vp_i = vRevisit_call.begin(); set_vp_i != vRevisit_call.end(); set_vp_i++)
   	  (*set_vp_i)->calcAggCall = false;
		
	for (set_vp_i = vRevisit_call.begin(); set_vp_i != vRevisit_call.end(); set_vp_i++)
	  calcAggCallRecursive(*set_vp_i, vStack_call, vRevisit_call);
		
	// Revisit them all to take care of scragglers		
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++)
	  (*ivh_i)->calcAggCall = false;
		
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	  if (ivp->calcAggCall == false)
		calcAggCallRecursive(ivp, vStack_call, vRevisit_call);
	}
}

//not used anymore
void FunctionBFC::trimLocalVarPointers()
{
	set<NodeProps *>::iterator ivh_i;
	
	
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++)
	{
		NodeProps * ivp = (*ivh_i);
		//int sizeParents;
		
		
		
		// if it's a data pointer and no parents access it then we fold it into
		//   it's container
		if (( ivp->nStatus[LOCAL_VAR_PTR] || ivp->nStatus[EXIT_VAR_PTR] )
				&& (ivp->isWritten == false && ivp->children.size() == 0) &&
				!ivp->nStatus[CALL_PARAM]  && !ivp->nStatus[CALL_RETURN])
		{
			
			//NodeProps * ivpPT = ivp->pointsTo;;
			
			//if (ivpPT == NULL)
			//continue;
			
			//if (ivpPT->arrayAccess.size() == 0)
			//continue;		
			
#ifdef DEBUG_IMPORTANT_VERTICES		
			blame_info<<"Trimming "<<ivp->name<<endl;
#endif
			ivp->isExported = false;
			impVertCount--;
			
			//if (ivp->pointsTo == NULL)
			//{
			//cerr<<"Why does data ptr have NULL points to?"<<endl;
			//continue;
			//	}
			
			
			set<NodeProps *>::iterator set_vp_i;
			set<int>::iterator set_int_i;
			
#ifdef DEBUG_IMPORTANT_VERTICES		
			blame_info<<"DF up ptr is "<<ivp->dpUpPtr->name<<endl;
#endif
#ifdef DEBUG_LINE_NUMS
			blame_info<<"Inserting line number(10) "<<ivp->line_num<<" to "<<ivp->dpUpPtr->name<<endl;
			#endif
			
			ivp->dpUpPtr->lineNumbers.insert(ivp->lineNumbers.begin(), ivp->lineNumbers.end());
			ivp->dpUpPtr->lineNumbers.insert(ivp->line_num);
			
			
		}
	}
}

int FunctionBFC::checkCompleteness()
{
	set<NodeProps *>::iterator ivh_i;
	
	set<int> firstCheck;
	set<int> secondCheck;
	
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
		NodeProps *ivp = (*ivh_i);
		firstCheck.insert(ivp->descLineNumbers.begin(), ivp->descLineNumbers.end());
		firstCheck.insert(ivp->line_num);
		
		if (ivp->eStatus > NO_EXIT || ivp->nStatus[EXIT_VAR_FIELD]) {
			secondCheck.insert(ivp->descLineNumbers.begin(), ivp->descLineNumbers.end());
			secondCheck.insert(ivp->line_num);	
			
			set<NodeProps *>::iterator s_vp_i;
			for (s_vp_i = ivp->descParams.begin(); s_vp_i != ivp->descParams.end(); s_vp_i++) {
				NodeProps *param = *s_vp_i;
#ifdef DEBUG_COMPLETENESS			
				blame_info<<"Param(1) "<<param->name<<" from IVP "<<ivp->name<<endl;
#endif			
				set<ImpFuncCall *>::iterator v_ifc_i;
				for (v_ifc_i = param->calls.begin(); v_ifc_i != param->calls.end(); v_ifc_i++) {
					ImpFuncCall *ifc = *v_ifc_i;
					NodeProps *cNode = ifc->callNode;
#ifdef DEBUG_COMPLETENESS								
					blame_info<<"Call Node(1) "<<cNode->name<<" from Param "<<param->name<<endl;
#endif				
					secondCheck.insert(cNode->line_num);
					set<ImpFuncCall *>::iterator v_ifc_i2;
					
					for (v_ifc_i2 = cNode->calls.begin(); v_ifc_i2 != cNode->calls.end(); v_ifc_i2++) {
						ImpFuncCall *ifc2 = *v_ifc_i2;
						NodeProps *cNode2 = ifc2->callNode;
#ifdef DEBUG_COMPLETENESS									
						blame_info<<"Param Node(2) "<<cNode2->name<<" from Call Node "<<cNode->name<<endl;
#endif
						secondCheck.insert(cNode2->line_num);
						secondCheck.insert(cNode2->descLineNumbers.begin(), cNode2->descLineNumbers.end());
					}						
				}
			}
		}
		// We don't care what line the variable was defined on for line num totals
		if (ivp->nStatus[LOCAL_VAR])
			secondCheck.insert(ivp->line_num);
		
		if (isBFCPoint && (ivp->nStatus[LOCAL_VAR] || ivp->nStatus[LOCAL_VAR_FIELD])) {
			secondCheck.insert(ivp->descLineNumbers.begin(), ivp->descLineNumbers.end());
			secondCheck.insert(ivp->line_num);	
			
			set<NodeProps *>::iterator s_vp_i;
			for(s_vp_i = ivp->descParams.begin(); s_vp_i != ivp->descParams.end(); s_vp_i++) {
				NodeProps *param = *s_vp_i;
#ifdef DEBUG_COMPLETENESS													
				blame_info<<"Param(2) "<<param->name<<" from IVP "<<ivp->name<<endl;
#endif			
				set<ImpFuncCall *>::iterator v_ifc_i;
				
				for (v_ifc_i = param->calls.begin(); v_ifc_i != param->calls.end(); v_ifc_i++) {
					ImpFuncCall *ifc = *v_ifc_i;
					NodeProps *cNode = ifc->callNode;
#ifdef DEBUG_COMPLETENESS									
					blame_info<<"Call Node(2) "<<cNode->name<<" from Param "<<param->name<<endl;
#endif 
					secondCheck.insert(cNode->line_num);
				}
			}
		}
	}
	
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
		NodeProps *ivp = (*ivh_i);
		if (ivp->nStatus[CALL_NODE]) {
			secondCheck.insert(ivp->line_num);
			secondCheck.insert(ivp->descLineNumbers.begin(), ivp->descLineNumbers.end());
			set<ImpFuncCall *>::iterator v_ifc_i;
			
			for (v_ifc_i = ivp->calls.begin(); v_ifc_i != ivp->calls.end(); v_ifc_i++) {
				ImpFuncCall *ifc = *v_ifc_i;
				NodeProps *cNode = ifc->callNode;
#ifdef DEBUG_COMPLETENESS									
				blame_info<<"Param Node(1) "<<cNode->name<<" from Call Node "<<ivp->name<<endl;
#endif
				secondCheck.insert(cNode->line_num);
				secondCheck.insert(cNode->descLineNumbers.begin(), cNode->descLineNumbers.end());
			}
		}
	}
	
	firstCheck.erase(0);
	secondCheck.erase(0);
	
	int isMissing = 0;
#ifdef DEBUG_COMPLETENESS									
	blame_info<<"All line Nums"<<endl;
	set<int>::iterator s_i_i;
	
	for (s_i_i = allLineNums.begin(); s_i_i != allLineNums.end(); s_i_i++)
		blame_info<<*s_i_i<<" ";
	blame_info<<endl;
	
	blame_info<<"First Check"<<endl;
	for (s_i_i = firstCheck.begin(); s_i_i != firstCheck.end(); s_i_i++)
		blame_info<<*s_i_i<<" ";
	blame_info<<endl;
	
	blame_info<<"Second Check"<<endl;
	for (s_i_i = secondCheck.begin(); s_i_i != secondCheck.end(); s_i_i++)
		blame_info<<*s_i_i<<" ";
	blame_info<<endl;
	
	if (firstCheck == allLineNums)
		blame_info<<"FC_All line numbers accounted for "<<endl;
	else
		blame_info<<"FC_MISSING a line number"<<endl;
#endif	
	if (secondCheck == firstCheck) {
#ifdef DEBUG_COMPLETENESS									
		blame_info<<"SC_All line numbers accounted for "<<endl;
#endif
	}
	else {
		isMissing = 1;
#ifdef DEBUG_SUMMARY_CC									
		cout<<"SC_MISSING a line number"<<endl;
		blame_info<<"SC_MISSING a line number"<<endl;
#endif
	}	
	
#ifdef DEBUG_SUMMARY_CC
	set<int> setDiff;
	if (isMissing) {
		cout<<"Set Diff is "<<endl;
		set_difference(firstCheck.begin(), firstCheck.end(),//It's a std template func
	    secondCheck.begin(), secondCheck.end(), ostream_iterator<int>(cout, " "));
		
		cout<<endl<<endl;		
	}
#endif
	return isMissing;
}


void FunctionBFC::moreThanOneEV(int &numMultipleEV, int &afterOp1, int &afterOp2)
{
	vector<ExitVariable *>::iterator v_ev_i;
	
	int numEV = 0;
	int numEV2 = 0;
	int numEV3 = 0;
	
	for (v_ev_i = exitVariables.begin(); v_ev_i != exitVariables.end(); v_ev_i++) {
		ExitVariable *ev = *v_ev_i;
		if (ev->whichParam >= 0 && ev->vertex != NULL ) { //changed by Hui from >0 to >=0
			numEV++;
			if (ev->vertex->descLineNumbers.size() > 1) {
				numEV2++;
				numEV3++;
			}
			else if (ev->vertex->aliasParams.size() > 1) {
				numEV3++;
			}
		}
	}
	
	if(numEV > 1)
		numMultipleEV++;
	
	if (numEV2 > 1)
		afterOp1++;
	
	if (numEV3 > 1)
		afterOp2++;
}


NodeProps *FunctionBFC::resolveSideEffectsCheckParentEV(NodeProps *vp, set<NodeProps *> &visited)
{
#ifdef DEBUG_SIDE_EFFECTS
	blame_info<<"In resolveSideEffectsCheckParentEV for "<<vp->name<<endl;
#endif
	if (visited.count(vp) > 0)
		return NULL;
	
	visited.insert(vp);
	
	if (vp->eStatus >= EXIT_VAR_PARAM || vp->nStatus[EXIT_VAR_FIELD]) //changed by Hui 03/15/16
		return vp;                                                  //from >EXIT_VAR_PARAM to >=...
	
	if (vp->dpUpPtr != NULL || vp->dpUpPtr != vp) //changed by Hui 08/11/16: condition from || to &&
		return resolveSideEffectsCheckParentEV(vp->dpUpPtr, visited);
	
	if (vp->dfaUpPtr != NULL || vp->dfaUpPtr != vp) //changed by Hui 08/11/16: condition from || to &&
		return resolveSideEffectsCheckParentEV(vp->dfaUpPtr, visited);
	
	return NULL;
}


NodeProps * FunctionBFC::resolveSideEffectsCheckParentLV(NodeProps * vp, set<NodeProps *> & visited)
{
#ifdef DEBUG_SIDE_EFFECTS
	blame_info<<"In resolveSideEffectsCheckParentLV for "<<vp->name<<" "<<vp->name<<endl;
#endif
	if (visited.count(vp) > 0)
		return NULL;
	
	visited.insert(vp);
	
	if (vp->nStatus[LOCAL_VAR] || vp->nStatus[LOCAL_VAR_FIELD])
		return vp;
	
	if (vp->dpUpPtr != NULL || vp->dpUpPtr != vp)
		return resolveSideEffectsCheckParentLV(vp->dpUpPtr, visited);
	
	if (vp->dfaUpPtr != NULL || vp->dfaUpPtr != vp)
		return resolveSideEffectsCheckParentLV(vp->dfaUpPtr, visited);
	
	return NULL;
}



// TODO: We need to cover side effect relations where the EVs that are "sucked in" are 
// written to as well
void FunctionBFC::resolveSideEffectsHelper(NodeProps *rootVP, NodeProps *vp, set<NodeProps *> &visited)
{
#ifdef DEBUG_SIDE_EFFECTS
	blame_info<<"Entering resolveSideEffectsHelper for "<<rootVP->name<<" "<<vp->name<<endl;
#endif
	
	if (visited.count(vp) > 0) {
#ifdef DEBUG_SIDE_EFFECTS	
		blame_info<<"Exiting(V) resolveSideEffectsHelper for "<<rootVP->name<<" "<<vp->name<<endl;
#endif		
		return;
	}
	
	visited.insert(vp);
	set<NodeProps *>::iterator set_ivp_i;
	if (vp->suckedInEVs.size() > 0) {
#ifdef DEBUG_SIDE_EFFECTS		
		blame_info<<"Sucked in EV match(0) for "<<rootVP->name<<" and "<<vp->name<<endl;
#endif			
		set<NodeProps *>::iterator set_ivp_i2;
		for (set_ivp_i2 = vp->suckedInEVs.begin(); set_ivp_i2 != vp->suckedInEVs.end();
				 set_ivp_i2++) {
#ifdef DEBUG_SIDE_EFFECTS
			blame_info<<"Match(0) between "<<rootVP->name<<" and "<<(*set_ivp_i2)->name<<endl;
#endif
			addSERelation(rootVP, (*set_ivp_i2));	
		}
	}		
	
	for (set_ivp_i = vp->dataPtrs.begin(); set_ivp_i != vp->dataPtrs.end(); set_ivp_i++) {
#ifdef DEBUG_SIDE_EFFECTS
		blame_info<<"rSEH - DP "<<(*set_ivp_i)->name<<" for "<<vp->name<<endl;
#endif
		if ((*set_ivp_i)->isWritten)
			resolveSideEffectsHelper(rootVP, *set_ivp_i, visited);
	}	
	
	for (set_ivp_i = vp->children.begin(); set_ivp_i != vp->children.end(); set_ivp_i++) {
#ifdef DEBUG_SIDE_EFFECTS
		blame_info<<"rSEH - Child "<<(*set_ivp_i)->name<<" for "<<vp->name<<endl;
#endif
		if ((*set_ivp_i)->suckedInEVs.size() > 0) {
#ifdef DEBUG_SIDE_EFFECTS
			blame_info<<"Sucked in EV match(1) for "<<rootVP->name<<" and "<<(*set_ivp_i)->name<<endl;
#endif
			set<NodeProps *>::iterator set_ivp_i2;
			for (set_ivp_i2 = (*set_ivp_i)->suckedInEVs.begin(); set_ivp_i2 != (*set_ivp_i)->suckedInEVs.end();
					 set_ivp_i2++) {
#ifdef DEBUG_SIDE_EFFECTS
				blame_info<<"Match(1) between "<<rootVP->name<<" and "<<(*set_ivp_i2)->name<<endl;
#endif
				addSERelation(rootVP, (*set_ivp_i2));
			}
		}		
		//recursively call itself
		resolveSideEffectsHelper(rootVP, *set_ivp_i, visited);
	
		set<NodeProps *> visited2;
		NodeProps *match = resolveSideEffectsCheckParentEV(*set_ivp_i, visited2);
		
		if (match == NULL)
			continue;
		
		if (match != NULL && match != rootVP) {
#ifdef DEBUG_SIDE_EFFECTS
			blame_info<<"Match(2) between "<<rootVP->name<<" and "<<match->name<<endl;
#endif
			addSERelation(rootVP, match);
		}
	} 
	
	for (set_ivp_i = vp->dfChildren.begin(); set_ivp_i != vp->dfChildren.end(); set_ivp_i++) {
#ifdef DEBUG_SIDE_EFFECTS
		blame_info<<"rSEH - dfChild "<<(*set_ivp_i)->name<<" for "<<vp->name<<endl;
#endif
		
		if ((*set_ivp_i)->suckedInEVs.size() > 0) {
#ifdef DEBUG_SIDE_EFFECTS
			blame_info<<"Sucked in EV match (3) for "<<rootVP->name<<" and "<<(*set_ivp_i)->name<<endl;
#endif
			set<NodeProps *>::iterator set_ivp_i2;
			for (set_ivp_i2 = (*set_ivp_i)->suckedInEVs.begin(); set_ivp_i2 != (*set_ivp_i)->suckedInEVs.end();
					 set_ivp_i2++) {
#ifdef DEBUG_SIDE_EFFECTS
				blame_info<<"Match(3) between "<<rootVP->name<<" and "<<(*set_ivp_i2)->name<<endl;
#endif
				addSERelation(rootVP, (*set_ivp_i2));
			}
		}		
		
		resolveSideEffectsHelper(rootVP, *set_ivp_i, visited);
		
		set<NodeProps *> visited2;
		NodeProps *match = resolveSideEffectsCheckParentEV(*set_ivp_i, visited2);//TC
		
		if (match == NULL)
			continue;
		
		if (match != NULL && rootVP != match) {
#ifdef DEBUG_SIDE_EFFECTS
			blame_info<<"Match(4) between "<<rootVP->name<<" and "<<match->name<<endl;
#endif
			addSERelation(rootVP, match);
		}
	}
#ifdef DEBUG_SIDE_EFFECTS
	blame_info<<"Exiting resolveSideEffectsHelper for "<<rootVP->name<<" "<<vp->name<<endl;
#endif
}

void FunctionBFC::addSEAlias(NodeProps *source, NodeProps *target)
{
	vector< pair<NodeProps *, NodeProps *> >::iterator vec_pair_i;
	
	if (source == target)
		return;
	
	if (source->getFullName().compare(target->getFullName()) == 0)
		return;
	
	for (vec_pair_i = seAliases.begin(); vec_pair_i != seAliases.end(); vec_pair_i++) {
		NodeProps *fir = (*vec_pair_i).first;
		NodeProps *sec = (*vec_pair_i).second;
		
		if (fir == source && sec == target)
			return;
		
		if (fir->getFullName().compare(source->getFullName()) == 0 &&
				sec->getFullName().compare(target->getFullName()) == 0)
			return;
		
	}
	pair<NodeProps *, NodeProps *> newOne(source, target);
	seAliases.push_back(newOne);
}


void NodeProps::getStructName(string & structName, set<NodeProps *> & visited)
{
	//cout<<"Entering getStructName"<<endl;
	//cout<<"Here "<<structName<<endl;
	
	if (visited.count(this) > 0)
	{
		structName.insert(0, name);
		return;
	}
	
	visited.insert(this);
	
	if (fieldUpPtr == NULL)
	{
		//cout<<"Here 2 "<<structName<<endl;
		structName.insert(0, name);
		//structName.insert(0,".");
		return;
	}
	else
	{
		if (sField == NULL)
		{
			//cout<<"Here 3 "<<structName<<endl;
			structName.insert(0, name);
			structName.insert(0,"NFP_");
			structName.insert(0,".");
			
		}
		else
		{
			//cout<<"Here 4 "<<structName<<endl;
			structName.insert(0,sField->fieldName);
			structName.insert(0,".");
			
		}				
		//cout<<"Here 5 "<<structName<<endl;
		fieldUpPtr->getStructName(structName, visited); 
	}
	//cout<<"Exiting getStructName"<<endl;
}



string & NodeProps::getFullName()
{
	//cout<<"Entering getStructName"<<endl;
	//cout<<"Here "<<structName<<endl;
	
	if (calcName) {
		//cout<<"Exiting(1) getFullName"<<endl;
		return fullName;
	}
	
	set<NodeProps *> visited;
	
	if (nStatus[EXIT_VAR_FIELD])
		getStructName(fullName, visited);
	else
		fullName = name;
	
	calcName = true;
	//cout<<"Exiting getFullName"<<endl;
	return fullName;
}

int NodeProps::getParamNum(set<NodeProps *> & visited)
{
	//cout<<"Entering getStructName"<<endl;
	//cout<<"Here "<<structName<<endl;
	
	if (paramNum)
		return paramNum;
	
	if (visited.count(this) > 0) {
		return 0;
	}
	
	visited.insert(this);
	
	if (fieldUpPtr != NULL) {
		paramNum = fieldUpPtr->getParamNum(visited);
		return paramNum;
	}
	
	return 0;
}

void FunctionBFC::addSERelation(NodeProps *source, NodeProps *target)
{
	//blame_info<<"Entering addSERelation for "<<source->name<<" to "<<target->name<<endl;
	//cout<<"Entering addSERelation for "<<source->name<<" to "<<target->name<<endl;
	
	vector< pair<NodeProps *, NodeProps *> >::iterator vec_pair_i;
	
	//blame_info<<"asr(1)"<<endl;
	//cout<<"asr(1) "<<getSourceFuncName()<<endl;
	
	if (source == target)
		return;
	
	if (source->getFullName().compare(target->getFullName()) == 0)
		return;
	
	//blame_info<<"asr(2)"<<endl;
	//cout<<"asr(2) "<<getSourceFuncName()<<endl;
	for (vec_pair_i = seRelations.begin(); vec_pair_i != seRelations.end(); vec_pair_i++) {
		//blame_info<<"asr(3)"<<endl;
		//cout<<"asr(3) "<<getSourceFuncName()<<endl;
		NodeProps *fir = (*vec_pair_i).first;
		NodeProps *sec = (*vec_pair_i).second;
		
		if (fir == source && sec == target)
			return;
		
		//blame_info<<"asr(4)"<<endl;
		//cout<<"asr(4) "<<getSourceFuncName()<<endl;
		
		if (fir->getFullName().compare(source->getFullName()) == 0 &&
				sec->getFullName().compare(target->getFullName()) == 0)
			return;
		//blame_info<<"asr(5)"<<endl;
		//cout<<"asr(5) "<<getSourceFuncName()<<endl;
	}
	
	pair<NodeProps *, NodeProps *> newOne(source, target);
	seRelations.push_back(newOne);
	//blame_info<<"Exiting addSERelation for "<<source->name<<" to "<<target->name<<endl;
}


// We go until we hit a field or an exit variable
void FunctionBFC::recursiveSEAliasHelper(set<NodeProps *> &visited, NodeProps *orig, NodeProps *target)
{
	if (target == NULL)
		return;

	//blame_info<<"Examining rSEAH for "<<orig->name<<" and "<<target->name<<endl;
	if (visited.count(target))
		return;
	
	visited.insert(target);

	set<NodeProps*>::iterator set_vp_i;
	
	if (target->eStatus >= EXIT_VAR_PARAM || target->nStatus[EXIT_VAR_FIELD]) {
	#ifdef DEBUG_SIDE_EFFECTS
		blame_info<<"Adding alias(19) between "<<orig->name<<" and "<<target->name<<endl;
	#endif 
		addSEAlias(orig, target);
		return;
	}	
	
	for (set_vp_i = target->aliasesIn.begin(); set_vp_i != target->aliasesIn.end(); set_vp_i++) {
		recursiveSEAliasHelper(visited, orig, *set_vp_i);
	}
	
	for (set_vp_i = target->aliasesOut.begin(); set_vp_i != target->aliasesOut.end(); set_vp_i++) {
		recursiveSEAliasHelper(visited, orig, *set_vp_i);
	}
	recursiveSEAliasHelper(visited, orig, target->dfaUpPtr);
}

void FunctionBFC::resolveSideEffects()
{
#ifdef DEBUG_SIDE_EFFECTS
	blame_info<<"In resolveSideEffects "<<endl;
#endif
	set<NodeProps *>::iterator ivh_i;
	
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	  //clearAfter.clear();
	  if (ivp->eStatus >= EXIT_VAR_PARAM || ivp->nStatus[EXIT_VAR_FIELD] || ivp->nStatus[EXIT_VAR_PTR]) {
#ifdef DEBUG_SIDE_EFFECTS
		blame_info<<"Creating SE Info for "<<ivp->name<<endl;
#endif 
    	set<NodeProps *>::iterator vec_vp_i;
		set<NodeProps *> visited;
		for (vec_vp_i = ivp->dfAliases.begin(); vec_vp_i != ivp->dfAliases.end(); vec_vp_i++) {
	      resolveSideEffectsHelper(ivp, *vec_vp_i, visited);
		  if ((*vec_vp_i)->eStatus >= EXIT_VAR_PARAM || (*vec_vp_i)->nStatus[EXIT_VAR_FIELD]) {//changed by Hui 03/15/16
#ifdef DEBUG_SIDE_EFFECTS		                                                                    //from >EXIT_VAR_PARAM to >=...
			blame_info<<"Match(DFA) between "<<ivp->name<<" and "<<(*vec_vp_i)->name<<endl;
#endif
			addSEAlias(ivp, (*vec_vp_i));
		  }
		  else {
#ifdef DEBUG_SIDE_EFFECTS
			blame_info<<"Match(DFA)(N/A) between "<<ivp->name<<" and "<<(*vec_vp_i)->name<<endl;			 
#endif
		  }
		}
			
		for (vec_vp_i = ivp->aliases.begin(); vec_vp_i != ivp->aliases.end(); vec_vp_i++) {
		  if ((*vec_vp_i)->eStatus >= EXIT_VAR_PARAM || (*vec_vp_i)->nStatus[EXIT_VAR_FIELD]  ||(*vec_vp_i)->nStatus[EXIT_VAR_PTR]) {
#ifdef DEBUG_SIDE_EFFECTS                                                               //changed by Hui 03/15/16 from >EXIT_VAR_PARAM to >=...
			blame_info<<"Match(A) between "<<ivp->name<<" and "<<(*vec_vp_i)->name<<endl;
#endif
			addSEAlias(ivp, (*vec_vp_i));
		  }
		  else {
#ifdef DEBUG_SIDE_EFFECTS
			blame_info<<"Match(A)(N/A) between "<<ivp->name<<" and "<<(*vec_vp_i)->name<<endl;
			blame_info<<"Node Props for matched node "<<(*vec_vp_i)->name<<": ";
			for (int a = 0; a < NODE_PROPS_SIZE; a++)
		      blame_info<<(*vec_vp_i)->nStatus[a]<<" ";
			blame_info<<endl;
#endif
            if ((*vec_vp_i)->nStatus[EXIT_VAR_FIELD_ALIAS] || (*vec_vp_i)->nStatus[EXIT_VAR_ALIAS]) {
			  set<NodeProps *> visited;
			  visited.insert(ivp);
			  recursiveSEAliasHelper(visited, ivp, (*vec_vp_i));
			}
		  }
		}
			
		for (vec_vp_i = ivp->dataPtrs.begin(); vec_vp_i != ivp->dataPtrs.end(); vec_vp_i++) {
		  resolveSideEffectsHelper(ivp, *vec_vp_i, visited);
		}	
	  }
	}
}



void FunctionBFC::resolveSideEffectCalls()
{
#ifdef DEBUG_SIDE_EFFECTS
	blame_info<<"In resolveSideEffectCalls "<<endl;
#endif
	
	set<NodeProps *>::iterator ivh_i;
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	  if (ivp->nStatus[CALL_NODE]) {
		set<ImpFuncCall *>::iterator v_ifc_i;
	
		set<NodeProps *> parameters;
		set<NodeProps *> LVparameters;
			
		FuncCallSEElement *fcseElement = NULL;
		FuncCallSE *fcse = new FuncCallSE();
		fcse->callNode = ivp;
			
		for (v_ifc_i = ivp->calls.begin(); v_ifc_i != ivp->calls.end(); v_ifc_i++) {
	      ImpFuncCall *ifc = *v_ifc_i;
		  NodeProps *cNode = ifc->callNode;
		  set<NodeProps *> visited2;
		  NodeProps *match = resolveSideEffectsCheckParentEV(cNode, visited2);	
				
		  if (match != NULL) {
			pair<set<NodeProps *>::iterator,bool> ret;
			ret = parameters.insert(match);
					
			if (ret.second== true) {
		      fcseElement = new FuncCallSEElement();
			  fcseElement->ifc = ifc;
			  fcseElement->paramEVNode = match;
			  fcse->parameters.push_back(fcseElement);
			}
		  }
		  else {
			visited2.clear();
		    match = resolveSideEffectsCheckParentLV(cNode, visited2);
			if (match != NULL) {
			  pair<set<NodeProps *>::iterator,bool> ret;
			  ret = LVparameters.insert(match);
			  if (ret.second== true) {
				fcseElement = new FuncCallSEElement();
			    fcseElement->ifc = ifc;
				fcseElement->paramEVNode = match;
				fcse->NEVparams.push_back(fcseElement);
			  }
			}					
		  }
		}
			
		if (parameters.size() > 1) {
#ifdef DEBUG_SIDE_EFFECTS
		  blame_info<<"Call has more than 1 EVs as parameters :"<<endl;
		  set<NodeProps *>::iterator set_vp_i;
		  for (set_vp_i = parameters.begin(); set_vp_i != parameters.end(); set_vp_i++) {
			blame_info<<(*set_vp_i)->name<<endl;
		  }	
#endif
	      seCalls.push_back(fcse);
		}
		else {
		  if (fcseElement != NULL)
		    delete fcseElement;
		  delete fcse;
		}
	  }
	}
}


void FunctionBFC::resolveCallsDomLines()
{
    set<NodeProps *>::iterator ivh_i;
    for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	  //clearAfter.clear();
	  if (ivp->nStatus[CALL_NODE]) {
        cfg->setDomLines(ivp);
	  }
    }
}


void FunctionBFC::resolveLooseStructs()
{
	set<NodeProps *>::iterator ivh_i;
	
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++) {
	  NodeProps *ivp = (*ivh_i);
	
#ifdef DEBUG_STRUCTS
  	  blame_info<<"Looking at IVP "<<ivp->name<<" in resolveLooseStructs. "<<endl;
#endif 			
	  //clearAfter.clear();
	  if (ivp->llvm_inst != NULL) {
		if (isa<Instruction>(ivp->llvm_inst)) {
	      const llvm::Type *origT = 0;		
		  Instruction *pi = cast<Instruction>(ivp->llvm_inst);	
		  origT = pi->getType();	
		  //newPtrLevel = pointerLevel(origT,0);
		
		  string origTStr = returnTypeName(origT, string(""));
#ifdef DEBUG_STRUCTS
          blame_info<<"Type name (resolveLooseStructs) "<<origTStr<<endl;
#endif 
    	  if (origTStr.find("Struct") != string::npos) {
#ifdef DEBUG_STRUCTS
			blame_info<<"ivp->sBFC (resolveLooseStructs)"<<ivp->sBFC<<endl;
#endif 
            if (ivp->sBFC == NULL) {
#ifdef DEBUG_STRUCTS
			  blame_info<<"Struct "<<ivp->name<<" has no sBFC"<<endl;
#endif
			  Value *v = cast<Value>(ivp->llvm_inst);	
			  const llvm::Type *pointT = v->getType();
			  unsigned typeVal = pointT->getTypeID();
#ifdef DEBUG_STRUCTS
              blame_info<<"Before while, typeVal="<<typeVal<<endl;
#endif
	 		  while (typeVal == Type::PointerTyID) {		
			    pointT = cast<PointerType>(pointT)->getElementType();
			    //string origTStr = returnTypeName(pointT, string(" "));
			    typeVal = pointT->getTypeID();
              }
#ifdef DEBUG_STRUCTS
              blame_info<<"After while, typeVal="<<typeVal<<endl;
#endif
			  if (typeVal == Type::StructTyID) {
			    const llvm::StructType *type = cast<StructType>(pointT);
			    if (!type->isLiteral()) {//literal structs do not have names	
                  string structNameFull = type->getName().str();
#ifdef DEBUG_STRUCTS
	              blame_info<<"structNameFull -- "<<structNameFull<<endl;
#endif
	              if (structNameFull.find("struct.") == 0) { 
	                // need to get rid of preceding "struct." and trailing NULL character
	                structNameFull = structNameFull.substr(7, structNameFull.length()-7);
	              }

                  StructBFC *sb = mb->structLookUp(structNameFull);
     			  if (sb == NULL) {
#ifdef DEBUG_STRUCTS
	                blame_info<<"SB is NULL for "<<structNameFull<<" for IVP "<<ivp->name<<endl;
#endif
				    continue;
				  }
#ifdef DEBUG_STRUCTS
				  blame_info<<"Found sb for "<<structNameFull<<" assiging sBFC to "<<ivp->name<<endl;
#endif
				  ivp->sBFC = sb;
                }
			  }
            } //struct no sBFC
           
            else {
#ifdef DEBUG_STRUCTS
			  blame_info<<"Struct  "<<ivp->name<<" already has sBFC "<<endl;
#endif
		    }
		  } //struct
	    } //is inst
	  } //has llvm_inst
	} // for all impVp	
}

//This func is Moved here from FunctionBFCGraph.cpp
int FunctionBFC::pointerLevel(const llvm::Type *t, int counter)
{
	unsigned typeVal = t->getTypeID();
	if (typeVal == Type::PointerTyID)
		return pointerLevel(cast<PointerType>(t)->getElementType(), counter +1);
	else
		return counter;
}

void FunctionBFC::setModuleAndPathNames(string file, string path)
{
	moduleName = file;
    modulePathName = path;
    moduleSet = true;
}

// Constructor for function blame //
FunctionBFC::FunctionBFC(Function * F, FuncSigHash &kFI)
{
    func = F;
    //funcT = V_PARAM_V_RET;

	voidReturn = false;
	numPointerParams = 0;
	numParams = 0;
	isBFCPoint = false;
	isExternFunc = false; //distinguish user func and extern func
    moduleSet = false;
	//nStatus = UNKNOWN_CALL;
    //fullyEvaluated = false;
	
	exitOutput = new ExitOutput();
	
	//outputFunc = false;
	//callCount = invokeCount = 0;
	
    startLineNum = 9999999;
    endLineNum =   0;
	
	knownFuncsInfo = kFI;
	//cfg has a member points to the FunctionBFC it builds from
	cfg = new FunctionBFCCFG(this);
	
}


// Tweak blamedArgs for this function, may add-in more in the future
void FunctionBFC::tweakBlamedArgs()
{
    string fname = getSourceFuncName();
    if (fname.find("chpl__autoCopy") != string::npos)
      blamedArgs.clear();
    else if (fname.find("chpl__autoDestroy") != string::npos) {
      blamedArgs.clear();
      blamedArgs.insert(0);
    }
    /* wrap* funcs will never be called explicitly(triggered implicitly with fid),
    // on/coforall/cobegin can be called outside wrap*, so we should keep their blamed args
    else if (fname.find("_local_wrapon_fn") == 0 || fname.find("_local_on_fn") == 0) 
      blamedArgs.clear();
    else if (fname.find("wrapon_fn") == 0 || fname.find("on_fn") == 0)
      blamedArgs.clear();
    else if (fname.find("wrapcoforall_fn") == 0 || fname.find("coforall_fn") == 0)
      blamedArgs.clear();
    else if (fname.find("wrapcobegin_fn") == 0 || fname.find("cobegin_fn") == 0)
      blamedArgs.clear();
    */
    else if (fname.find("chpl_executeOn") == 0 || fname.find("chpl_executeOnFast") == 0
          || fname.find("chpl_executeOnNB") == 0 || fname.find("chpl_taskListAddBegin") == 0
          || fname.find("chpl_taskListAddCoStmt") == 0 || fname.find("chpl_taskListProcess") == 0
          || fname.find("chpl_taskListExecute") == 0 || fname.find("chpl_taskListFree") == 0)
      blamedArgs.clear();
    
    else if (fname.find("chpl_gen_comm_get") == 0) {
      blamedArgs.clear();
      blamedArgs.insert(0);
    }
    else if (fname.find("chpl_gen_comm_put") == 0) {
      blamedArgs.clear();
      blamedArgs.insert(2);
    }
    else if (fname.find("chpl_wide_ptr_get_address") == 0) {
      blamedArgs.clear();
      blamedArgs.insert(0);
    }
    else if (fname.find("chpl_wide_ptr_get_node") == 0) {
      blamedArgs.clear();
      blamedArgs.insert(0);
    }
    else if (fname.find("accessHelper") == 0) {
      blamedArgs.clear();
      blamedArgs.insert(1);
    }
    else if (fname.find("chpl__dynamicFastFollowCheck") == 0) {
      blamedArgs.clear();
    }
    else if (fname.find("this") == 0) {
      //we need to double check the name has no other alpabet chars, except numbers: thisXXXX
      bool realThis = true; //default to be real "this"function
      for (int i=4; i<fname.size(); i++) {
        if (!isdigit(fname[i])) {
          realThis = false;
          break;
        }
      }
      
      if (realThis)
        blamedArgs.insert(0);
    }

    // We really shouldn't add lineNumber(_ln) and fileName(_fn) into blamedArgs
    int whichParam = 0;
    // Iterates through all formal args for a function 
    for (Function::arg_iterator af_i = func->arg_begin(); af_i != func->arg_end(); af_i++) {
      Argument *v = dyn_cast<Argument>(af_i);
      const Type *origT = getPointedType(v->getType());
      if (v->hasName()) {
        string argName = v->getName().str();
        if (argName.compare("_fn")==0 || argName.compare("_ln")==0) {
          if (blamedArgs.find(whichParam) != blamedArgs.end()) //if it's already in blamedParams
            blamedArgs.erase(whichParam);
        }
        // TODO: make it more robust, special case for Barrier class'related functions
        else if (origT->getTypeID() == Type::StructTyID) {
		  const llvm::StructType *arg_st = cast<StructType>(origT);
          if (arg_st->hasName()) { //conservative way for the Barrier-related type name as far as we know
			string arg_st_name = arg_st->getName().str(); //type name of arg
            if (arg_st_name.find("Barrier_") != string::npos 
                    || arg_st_name.find("_Barrier") != string::npos)
              blamedArgs.insert(whichParam);
          }
        }
      }//formal args should always have names
      //check next formal arg
      whichParam++;
    }   
}



// Pass on Chapel internal module functions to get the chapel_internals.bs file
void FunctionBFC::externFuncPass(Function *F, vector<NodeProps *> &globalVars, 
        ExternFuncBFCHash &efInfo, ostream &args_file)
{
    string infoO("OUTPUT/MODULES/");
    infoO += getSourceFuncName();
    blame_info.open(infoO.c_str());

    // First, if the function has a non-void return value, then add -1 first
    if (func->getReturnType()->getTypeID() != Type::VoidTyID) {
 	  blamedArgs.insert(-1);
    }
    // Then run the normal parseLLVM, and truncate genGraph
    parseLLVM(globalVars);
    genGraphTrunc(efInfo);
    // We need to manually tweak blamedArgs
    tweakBlamedArgs();
  
    //output blamed arguments to args_file only if size>0
    if (blamedArgs.size()>0) {
      set<int>::iterator si;
      args_file<<getSourceFuncName();
      for (si=blamedArgs.begin(); si!=blamedArgs.end(); si++) 
        args_file<<"\t"<<(*si);
      args_file<<"\n";
    }
    cout<<"End of externFuncPass on "<<getSourceFuncName()
        <<", bA.size="<<blamedArgs.size()<<endl;
} 
  

/* LLVM pass that is ran on each function, calculates explicit/implicit relationships
 and generates transfer function for each */
void FunctionBFC::firstPass(Function *F, vector<NodeProps *> &globalVars,
        ExternFuncBFCHash &efInfo, ostream &blame_file, ostream &blame_se_file,
		ostream &call_file, int &numMissing) 
{
    string infoO("OUTPUT/");
	infoO += getSourceFuncName(); //llvm func name
	blame_info.open(infoO.c_str());
	time_t start, end;
	double dif;
	time(&start);
    cout<<"Starting first pass for "<<getSourceFuncName()<<endl;
	
	string cInfo("CALLS/");
	cInfo += getSourceFuncName();
	ofstream call_info;
	call_info.open(cInfo.c_str());
	
    parseLLVM(globalVars);
#ifdef DEBUG_P
    cout<<"parseLLVM in firstPass finished"<<endl;
#endif 
	
    genGraph(efInfo);
	
	// calls recursiveExamineChildren and populates parent/child dependencies and
	// the line numbers
	populateImportantVertices();
	
	// For local vars we want to get rid of redundant information we'll never use
	//trimLocalVarPointers();
	
	// changes the parent/child relationship between calls and parameters into explicit
	// call relationships
	resolveIVCalls();
	
	resolveLooseStructs();
	
	// Calc aggregate line numbers (and calls)
	calcAggregateLN(); //result doesn't depend on the new Graph since that's for .dot
	//makeNewTruncGraph();//Removed by Hui 07/12/17:cause infinite loop on advance22
	                      //we don't use .dot files for now, so delete it temp
	resolveSideEffects();
	resolveSideEffectCalls();
	
	resolveCallsDomLines();
	
	int ccValue = 0;
    //01/19/18: not sure why we call this function
	//ccValue = checkCompleteness();
	
	exportEverything(blame_file, false);
#ifdef DEBUG_SUMMARY
	exportEverything(blame_info, false);
#endif
	
	exportSideEffects(blame_se_file);
#ifdef DEBUG_SUMMARY
	exportSideEffects(blame_info);
#endif
	//exportCalls(call_info);
	exportCalls(call_file, efInfo);
	//exportParams(param_file);
	numMissing += ccValue;
	
	time(&end);
	dif = difftime(end,start);
	int dif_i = (int) dif;
	
	cout<<"Time for firstPass on "<<getSourceFuncName()<<" = "<<dif_i<<endl;
	//printf("  -- %d\n", dif);
	cout<<"End of firstPass on "<<getSourceFuncName()<<endl;
	//if (moreThanOneEV())
	//numMultipleEV++;
}
