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

#include "ExitVars.h"
#include "FunctionBFC.h"
#include <iostream>
#include <map>

#include <ctype.h>
#include <string.h>

using namespace std;

void FunctionBFC::transferExternCallEdges(vector< pair<int, int> > &blamedNodes, int callNode, vector< pair<int, int> > & blameeNodes)
{
  bool inserted, existed;
  graph_traits < MyGraphType >::edge_descriptor ed, oldEg;
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
  vector< pair<int, int> >::iterator v_pair_i;
  for (v_pair_i = blameeNodes.begin(); v_pair_i != blameeNodes.end(); v_pair_i++){
    // First int (param number), Second int (node number)
    pair<int, int> blamee = *v_pair_i;
    //1st, remove old edge: blamee->callNode; it could be possible that the previous blamee->callnode edge was already removed    
    tie(oldEg, existed) = edge(blamee.second, callNode, G);
    if (existed) {
      remove_edge(blamee.second, callNode, G);
    }
    //2nd, add edge: callNode->blamee
    tie(ed, inserted) = add_edge(callNode, blamee.second, G);

    blame_info<<"Adding ERCALL(1) edge for blamee param "<<blamee.first<<"("<<blamee.second<<") of "<<callNode<<" call"<<endl;
    
    if (inserted) {
      // We need to have as much info in the graph as possible,
      // so we want to record the param number of the call
      // since this is only partially resolved we need to keep this
      // info around as long as possible
      edge_type[ed] = RESOLVED_EXTERN_OP;  
      
      // It's resolved, no need to acknowledge it as a call node anymore
      NodeProps *callV = get(get(vertex_props, G), callNode);
      callV->nStatus[CALL_NODE] = false;
      
      NodeProps * blameeV = get(get(vertex_props, G), blamee.second);
      blameeV->line_num = 0;
      
      int remainingCalls = 0;
      // We can reassign call statuses since these calls have now been resolved
      boost::graph_traits<MyGraphType>::out_edge_iterator oe_beg, oe_end;
      oe_beg = boost::out_edges(blamee.second, G).first;//edge iter begin
      oe_end = boost::out_edges(blamee.second, G).second;//edge iter end
      for(; oe_beg != oe_end; ++oe_beg) {
        int opCode = get(get(edge_iore, G),*oe_beg);
        if (opCode == Instruction::Call)
          remainingCalls++;
      }
      
      boost::graph_traits<MyGraphType>::in_edge_iterator ie_beg, ie_end;
      ie_beg = boost::in_edges(blamee.second, G).first;//edge iterator begin
      ie_end = boost::in_edges(blamee.second, G).second;//edge iterator end
      for(; ie_beg != ie_end; ++ie_beg) {
        int opCode = get(get(edge_iore, G),*ie_beg);
        if (opCode == Instruction::Call)
          remainingCalls++;
      }
      
      if (!remainingCalls) {
        blameeV->nStatus[CALL_PARAM] = false;
        blameeV->nStatus[CALL_RETURN] = false;
        blameeV->nStatus[CALL_NODE] = false;

#ifdef ONLY_FOR_PARAM1 // we still want all call-related nodes to be important
                //blameeV->nStatus[IMP_REG] = true; //commented out by Hui 07/31/17
#endif
      }
    }
  }
  
  for (v_pair_i = blamedNodes.begin(); v_pair_i != blamedNodes.end(); v_pair_i++) {
    // First int (param number), Second int (node number)
    pair<int, int> blamed = *v_pair_i;
    remove_edge(blamed.second, callNode, G);
    tie(ed, inserted) = add_edge(blamed.second, callNode, G);
#ifdef DEBUG_EXTERN_CALLS
    blame_info<<"Adding ERCALL(2) edge for blamed param "<<blamed.first<<"("<<blamed.second<<") of "<<callNode<<" call"<<endl;
#endif
    
    if (inserted) {
      edge_type[ed] = RESOLVED_EXTERN_OP;  
      NodeProps *callV = get(get(vertex_props, G), callNode);
      callV->nStatus[CALL_NODE] = false;

      NodeProps *blamedV = get(get(vertex_props, G), blamed.second);
      //TODO: 03/10/17: Is it to set the blamed Arg to be written ? or establish a new relationship between the blamed and blamee
      //blamedV->isWritten = true;
      /*
        For an external call like: a = exfunc(b,c,d), NONE of a~d is written for sure if they are pointers (pointers are written if
        the memory address they represents are written into. If b is blamed, then b has outgoing edges to a,c,d through the callnode
        "exfunc", and we establish a blamees set for each blamed arg, when we do checkIfWritten for blamed arg "b", we count these
        blamees to its sets. Blamed args are NOT necessarily written but they are if anyone of the blamees is written
      */
      vector<pair<int, int>>::iterator v_pair_i2;
      for (v_pair_i2=blameeNodes.begin(); v_pair_i2!=blameeNodes.end(); v_pair_i2++) {
        pair<int, int> blamee2 = *v_pair_i2;
        NodeProps *blameeV2 = get(get(vertex_props, G), blamee2.second);
        blamedV->blameesFromExFunc.insert(blameeV2);
      }

      int remainingCalls = 0;
      //blamedV->externCallLineNumbers.insert(callNode->line_num);
      // We can reassign call statuses since these calls have now been resolved
      boost::graph_traits<MyGraphType>::out_edge_iterator oe_beg, oe_end;
      oe_beg = boost::out_edges(blamed.second, G).first;//edge iter begin
      oe_end = boost::out_edges(blamed.second, G).second;//edge iter end
      for(; oe_beg != oe_end; ++oe_beg) {
        int opCode = get(get(edge_iore, G),*oe_beg);
        if (opCode == Instruction::Call)
          remainingCalls++;
      }
      
      boost::graph_traits<MyGraphType>::in_edge_iterator ie_beg, ie_end;
      ie_beg = boost::in_edges(blamed.second, G).first;//edge iter begin
      ie_end = boost::in_edges(blamed.second, G).second;//edge iter end
      for(; ie_beg != ie_end; ++ie_beg) {
        int opCode = get(get(edge_iore, G),*ie_beg);
        if (opCode == Instruction::Call)
          remainingCalls++;
      }
      
      if (!remainingCalls) {
        blamedV->nStatus[CALL_PARAM] = false;
        blamedV->nStatus[CALL_RETURN] = false;
        blamedV->nStatus[CALL_NODE] = false;
  
#ifdef ONLY_FOR_PARAM1  //we still want all call-related nodes to be important
                //blamedV->nStatus[IMP_REG] = true;//commented out by Hui 07/31/17
#endif
      }
    }
  }
}

char* FunctionBFC::trimTruncStr(const char *truncStr)
{
  int last = (int)strlen(truncStr);
  if (!last) //last==0
    return NULL;

  while (isdigit(truncStr[last-1])) {
    last--;
  }

  const char *startPtr = truncStr;
  char *tempStr = new char[last + 1];
  strncpy(tempStr, startPtr, last);
  tempStr[last] = '\0';
  char *trimedStr = tempStr;
  return trimedStr;
}

string FunctionBFC::getTruncStr(string fullStr)
{
  size_t endPtr = fullStr.find("--");
  string res("");

  if (endPtr == string::npos || fullStr.empty())
    return res;

  //for llvm.memcpy* like intrinsic externfuncs, we also need to 
  //remove the register names after it, like .p0i8.p0i8.i64
  if (fullStr.find("llvm.") == 0) {
    endPtr = fullStr.find('.', 6); //we are interested in the second dot
  }
  
  res = fullStr.substr(0, endPtr);
  return res;
}


void FunctionBFC::handleOneExternCall(ExternFunctionBFC *efb, NodeProps *v)
{
#ifdef DEBUG_EXTERN_CALLS
  blame_info<<"Calls__(handleOneExternCall) -looking at "<<v->name<<endl;
#endif
  int v_index = v->number;
  
  boost::graph_traits<MyGraphType>::in_edge_iterator ie_beg, ie_end;
  ie_beg = boost::in_edges(v_index, G).first;  // edge iterator begin
  ie_end = boost::in_edges(v_index, G).second;// edge iterator end
  
  // First int (param number), Second int (node number)
  vector< pair<int, int> > blameeNodes;
  vector< pair<int, int> > blamedNodes;

  for(; ie_beg != ie_end; ++ie_beg) {
    NodeProps *inTargetV = get(get(vertex_props,G), source(*ie_beg,G));
    set<FuncCall *>::iterator fc_i = inTargetV->funcCalls.begin();
    
    for (; fc_i != inTargetV->funcCalls.end(); fc_i++) {
      FuncCall *fc = *fc_i;
      pair<int, int> tmpPair(fc->paramNumber, inTargetV->number);
      
      if (fc->funcName != v->name)
        continue;
      
      if (efb->paramNums.count(fc->paramNumber)) {
        blamedNodes.push_back(tmpPair);
        inTargetV->isBlamedExternCallParam = true;
#ifdef DEBUG_EXTERN_CALLS
        blame_info<<inTargetV->name<<" receives blame for extern call to "<<v->name<<endl;
#endif
        
        inTargetV->externCallLineNumbers.insert(v->line_num);
        // TODO: Make this part of the config file
        if (v->name.find("gfortran_transfer") != string::npos) {
          inTargetV->isWritten = true;
        }
      }
      else {
        blameeNodes.push_back(tmpPair);
      }
    }//end of for funcCalls    
  } //end of for in_edges
  
  if (blamedNodes.size() > 0) {
    transferExternCallEdges(blamedNodes, v->number, blameeNodes);    
  }
  else {  
#ifdef DEBUG_ERROR
    blame_info<<"The blamed node vector(extern) was empty for "<<v->name<<" in "<<getSourceFuncName()<<endl;
#endif
  }
}


