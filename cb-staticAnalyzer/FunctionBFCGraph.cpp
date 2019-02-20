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
#include <sstream>

using namespace std;

/*void NodeProps::addFuncCall(FuncCall * fc)
{
  set<FuncCall*>::iterator vec_fc_i;
  
  for (vec_fc_i = funcCalls.begin(); vec_fc_i != funcCalls.end(); vec_fc_i++) {
    if ((*vec_fc_i)->funcName == fc->funcName && (*vec_fc_i)->paramNumber == fc->paramNumber)
    {
      return;
    }
  }
  
  //cout<<"Pushing back call to "<<fc->funcName<<" param "<<fc->paramNumber<<" for node "<<name<<endl;
  funcCalls.insert(fc);
}*/

//Truncate determineBFCForVertexLite for Chapel internal module functions
void FunctionBFC::determineBFCForVertexTrunc(NodeProps *v)
{
#ifdef DEBUG_EXIT
  blame_info<<"EV__(determineBFCForVertexTrunc) -- for "<<v->name<<endl;
#endif   
  
  int v_index = v->number;
  
  /*--- WHY we need the following ? delete it for now, check it later--//
  // NO E: UNREAD RETURNS
  // We check to see if this node is simply an unread return value
  int returnVal = checkForUnreadReturn(v, v_index);
  if (returnVal && !(v->isGlobal)) {
#ifdef DEBUG_EXIT
    blame_info<<"EXIT__(determineBFCForVertexTrunc) -- Program Exit (Unread Return) added for "<<v->name<<endl;
#endif
    //addExitProg(new ExitProgram(v->name, UNREAD_RET));
    //v->exitStatus = EXIT_PROG;
    return;
  }
  //-------------------------------------------------------------*/
  
  // EV:  RETURN VALUES
  // This sees if the LLVM exit variable (no incoming edges, 1+ outgoing edges) 
  // matches a previously discovered return variable (DEFAULT_RET)
  if (v->name.find("retval") != string::npos ) {
    for (vector<ExitVariable *>::iterator ev_i = exitVariables.begin(); 
         ev_i != exitVariables.end();  ev_i++) {
      if ((*ev_i)->realName.compare("DEFAULT_RET") == 0) {
        (*ev_i)->addVertex(v);
        (*ev_i)->vertex = v;
        v->eStatus = EXIT_VAR_RETURN;
                
        blame_info<<"Weird: we found a default retval !"<<endl;
        blamedArgs.insert(-1); //unlikely to happen
        return;
      }
    }
    // This would occur in a case where our prototype analyzer showed a void
    // return value and LLVM still had a return value to go with it
#ifdef DEBUG_ERROR
    cerr<<"Where is the return exit variable?  "<<v->name<<"\n";
#endif
  }
    
  // EV:  GENERAL CASE (PARAMS) 
  // Looking for an exact match to the predetermined exit variables or 
  //   a match for the pointsTo NodeProps in the case of a GEP node
  for (vector<ExitVariable *>::iterator ev_i = exitVariables.begin(); 
        ev_i != exitVariables.end();  ev_i++) {
    // compare()==0 means two strings are equal
    if (v->name.compare((*ev_i)->realName)==0 && v->isWritten) {
      (*ev_i)->addVertex(v);
      (*ev_i)->vertex = v;
#ifdef DEBUG_EXIT 
      blame_info<<"EV__(LLVMtoExitVars3) -- Add node for "<<v->name<<endl;
#endif 
      v->eStatus = EXIT_VAR_PARAM + (*ev_i)->whichParam;
      v->paramNum = (*ev_i)->whichParam;
            //Added by Hui 12/19/16
      blamedArgs.insert((*ev_i)->whichParam);
      return;
    }
  }
}

// This function examines vertices that have no incoming edges, 
//   but have at least one outgoing edge
void FunctionBFC::determineBFCForVertexLite(NodeProps *v)
{
  
#ifdef DEBUG_EXIT
  blame_info<<"EV__(determineBFCForVertexLite) -- for "<<v->name<<endl;
#endif   
  
  int v_index = v->number;
  int in_d = in_degree(v_index, G);
  int out_d = out_degree(v_index, G);
  
    /*--- WHY we need the following ? delete it for now, check it later--//
  // NO E: UNREAD RETURNS
  // We check to see if this node is simply an unread return value
  /int returnVal = checkForUnreadReturn(v, v_index);
  /if (returnVal && !(v->isGlobal)) {
#ifdef DEBUG_EXIT
    blame_info<<"EXIT__(determineBFCForVertexLite) -- Program Exit (Unread Return) added for "<<v->name<<endl;
#endif
    //addExitProg(new ExitProgram(v->name, UNREAD_RET));
    //v->exitStatus = EXIT_PROG;
    return;
  }
  //-------------------------------------------------------------*/
  // EV: GLOBALS
  // By being in this function we already know that the return value was 
    // used in some form in this function (in/out degree of greater than 1) 
    // so we assign it an exit value
  if (v->isGlobal) {
#ifdef DEBUG_EXIT
      blame_info<<v->name<<" isGlobal !"<<endl;
#endif
    v->eStatus = EXIT_VAR_GLOBAL;
    }
  
  // EV:  RETURN VALUES
    // This sees if the LLVM exit variable (no incoming edges, 1+ outgoing edges) 
    // matches a previously discovered return variable (DEFAULT_RET)
    if (v->name.find("retval") != string::npos ) {
    for (vector<ExitVariable *>::iterator ev_i = exitVariables.begin(); 
          ev_i != exitVariables.end();  ev_i++) {
    if ((*ev_i)->realName.compare("DEFAULT_RET") == 0) {
      (*ev_i)->addVertex(v);
      (*ev_i)->vertex = v;
      v->eStatus = EXIT_VAR_RETURN;
      return;
    }
    }
    
    // This would occur in a case where our prototype analyzer showed a void
    // return value and LLVM still had a return value to go with it
#ifdef DEBUG_ERROR
    blame_info<<"Where is the return exit variable?  "<<v->name<<"\n";
#endif
  }
  
  // EV:  GENERAL CASE (PARAMS)
    // Looking for an exact match to the predetermined exit variables or 
    //   a match for the pointsTo NodeProps in the case of a GEP node
    for (vector<ExitVariable *>::iterator ev_i = exitVariables.begin(); 
      ev_i != exitVariables.end();  ev_i++) {
    // compare()==0 means two strings are equal
    if (v->name.compare((*ev_i)->realName)==0) {
    (*ev_i)->addVertex(v);
    (*ev_i)->vertex = v;
#ifdef DEBUG_EXIT 
    blame_info<<"EV__(LLVMtoExitVars2) -- Add node for "<<v->name<<endl;
#endif 
    v->eStatus = EXIT_VAR_PARAM + (*ev_i)->whichParam;
    v->paramNum = (*ev_i)->whichParam;
      return;
    }
  }
}


void FunctionBFC::determineBFCForOutputVertexLite(NodeProps *v, int v_index)
{
#ifdef DEBUG_EXIT_OUT
  blame_info<<"OUT__(determineBFCForOutputVertex) - Entering func for "<<v->name<<endl;
#endif
  set<FuncCall *>::iterator fc_i = v->funcCalls.begin();
  
  for (; fc_i != v->funcCalls.end(); fc_i++) {
      FuncCall *fc = *fc_i;
    if (fc && fc->funcName == v->name) 
    fc->outputEvaluated = true;
  }
  
  set<int> inputVertices;
  int zeroParam = -1;
  NodeProps *zParam = NULL;
  
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
    bool inserted;
    graph_traits < MyGraphType >::edge_descriptor ed;
  
  boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
  
  e_beg = boost::in_edges(v_index, G).first;    // edge iterator begin
  e_end = boost::in_edges(v_index, G).second;    // edge iterator end
  
  // iterate through the edges to find matching opcode
  for (; e_beg != e_end; ++e_beg) {
    int opCode = get(get(edge_iore, G),*e_beg);
    
    if (opCode == Instruction::Call) {
      int sourceV = get(get(vertex_index, G), source(*e_beg, G));
    //int targetV = get(get(vertex_index, G), target(*e_beg, G));
      
    NodeProps *sourceVP = get(get(vertex_props, G), source(*e_beg, G));
    NodeProps *targetVP = get(get(vertex_props, G), target(*e_beg, G));
        if (!sourceVP || !targetVP) continue;

    int paramNum = MAX_PARAMS + 1;
#ifdef DEBUG_EXIT_OUT
    blame_info<<"OUT__(determineBFCForOutputVertexLite) - Call from "<<sourceVP->name<<" to "<<targetVP->name<<endl;
#endif
    //vector<FuncCall *>::iterator 
    fc_i = sourceVP->funcCalls.begin();
      
    for (; fc_i != sourceVP->funcCalls.end(); fc_i++) {
      FuncCall *fc = *fc_i;
#ifdef DEBUG_EXIT_OUT
      blame_info<<"OUT__(determineBFCForOutputVertexLite) - FC -- "<<fc->funcName<<"  "<<targetVP->name<<endl;
#endif  
      if (fc->funcName == targetVP->name) {
      fc->outputEvaluated = true;
      paramNum = fc->paramNumber;
      if (paramNum == -1) { //changed by Hui 12/31/15: 0=>-1
          zeroParam = sourceVP->number;
        zParam = sourceVP;
#ifdef DEBUG_EXIT_OUT
        blame_info<<"Inputs to "<<v->name<<" is "<<sourceVP->name<<" "<<paramNum<<endl;
#endif
      }
      else if (paramNum >= 0) {//changed by Hui 12/31/15:'>' => '>='
        inputVertices.insert(sourceV);
#ifdef DEBUG_EXIT_OUT
        blame_info<<"Inputs to "<<v->name<<" is "<<sourceVP->name<<" "<<paramNum<<endl;
#endif
            }
      break;
      }            
    }
    }
  }      
  
  if (zeroParam > -1) {
#ifdef DEBUG_EXIT_OUT
    blame_info<<"OUT__(determineBFCForOutputVertexLite) - Removing edge to "<<v->name<<" from "<<zeroParam<<endl;
#endif
    int in_d = in_degree(zeroParam, G);
    if (in_d == 0)
    remove_edge(zeroParam, v->number, G);
    zParam->nStatus[CALL_RETURN] = false;  
  }
  
  set<int>::iterator set_i;
  for (set_i = inputVertices.begin(); set_i != inputVertices.end();  set_i++) {
    remove_edge(*set_i, v->number, G);
    tie(ed, inserted) = add_edge(v->number, *set_i, G);
    
    if (inserted) {
    edge_type[ed] = RESOLVED_OUTPUT_OP;  
    NodeProps *dNode = get(get(vertex_props, G), *set_i);
    if (dNode->eStatus == EXIT_PROG)
        dNode->eStatus = NO_EXIT;
    
    //NodeProps * blamedV = get(get(vertex_props, G), blamed.second);
    int remainingCalls = 0;
    //We can reassign call statuses since these calls have now been resolved
    boost::graph_traits<MyGraphType>::out_edge_iterator oe_beg, oe_end;
    oe_beg = boost::out_edges(*set_i, G).first;    // edge iterator begin
    oe_end = boost::out_edges(*set_i, G).second;       // edge iterator end
    for (; oe_beg != oe_end; ++oe_beg) {
      int opCode = get(get(edge_iore, G),*oe_beg);
      if (opCode == Instruction::Call)
      remainingCalls++;
    }
      
        boost::graph_traits<MyGraphType>::in_edge_iterator ie_beg, ie_end;
    ie_beg = boost::in_edges(*set_i, G).first;    // edge iterator begin
    ie_end = boost::in_edges(*set_i, G).second;       // edge iterator end
    for (; ie_beg != ie_end; ++ie_beg) {
      int opCode = get(get(edge_iore, G),*ie_beg);
      if (opCode == Instruction::Call)
        remainingCalls++;
    }  
  
        if (!remainingCalls) {
      dNode->nStatus[CALL_PARAM] = false;
      dNode->nStatus[CALL_RETURN] = false;
      dNode->nStatus[CALL_NODE] = false;
      }  
    }
    }
  
    LLVMtoOutputVar(v);
}



bool FunctionBFC::isLibraryOutput(const char * tStr)
{
  if ((strcmp("printf",tStr) == 0) ||
        (strcmp("fprintf",tStr) == 0) ||
        (strcmp("vfprintf",tStr) == 0) ||
        (strcmp("vprintf",tStr) == 0) ||
        (strcmp("fputc",tStr) == 0) ||
        (strcmp("fputs",tStr) == 0) ||
        (strcmp("putc",tStr) == 0) ||
        (strcmp("putchar",tStr) == 0) ||
        (strcmp("puts",tStr) == 0) ||
        (strcmp("fwrite",tStr) == 0) ||
        (strcmp("fflush",tStr) == 0) ||
        (strcmp("perror",tStr) == 0) || 
        (strcmp("write",tStr) == 0) || //adde by Hui 12/31/15 for chpl
        (strcmp("writeln",tStr) == 0)) //added by Hui 12/31/15 for chpl
  {    
    return true;
  }
  
  /*
   if ( (strcmp("_gfortran_st_write",tStr) == 0) ||
   (strcmp("_gfortran_st_read",tStr) == 0))
   {    
   return true;
   }
   */
  
  
  
  return false;
  
}


// Add ExitVars for args that are passed in as Pid values
void FunctionBFC::recheckExitVars()
{
  int whichParam = 0;
  for (Function::arg_iterator af_i = func->arg_begin(); af_i != func->arg_end(); af_i++) {
    Argument *arg = dyn_cast<Argument>(af_i);
    const Type *argT = arg->getType();
    // if arg is simply an integer, then it wasn't added for EV before
    if (argT->isIntegerTy()) {
#ifdef DEBUG_EXIT
      blame_info<<"In recheckExitVars, we have an int arg: "<<arg->getName().str()<<endl;
#endif
      for (Value::use_iterator u_i=arg->use_begin(), u_e=arg->use_end(); u_i!=u_e; u_i++) {
        if (Instruction *i = dyn_cast<Instruction>((*u_i).getUser())) {
          if (i->getOpcode() == Instruction::BitCast ||
              i->getOpcode() == Instruction::AddrSpaceCast)  { //All OpCode can be found in instruction.def
            //Value * bcV = i;
            for (Value::use_iterator u_i2 = i->use_begin(), u_e2 = i->use_end(); u_i2 != u_e2; ++u_i2) { 
              // Verify that the Value is that of an instruction 
              if (Instruction *i2 = dyn_cast<Instruction>((*u_i2).getUser())) {
                if (i2->getOpcode() == Instruction::Store) {
                  User::op_iterator op_i = i2->op_begin(); //typedef Use* op_iterator
                  op_i++;
                  Value *second = op_i->get(); // second is the actual mem address where to store the value
            
                  if (second->hasName()) { //in this case, firt can be a register since the original arg has been bicasted 
                    string argHolderName = second->getName().str();
                    if (argHolderName.find(PARAM_REC) != string::npos || argHolderName.find(PARAM_REC2) != string::npos) {
                      if (variables.count(argHolderName)) {
                        NodeProps *vp = variables[argHolderName];
                        // We only add EV for Pids, otherwise all int param will be EVs
                        if (vp->isPid) {
                          addExitVar(new ExitVariable(argHolderName, PARAM, whichParam, false));
#ifdef DEBUG_EXIT  
                          blame_info<<"LLVM_(checkFunctionProto) - Adding exit var for Pid Param: "<<argHolderName<<" "<<whichParam<<endl;
#endif
                        }
                      }
                    }
#ifdef DEBUG_EXIT
                    blame_info<<"LLVM_(checkFunctionProto) - Wrong argHolderName: "<<argHolderName<<endl;
#endif
                  }

                  else {//second no name 
#ifdef DEBUG_EXIT  
                    blame_info<<"LLVM_(checkFunctionProto) - what's going on here(2)"<<endl;
#endif
                  } //second no name
                } //i2 is Store
              }  //i2 is instruction
            } //for loop of i2's uses
          } //i is BitCast
      
          else if (i->getOpcode() == Instruction::Store) {
            User::op_iterator op_i = i->op_begin();
            Value  *first = op_i->get();
            op_i++;
            Value *second = op_i->get();
        
            if (first->hasName() && second->hasName()) { //In this case, first must have name since it's the Arg
              string argHolderName = second->getName().str();
              if (argHolderName.find(PARAM_REC) != string::npos || argHolderName.find(PARAM_REC2) != string::npos) {
                if (variables.count(argHolderName)) {
                  NodeProps *vp = variables[argHolderName];
                  // We only add EV for Pids, otherwise all int param will be EVs
                  if (vp->isPid) {
#ifdef DEBUG_EXIT  
                    blame_info<<"LLVM_(checkFunctionProto) - Adding exit var for Pid Param: "<<argHolderName<<" "<<whichParam<<endl;
#endif
                    addExitVar(new ExitVariable(argHolderName, PARAM, whichParam, false));
                  }
                }
              }
              else {
#ifdef DEBUG_EXIT
                blame_info<<"LLVM_(checkFunctionProto) - wrong name? arg: "<<first->getName().str()<<" holder: "<<argHolderName<<endl;
#endif
              }
            } //first and second has name
            else { //either of first and second doesn't have name
#ifdef DEBUG_EXIT  
              blame_info<<"LLVM_(checkFunctionProto) - what's going on here? "<<first->hasName()<<""<<second->hasName()<<endl;
#endif
            }
          } // i is Store
        } // i is Instruction
      } // end of i's uses, for loop  
    } // arg is Integer

    whichParam++;
  } // for all args         
}


// Truncate determineBFCHoldersLite function for 
void FunctionBFC::determineBFCHoldersTrunc()
{
  // Iterate through all vertices and determine those that have no incoming edges (blame candidates)
  // but also have some out_degrees (eliminating outlier symbols)
    graph_traits<MyGraphType>::vertex_iterator i, v_end;
    for(tie(i,v_end) = vertices(G); i != v_end; ++i) {
    int v_index = get(get(vertex_index, G),*i);
    NodeProps *v = get(get(vertex_props, G),*i);
    int in_d = in_degree(v_index, G);
    int out_d = out_degree(v_index, G);
    if (!v) {
        blame_info<<"Weird, Null V in determineBFCHoldersTrunc, index="<<v_index<<endl;
        continue;
      }
      else {
    if (v->deleted)
      continue;
    
    if (v->isLocalVar)
      v->nStatus[LOCAL_VAR] = true;
    
    if (in_d == 0){ //reg or local vars that have no incoming edges
      if (out_d > 0 || v->isLocalVar)
          determineBFCForVertexTrunc(v);
    }
      else if (v->isFormalArg) {
        // We always check the parameters, the EV sanity check will kick in ... hopefully
#ifdef DEBUG_EXIT
        blame_info<<"EXIT__(determineBFCHoldersTrunc) for isFormalArg(no holder) "<<v->name<<endl;
#endif
        determineBFCForVertexTrunc(v);
      }
    else if (v->isGlobal && (in_d > 0 || out_d > 0)) {
#ifdef DEBUG_EXIT
      blame_info<<"EXIT__(determineBFCHoldersTrunc) for GLOBAL with in_d || out_d > 0 "<<v->name<<endl;
#endif
      determineBFCForVertexTrunc(v);
    }
    else if (v->name.find("retval") != string::npos) {
#ifdef DEBUG_EXIT
      blame_info<<"EXIT__(determineBFCHoldersTrunc) for return val with in_d || out_d > 0 "<<v->name<<endl;
#endif
      determineBFCForVertexTrunc(v);  
    }
      } // v exist
  } //all vertices
  
}

// Normal determineBFCHoldersLite for user function
void FunctionBFC::determineBFCHoldersLite()
{
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
  // First pass:iterate through all vertices and determine those that have no incoming edges 
  // (blame candidates) but also have some out_degrees (eliminating outlier symbols)
  for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    int v_index = get(get(vertex_index, G),*i);
    NodeProps *v = get(get(vertex_props, G),*i);
    int in_d = in_degree(v_index, G);
    int out_d = out_degree(v_index, G);
    if (!v) continue;

    if (v->deleted)
      continue;
  
    if (v->isLocalVar)
      v->nStatus[LOCAL_VAR] = true;
    
    if (in_d == 0 && (out_d > 0 || v->isLocalVar)) {
      determineBFCForVertexLite(v);
    }

    else if (v->isFormalArg) {
      //We always check the parameters, the EV sanity check will kick in
#ifdef DEBUG_EXIT
      blame_info<<"EXIT__(determineBFCHolders) for isFormalArg(holder) "
          <<v->name<<endl;
#endif
      determineBFCForVertexLite(v);
    }
    else if (v->isGlobal && (in_d > 0 || out_d > 0)) {
#ifdef DEBUG_EXIT
      blame_info<<"EXIT__(determineBFCHolder) for GLOBAL with in_d || out_d > 0 "
          <<v->name<<endl;
#endif
      determineBFCForVertexLite(v);
    }
    else if (v->name.find("retval") != string::npos) {
#ifdef DEBUG_EXIT
      blame_info<<"EXIT__(determineBFCHolder) for return val with in_d > 0 "
          <<v->name<<endl;
#endif
      determineBFCForVertexLite(v);  
    }
  }
  
  // Second pass: through the vertices to handle 
  // cases for output, we don't want it to interfere
  // with our normal pass, which is why we do a separate
  // pass
  for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    int v_index = get(get(vertex_index, G),*i);
    NodeProps *v = get(get(vertex_props, G),*i);
  
    if (v == NULL) {
#ifdef DEBUG_ERROR
      blame_info<<"ERROR__(DBHLite) - V is null: idx="<<v_index<<endl;
#endif
      continue;
    }  
    
    if (v->name.empty())
      continue;
    
    string tStr = getTruncStr(v->name);
    if (tStr.empty())
      continue;
    
    if (isLibraryOutput(tStr.c_str()))
      determineBFCForOutputVertexLite(v, v_index);
  }
}

void FunctionBFC::identifyExternCalls(ExternFuncBFCHash &efInfo)
{
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
  
  // We need to examine all of the vertices to see those that are a function call
  for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    NodeProps *v = get(get(vertex_props, G),*i);
    if (v) { 
      if (v->deleted == true)
        continue;
    
      if (v->funcCalls.size() > 0 && v->nStatus[CALL_NODE]) { // we only check call node for obvious reason
#ifdef DEBUG_GRAPH_BUILD
        blame_info<<"Graph__(identifyExternCalls) - Vertex "<<v->name<<" involved"<<endl;
#endif
        set<FuncCall *>::iterator fc_i = v->funcCalls.begin();  
#ifdef DEBUG_GRAPH_BUILD
        for (; fc_i != v->funcCalls.end(); fc_i++) {
          FuncCall *fc = *fc_i;  
          blame_info<<"Graph__(identifyExternCalls) - Func -  "<<fc->funcName<<" Param - "<<fc->paramNumber<<endl;
        }
#endif
        ExternFunctionBFC *efb = NULL;
        string tStr = getTruncStr(v->name); //v has to be the CALL_NODE, so that it has the func name
#ifdef DEBUG_GRAPH_BUILD
        blame_info<<"Graph__(identifyExternCalls) - Vertex name after trunc: "<<tStr<<endl;
#endif
        if (!tStr.empty()) 
          efb = efInfo[tStr];

        if (efb != NULL)
          handleOneExternCall(efb,v);
      }//if v is a CALL_NODE
    }//v exist
  }//end of for all vertices
}


void FunctionBFC::resolvePointersHelper2(NodeProps *origV, int origPLevel, NodeProps *targetV, 
        set<int> &visited, set<NodeProps *> &tempPointers, NodeProps *alias, int origOpCode)
{
#ifdef DEBUG_RP
  blame_info<<"In resolvePointersHelper for "<<targetV->name<<" oV - "<<origV->name<<endl;
#endif 
  int newPtrLevel = 0;
  const llvm::Type *origT = NULL;    
  if (targetV->llvm_inst != NULL) {
    if (isa<Instruction>(targetV->llvm_inst)) {
      Instruction *pi = cast<Instruction>(targetV->llvm_inst);  
      origT = pi->getType();  
      newPtrLevel = pointerLevel(origT,0);
    }
    else if (isa<ConstantExpr>(targetV->llvm_inst)) {
      ConstantExpr *ce = cast<ConstantExpr>(targetV->llvm_inst);
      origT = ce->getType();
      newPtrLevel = pointerLevel(origT, 0);
    }
  }
  //targetV is blamed in the call to an extern func
  if (targetV->isBlamedExternCallParam) { 
    if (targetV->isWritten)
      origV->isWritten = true;
  }
  
  if (visited.count(targetV->number) > 0)
    return;

  visited.insert(targetV->number);
#ifdef DEBUG_RP
  blame_info<<"Ptr Level(targetV) - "<<newPtrLevel<<" oV: "<<origPLevel<<endl;
#endif
  
  // Not interested anymore, we're out of pointer territory
  //TODO: Check if origV is *struct and targetV is struct, even though newPtrLevel=0, 
  //should we also calculate aliases for origV (*struct) ??? 07/31/17
  if (newPtrLevel == 0)//&& targetV->storeLines.size() == 0)
    return;
  
  else if (newPtrLevel <= origPLevel) {// still in the game
#ifdef DEBUG_RP
    blame_info<<"In inner loop for "<<targetV->name<<endl;
#endif 
    boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
    e_beg = boost::in_edges(targetV->number, G).first;  //edge iterator begin
    e_end = boost::in_edges(targetV->number, G).second; // edge iterator end
    
    // iterate through the edges trying to find relationship between pointers
    for (; e_beg != e_end; ++e_beg) {
      int opCode = get(get(edge_iore, G),*e_beg);
      NodeProps *tV = get(get(vertex_props,G), source(*e_beg,G));
      if (!tV) continue;
#ifdef DEBUG_RP
      blame_info<<"In Edges for "<<targetV->name<<"  "<<opCode<<endl;
#endif 
      
      int tVPtrLevel = 0;
      const llvm::Type *origT2;    
      int tVPointedTypeID = -100; //for whatever reason it's not calculated, then it's invalid
      
      if (tV->llvm_inst != NULL) {
        if (isa<Instruction>(tV->llvm_inst)) {
          Instruction *pi = cast<Instruction>(tV->llvm_inst);  
          origT2 = pi->getType();  
          tVPtrLevel = pointerLevel(origT2,0);
          tVPointedTypeID = getPointedTypeID(origT2);
        }
      }
      else if (isa<ConstantExpr>(tV->llvm_inst)) {
        ConstantExpr *ce = cast<ConstantExpr>(tV->llvm_inst);
        origT2 = ce->getType();
        tVPtrLevel = pointerLevel(origT2, 0);
        tVPointedTypeID = getPointedTypeID(origT2);
      }      
      
      if (opCode == Instruction::Store) {  //*targetV = RLS/load **origV;
        // treat as alias               //store *targetV **tV;
#ifdef DEBUG_RP        
        blame_info<<"RHP Store between targetV("<<targetV->name<<") and tV("<<tV->name<<")"<<endl;
        blame_info<<tVPtrLevel<<" "<<origPLevel<<" "<<origV->resolvedLSFrom.size();
        blame_info<<" "<<targetV->resolvedLSFrom.size()<<endl;
#endif
        //---added by Hui 04/08/16-------------------------------//
        const llvm::Type *origT0;
        int origPointedTypeID = -200; //if for whatever reason it's not calculated, then it's invalid, should NOT = -100 !!!

        if (origV->llvm_inst != NULL) {
          if (isa<Instruction>(origV->llvm_inst)) {
            llvm::Instruction *origVpi = cast<Instruction>(origV->llvm_inst);  
            origT0 = origVpi->getType();  
            origPointedTypeID = getPointedTypeID(origT0);
          }
          else if (isa<ConstantExpr>(origV->llvm_inst)) {
            llvm::ConstantExpr *origVce = cast<ConstantExpr>(origV->llvm_inst);
            origT0 = origVce->getType();
            origPointedTypeID = getPointedTypeID(origT0);
          }
          else if (isa<ConstantPointerNull> (origV->llvm_inst)) { //added by Hui 04/19/16, for origV is global var case
#ifdef DEBUG_RP
            blame_info<<"origV("<<origV->name<<")"<<"  -- ConstantPointerNullVal"<<endl;
#endif
            ConstantPointerNull *origVcpn = cast<ConstantPointerNull>(origV->llvm_inst);
            origT0 = origVcpn->getType();
            origPointedTypeID = getPointedTypeID(origT0);    
          }
        }         

#ifdef DEBUG_RP
        blame_info<<"The tV's pointedTypeID="<<tVPointedTypeID<<", origV's pointedTypeID="<<origPointedTypeID<<endl;
#endif
        //For cases: targetV = GEP origV; store targetV, tV;
        // origV and tV should never treated as aliases, even if pointedTypeID and ptrLevel are equal
        // edge_type != GEP_BASE_OP added by Hui 04/26/16, for the above reason
        //*TypeID cond added 04/08/16:alias should be same ptr level with same Pointed type, eg: **double and **int aren't aliases 
        if (tVPtrLevel == origPLevel && origPointedTypeID == tVPointedTypeID && origOpCode != GEP_BASE_OP) { 
#ifdef DEBUG_RP                                                         
          blame_info<<"Pointer levels are equal"<<endl;
#endif
          if (tV->storesTo.size() > 1){
            blame_info<<"Inserting almost alias(1) between "<<origV->name<<" and "<<tV->name<<endl;
            //origV->almostAlias.insert(targetV); //TODO: shouldn't it between tV and origV ?
            //targetV->almostAlias.insert(origV);
            origV->almostAlias.insert(tV); //New added 01/28/16
            tV->almostAlias.insert(origV); //New
#ifdef DEBUG_RP
            blame_info<<"Inserting pointer(6) "<<targetV->name<<endl;
#endif
            //tempPointers.insert(targetV);
            tempPointers.insert(tV); //New
          }
          else {
            blame_info<<"Inserting alias(1) out "<<tV->name<<" into "<<origV->name<<endl;
            origV->aliasesOut.insert(tV);
            blame_info<<"Inserting alias(2) in "<<origV->name<<" into "<<tV->name<<endl;
            tV->aliasesIn.insert(origV);  
#ifdef DEBUG_RP
            blame_info<<"Inserting STORE ALIAS pointer(2) "<<tV->name<<endl;
#endif
            tempPointers.insert(tV);
          }
        }
        // treat as write within data space
        //how could it be not equal? like a=GEP b; store a, c; then b and c Ptrlevel could be different
        else { 
#ifdef DEBUG_RP
          blame_info<<"Pointer levels are not equal"<<endl;
#endif
          bool proceed = true;
          set<NodeProps *>::iterator vec_vp_i_in;
          for (vec_vp_i_in = origV->almostAlias.begin(); vec_vp_i_in != origV->almostAlias.end(); vec_vp_i_in++) {
#ifdef DEBUG_RP
            blame_info<<"dfAlias relation between "<<tV->name<<" and "<<(*vec_vp_i_in)->name<<endl;
#endif 
            tV->dfAliases.insert(*vec_vp_i_in);
            (*vec_vp_i_in)->dfaUpPtr = tV;
            
            // If it's returned than we find it as interesting as something written, even though it technically may not be
            if (tV->name.find("retval") != string::npos) {
              tV->isWritten = true;
              (*vec_vp_i_in)->isWritten = true;
            }
            //resolvePointersHelper2(origV, origPLevel, tV, visited, tempPointers, alias);
            proceed = false;//if origV has no almostAliases then it's still true
          }

          if (proceed) {
            origV->nonAliasStores.insert(tV);
            // if it's a local var, we will examine it later
            if (!tV->isLocalVar) {
#ifdef DEBUG_RP
              blame_info<<"Not local var, call rpH2"<<endl;
#endif 
              //TODO: Not sure about this special call, here since opCode is STORE, we still keep origV as arg"origV" and origOpCode
              resolvePointersHelper2(origV, origPLevel, tV, visited, tempPointers, alias, origOpCode);
            }
            else {
#ifdef DEBUG_RP
              blame_info<<"Is local var, do something later I guess." <<endl;
#endif
            }
          }
        }//else ptrlevel not equal
      }//it opCode == Store
      
      else if (opCode == Instruction::Load) { //if a = load b, c = load a, then b.loads.insert(c);
        if (tVPtrLevel > 0)
          origV->loads.insert(tV); 
     
        resolvePointersHelper2(targetV, origPLevel, tV, visited, tempPointers, alias, opCode);
      }

      else if(opCode == GEP_BASE_OP && origOpCode != GEP_BASE_OP) {
        string origTStr = returnTypeName(origT, string(""));
        // Dealing with a struct pointer
        if (origTStr.find("Struct") != string::npos) {
          if (origV->llvm_inst != NULL) {
            boost::graph_traits<MyGraphType>::out_edge_iterator e_beg2, e_end2;
            
            e_beg2 = boost::out_edges(tV->number, G).first;//edge iter b
            e_end2 = boost::out_edges(tV->number, G).second;//iter end 
              // iterate through the edges trying to find relationship between pointers
            for (; e_beg2 != e_end2; ++e_beg2) {
              int opCodeForField = get(get(edge_iore, G),*e_beg2);
              int fNum = 0;
              if (opCodeForField >= GEP_S_FIELD_OFFSET_OP) {
                fNum = opCodeForField - GEP_S_FIELD_OFFSET_OP; 
              
                if (isa<Value>(origV->llvm_inst)) {
                  Value *v = cast<Value>(origV->llvm_inst);  
#ifdef DEBUG_RP 
                  blame_info<<"Calling structResolve for "<<origV->name<<" of type "<<origTStr<<endl;
#endif
                  structResolve(v, fNum, tV);
                  break;
                }
              }
            }
          }
#ifdef DEBUG_RP
          blame_info<<"Transferring sBFC from "<<tV->name<<" to "<<origV->name<<endl;
#endif 
          if (tV->sField != NULL) {
            origV->sBFC = tV->sField->parentStruct;
#ifdef DEBUG_RP
            blame_info<<"sBFC added for "<<origV->name<<": "<<origV->sBFC->structName<<endl;
#endif
          }
          else {
#ifdef DEBUG_RP
            blame_info<<"Failed adding sBFC for "<<origV->name<<endl;
#endif
          }
          //Added by Hui 03/27/17
          //case like: origV<-(LD/RLS)-targetV<-GEP-tV, we can treat tV as origV's fields
          //We should have already add fields relation between tV and targetV in previous
          //resolvePointersForNode2 if targetV is 1-level ptr and is struct (TOBECONFIRM)
          if (origV != tV && origOpCode != GEP_BASE_OP) { //TODO: whether it's still needed 10/23/17
#ifdef DEBUG_RP
            blame_info<<"Adding "<<tV->name<<" to "<<origV->name<<" as a field(1)"<<endl;
#endif
            origV->fields.insert(tV);
            if (!tV->fieldUpPtr) //If not exist, we add
              tV->fieldUpPtr = origV;
            else if (tV->fieldUpPtr != origV) { //if existed and not equal, we quit
#ifdef DEBUG_RP     
              blame_info<<"fieldUpPtr already existed: "<<tV->name<<".fUP="<<tV->fieldUpPtr->name<<endl;
#endif
            }
          }
        }
        else { //if origV is not struct
          if (tVPtrLevel > 0) { //of course tVPtrLevel>0,since tV=GEP targetV, it's an address!
#ifdef DEBUG_RP
            blame_info<<"Adding GEP(1) "<<tV->name<<" "<<tV->isWritten<<" to "<<origV->name<<endl;
#endif        
            //03/27/17: if targetV = GEP orig, tV = GEP targetV, we don't add tV to origV's GEPs
            if (origOpCode != GEP_BASE_OP)
              origV->GEPs.insert(tV);
            //TODO: Check the below meaning 10/23/17      
            if (alias && tV->isWritten) {
#ifdef DEBUG_RP
              blame_info<<"Adding "<<tV->name<<" to list of resolvedLSSideEffects for "<<alias->name<<endl;
#endif
              alias->resolvedLSSideEffects.insert(tV);
            }
          }
            
          // we'll use this in cases where 2 loads replace a GEP
          if (targetV != origV) { //targetV = load origV;  tV = GEP targetV ...
#ifdef DEBUG_RP
            blame_info<<"Adding GEP(2) "<<tV->name<<" to load "<<targetV->name<<endl;
#endif 
            targetV->GEPs.insert(tV);
          }
        
          resolvePointersHelper2(targetV, origPLevel, tV, visited, tempPointers, alias, opCode);
        }
      }//opCode == GEP_BASE

      else if (opCode == RESOLVED_L_S_OP) {
#ifdef DEBUG_RP
        blame_info<<"R_L_S blame Source "<<tV->name<<" Dest "<<targetV->name<<endl;
#endif 
      }
    }//all in-edge of targetV

    //Now checking all out-edges of targetV
    boost::graph_traits<MyGraphType>::out_edge_iterator o_beg, o_end;
    o_beg = boost::out_edges(targetV->number, G).first;  // edge iterator begin
    o_end = boost::out_edges(targetV->number, G).second;// edge iterator end
    
    // iterate through the edges trying to find relationship between pointers
    for (; o_beg != o_end; o_beg++) {
      int opCode = get(get(edge_iore, G),*o_beg);
      if (opCode == Instruction::Store) {  
#ifdef DEBUG_RP
        blame_info<<"Here(2) "<<newPtrLevel<<" "<<targetV->name<<endl;
        // We have a write to the data section of the pointer
        blame_info<<"Vertex "<<targetV->name<<" is written(5)"<<endl;
        blame_info<<"Vertex "<<origV->name<<" is written(6)"<<endl;
#endif
        origV->isWritten = true;
        targetV->isWritten = true;

        //added by Hui 03/22/16 for cases: %1 = load a;  store %2, %1  ///
        //then %2 should be counted to a's child since it's the value to a's load
        //TODO: Whether we shoud keep or change the below? (like dataPtr?)
        /*NodeProps *storeValue = get(get(vertex_props,G), target(*o_beg,G));
          origV->children.insert(storeValue);
          storeValue->parents.insert(origV);*/
        //////////////////////////////////////////////////////////////////////
      }
    }
  } //newPtrLevel <= origPtrLevel
  
  else { // What to do with cases like: targetV = GEP origV, targetV
#ifdef DEBUG_RP
    blame_info<<"What's going on here? New pointer level is "
        <<newPtrLevel<<" and  old is "<<origPLevel<<endl;
#endif
  }
}


bool FunctionBFC::checkIfWritten2(NodeProps *currNode, set<int> &visited)
{
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", orig isWritten="<<currNode->isWritten<<endl;
#endif 
  
  if (visited.count(currNode->number) > 0)
    return currNode->isWritten; //changed from 0 to  currNode->isWritten
  visited.insert(currNode->number);
  
  short writeTotal = currNode->isWritten;
  set<NodeProps *>::iterator vec_vp_i;
  for (vec_vp_i = currNode->aliasesOut.begin(); vec_vp_i != currNode->aliasesOut.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
  }
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after aliasesOut, writeTotal="<<writeTotal<<endl;
#endif 

  for (vec_vp_i = currNode->aliasesIn.begin(); vec_vp_i != currNode->aliasesIn.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
  }
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after aliasesIn, writeTotal="<<writeTotal<<endl;
#endif 

    // Just for pids
    if (currNode->isPid) {
      // first make sure pid is written if the corresponding obj is written
      if (currNode->myObj) {
        if (currNode->myObj->isWritten) //Normally it's alreay written since it's a var
          currNode->isWritten = true;
      }
      // second we check its pidAliasesOut nodes  
    for (vec_vp_i = currNode->pidAliasesOut.begin(); vec_vp_i != currNode->pidAliasesOut.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
    }
#ifdef DEBUG_RP
    blame_info<<"In checkIfWritten for "<<currNode->name<<", after pidAliasesOut, writeTotal="<<writeTotal<<endl;
#endif
    }
  //Just for objs
    if (currNode->isObj) {
    for (vec_vp_i = currNode->objAliasesOut.begin(); vec_vp_i != currNode->objAliasesOut.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
    }
#ifdef DEBUG_RP
    blame_info<<"In checkIfWritten for "<<currNode->name<<", after objAliasesOut, writeTotal="<<writeTotal<<endl;
#endif 
    }
  
    //Just for isBlamedExternCallParam
    if (currNode->isBlamedExternCallParam) {
    for (vec_vp_i = currNode->blameesFromExFunc.begin(); vec_vp_i != currNode->blameesFromExFunc.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
    }
#ifdef DEBUG_RP
    blame_info<<"In checkIfWritten for "<<currNode->name<<", after blameesFromExFunc, writeTotal="<<writeTotal<<endl;
#endif 
    }

/*
    //special case for Barrier related typed variables (a barrier should alwasy be seen written and accounted for that line)
    if (currNode->llvm_inst) {
      Value *v = currNode->llvm_inst;
      if (isa<Instruction>(v) || isa<ConstantExpr>(v)) {
        Type *t = v->getType();
        const Type *origT = getPointedType(t);
        if (origT->getTypeID() == Type::StructTyID) {
      const llvm::StructType *st = cast<StructType>(origT);
          if (st->hasName()) { //conservative way for the Barrier-related type name as far as we know
      string st_name = st->getName().str(); //type name of v
            if (st_name.find("Barrier_") != string::npos || st_name.find("_Barrier") != string::npos) {
            //if (st_name.find("Barrier") != string::npos) {
              currNode->isWritten = true;
              writeTotal += currNode->isWritten; //= writeTotal+1
              }
            }
          }//type has name
        }//is struct type
      }//llvm_inst of currNode is inst or CE
    }//llvm_inst of currNode exists
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after check Barrier, writeTotal="<<writeTotal<<endl;
#endif 
*/        
    for (vec_vp_i = currNode->almostAlias.begin(); vec_vp_i != currNode->almostAlias.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
  }
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after almostAlias, writeTotal="<<writeTotal<<endl;
#endif 
  
  for (vec_vp_i = currNode->resolvedLS.begin(); vec_vp_i != currNode->resolvedLS.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
  }
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after resolvedLS, writeTotal="<<writeTotal<<endl;
#endif 
  
  for (vec_vp_i = currNode->fields.begin(); vec_vp_i != currNode->fields.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
  }
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after fields, writeTotal="<<writeTotal<<endl;
#endif 
  
  for (vec_vp_i = currNode->nonAliasStores.begin(); vec_vp_i != currNode->nonAliasStores.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
  }
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after nonAliasStores, writeTotal="<<writeTotal<<endl;
#endif 
  
  for (vec_vp_i = currNode->arrayAccess.begin(); vec_vp_i != currNode->arrayAccess.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
  }
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after arrayAccess, writeTotal="<<writeTotal<<endl;
#endif 

#ifdef ADD_MULTI_LOCALE
  for (vec_vp_i = currNode->GEPs.begin(); vec_vp_i != currNode->GEPs.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
  }
#ifdef DEBUG_RP
    blame_info<<"In checkIfWritten for "<<currNode->name<<", after GEPs, writeTotal="<<writeTotal<<endl;
#endif
#endif
  // We consider any param passed into a function as written, acting conservatively
  //  At runtime through transfer functions we can figure out if this is actually the case
  for (vec_vp_i = currNode->loads.begin(); vec_vp_i != currNode->loads.end(); vec_vp_i++) {
    NodeProps * v = (*vec_vp_i);
    if (!v) {
#ifdef DEBUG_ERROR    
    cerr<<"Null V in checkIfWritten\n";
#endif
    continue;
    }
    boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
    e_beg = boost::out_edges(v->number, G).first;    // edge iterator begin
    e_end = boost::out_edges(v->number, G).second;    // edge iterator end
    
    // iterate through the edges to find matching opcode
    for(; e_beg != e_end; ++e_beg) {
    int opCode = get(get(edge_iore, G),*e_beg);
      
    if (opCode == Instruction::Call || opCode == Instruction::Invoke)
      writeTotal++;
    }  
  }
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after loads, writeTotal="<<writeTotal<<endl;
#endif
    //Not sure whether we need this contribution
  // If it has GEPChildren, then if any of them is written, it should be considered written as well
  for (vec_vp_i = currNode->GEPChildren.begin(); vec_vp_i != currNode->GEPChildren.end(); vec_vp_i++) {
    writeTotal += checkIfWritten2((*vec_vp_i), visited);
  }
#ifdef DEBUG_RP
  blame_info<<"In checkIfWritten for "<<currNode->name<<", after GEPChildren, writeTotal="<<writeTotal<<endl;
#endif 

  if (writeTotal > 0)
      currNode->isWritten = true;
  return currNode->isWritten;
}


//At the very irst time this func was called, exitCand, currNode, exitV are all same
void FunctionBFC::resolveLocalAliases2(NodeProps * exitCand, NodeProps * currNode,
                    set<int> & visited, NodeProps * exitV)
{
#ifdef DEBUG_RP
  blame_info<<"In resolveLocalAliases2 for "<<exitCand->name<<"("<<exitCand->eStatus<<")";
  blame_info<<" "<<currNode->name<<"("<<currNode->eStatus<<")"<<endl;
#endif
  
  if (visited.count(currNode->number) > 0)
    return;
  visited.insert(currNode->number);
  
  set<NodeProps *>::iterator vec_vp_i;
  
  if (currNode->nStatus[LOCAL_VAR] || currNode->nStatus[LOCAL_VAR_ALIAS]) {
    for (vec_vp_i = currNode->aliasesOut.begin(); vec_vp_i != currNode->aliasesOut.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[LOCAL_VAR_ALIAS] = true;
        if (vp->pointsTo == NULL)
      vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveLocalAliases)(1) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif
      vp->exitV = exitV;
    }

    if (vp->nStatus[EXIT_VAR])
      return;
#ifdef DEBUG_RP
    blame_info<<"Inserting alias(3) straight up in "<<vp->name<<" into "<<exitCand->name<<endl;
#endif 
    exitCand->aliases.insert(vp);
    resolveLocalAliases2(exitCand, vp, visited, exitV);
    }
    
    for (vec_vp_i = currNode->aliasesIn.begin(); vec_vp_i != currNode->aliasesIn.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[LOCAL_VAR_ALIAS] = true;
    if (vp->pointsTo == NULL)
      vp->pointsTo = exitCand;
      
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveLocalAliases)(2) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
        
        //CHECK: WHY return if vp is exit var ?  Hui 02/04/16, because they were processed in resolveAliases
      if (vp->nStatus[EXIT_VAR]) {
#ifdef DEBUG_RP
          blame_info<<"Because vp: "<<vp->name<<" is EXIT_VAR, so we don't add alias between it and "<<exitCand->name<<endl;
#endif
          return;
        }
    exitCand->aliases.insert(vp);
    resolveLocalAliases2(exitCand, vp, visited, exitV);
    }
      
      for (vec_vp_i = currNode->almostAlias.begin(); vec_vp_i != currNode->almostAlias.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[LOCAL_VAR_A_ALIAS] = true;
      
    if (vp->pointsTo == NULL)
      vp->pointsTo = exitCand;  
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveLocalAliases)(3) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;          
    }
       
#ifdef DEBUG_RP
        blame_info<<"Inserting dfAlias(4) "<<vp->name<<"into set for "<<exitCand->name<<endl;
#endif 
        exitCand->dfAliases.insert(vp);
    vp->dfaUpPtr = exitCand;
    resolveLocalAliases2(exitCand, vp, visited, exitV);
    }
  }
  
  for (vec_vp_i = currNode->nonAliasStores.begin(); vec_vp_i != currNode->nonAliasStores.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[LOCAL_VAR_PTR] = true;
    
      if (vp->pointsTo == NULL)
    vp->pointsTo = currNode;  
      if (vp->exitV == NULL) {
#ifdef DEBUG_RP
    blame_info<<"PTRS__(resolveLocalAliases)(4) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
    vp->exitV = exitV;
    }
      
      vp->dpUpPtr = currNode;
      //Hui dataPtrs.insert test 12/04/15
      blame_info<<"dataPtrs.insert 1: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
    currNode->dataPtrs.insert(vp);
    resolveLocalAliases2(exitCand, vp, visited, exitV);
  }
  
  if (currNode->nStatus[LOCAL_VAR_FIELD] || currNode->nStatus[LOCAL_VAR_FIELD_ALIAS]) {
    for (vec_vp_i = currNode->aliasesOut.begin(); vec_vp_i != currNode->aliasesOut.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[LOCAL_VAR_FIELD_ALIAS] = true;
    if (vp->pointsTo == NULL)
      vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveLocalAliases)(5) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif
      vp->exitV = exitV;
    }
#ifdef DEBUG_RP
    blame_info<<"Inserting alias(4) straight up "<<vp->name<<" into "<<exitCand->name<<endl;
#endif 
    exitCand->aliases.insert(vp);
    if (currNode->nStatus[LOCAL_VAR_FIELD])
      resolveLocalAliases2(currNode, vp, visited, exitV);
    else if (currNode->nStatus[LOCAL_VAR_FIELD_ALIAS])
      resolveLocalAliases2(exitCand, vp, visited, exitV);  
    }
    
    for (vec_vp_i = currNode->aliasesIn.begin(); vec_vp_i != currNode->aliasesIn.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[LOCAL_VAR_FIELD_ALIAS] = true;
    if (vp->pointsTo == NULL)
      vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveLocalAliases)(6) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
#ifdef DEBUG_RP
      blame_info<<"Inserting alias(5) straight up "<<vp->name<<" into "<<exitCand->name<<endl;
#endif      
    exitCand->aliases.insert(vp);
    if (currNode->nStatus[LOCAL_VAR_FIELD])
      resolveLocalAliases2(currNode, vp, visited, exitV);
    else if (currNode->nStatus[LOCAL_VAR_FIELD_ALIAS])
      resolveLocalAliases2(exitCand, vp, visited, exitV);  
    }  
  }
  
  if (currNode->nStatus[LOCAL_VAR] || currNode->nStatus[LOCAL_VAR_FIELD] ||
    currNode->nStatus[LOCAL_VAR_FIELD_ALIAS]) {
    for (vec_vp_i = currNode->fields.begin(); vec_vp_i != currNode->fields.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[LOCAL_VAR_FIELD] = true;  
    if (vp->pointsTo == NULL)
      vp->pointsTo = exitCand;
    if (vp->exitV == NULL){
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveLocalAliases)(7) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    
        resolveLocalAliases2(vp, vp, visited, exitV);
    }
  }
  
  if (currNode->nStatus[LOCAL_VAR_ALIAS]) {
    for (vec_vp_i = currNode->fields.begin(); vec_vp_i != currNode->fields.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[LOCAL_VAR_FIELD_ALIAS] = true;
      
    if (vp->pointsTo == NULL)
      vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveLocalAliases)(8) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    //resolveLocalAliases2(currNode, vp, visited, exitV);
        //changed by Hui 01/20/16 from above
        resolveLocalAliases2(vp, vp, visited, exitV);
    }
  }
  
  for (vec_vp_i = currNode->arrayAccess.begin(); vec_vp_i != currNode->arrayAccess.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
#ifdef DEBUG_RP
    blame_info<<"VP AA - "<<vp->name<<endl;
#endif
    int out_d = out_degree(vp->number, G);
    if (out_d > 1) {
      vp->nStatus[LOCAL_VAR_PTR] = true;
    if (vp->pointsTo == NULL)
      vp->pointsTo = currNode;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveLocalAliases)(9) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif
      vp->exitV = exitV;
    }
    
        vp->dpUpPtr = currNode;
        blame_info<<"dataPtrs.insert 2: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
    currNode->dataPtrs.insert(vp);
    }  
  }
  
  for (vec_vp_i = currNode->GEPs.begin(); vec_vp_i != currNode->GEPs.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    // TODO: 10/30/10 investigate the out_d > 1 thing
    int out_d = out_degree(vp->number, G);
    if (out_d > 1) {
      vp->nStatus[LOCAL_VAR_PTR] = true;
      if (vp->pointsTo == NULL)
        vp->pointsTo = currNode;
      if (vp->exitV == NULL) {
#ifdef DEBUG_RP
        blame_info<<"PTRS__(resolveLocalAliases)(10) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif
        vp->exitV = exitV;
      }
      
      vp->dpUpPtr = currNode;
      blame_info<<"dataPtrs.insert 3: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
      currNode->dataPtrs.insert(vp);
    }  
  }
  
#ifdef DATAPTRS_FROM_FIELDS
  for (vec_vp_i = currNode->fields.begin(); vec_vp_i != currNode->fields.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    // TODO: 10/30/10 investigate the out_d > 1 thing
    int out_d = out_degree(vp->number, G);
    if (out_d > 1) {
      vp->nStatus[LOCAL_VAR_PTR] = true;
      
      if (vp->pointsTo == NULL)
      vp->pointsTo = currNode;
      if (vp->exitV == NULL) {
#ifdef DEBUG_RP
        blame_info<<"PTRS__(resolveLocalAliases)(10.2) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif
        vp->exitV = exitV;
      }
      
      vp->dpUpPtr = currNode;
      blame_info<<"dataPtrs.insert 3(fields): insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
      currNode->dataPtrs.insert(vp);
    }
  }
#endif
  
  for (vec_vp_i = currNode->resolvedLS.begin(); vec_vp_i != currNode->resolvedLS.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
#ifdef DEBUG_RP
    blame_info<<"RA (RLS_Local) - Current Node "<<currNode->name<<" Looking at "<<vp->name<<endl;
#endif   
    int out_d = out_degree(vp->number, G);
    if (out_d > 1 || vp->GEPs.size() == 0) {
#ifdef DEBUG_RP
      blame_info<<"RA (RLS_Local) - Adding "<<vp->name<<endl;
#endif 
      vp->nStatus[LOCAL_VAR_PTR] = true;  
      if (vp->pointsTo == NULL)
        vp->pointsTo = currNode;
      if (vp->exitV == NULL) {
#ifdef DEBUG_RP
        blame_info<<"PTRS__(resolveLocalAliases)(11) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
        vp->exitV = exitV;
      }
      
      vp->dpUpPtr = currNode;
      blame_info<<"dataPtrs.insert 4: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
      currNode->dataPtrs.insert(vp);  
    }
  }
  
  for (vec_vp_i = currNode->loads.begin(); vec_vp_i != currNode->loads.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    int out_d = out_degree(vp->number, G);
    if (out_d > 1 || vp->GEPs.size() == 0) {
      vp->nStatus[LOCAL_VAR_PTR] = true;
      if (vp->pointsTo == NULL)
        vp->pointsTo = currNode;
      if (vp->exitV == NULL) {
#ifdef DEBUG_RP
        blame_info<<"PTRS__(resolveLocalAliases)(11) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
        vp->exitV = exitV;
      }
      
      vp->dpUpPtr = currNode;
      blame_info<<"dataPtrs.insert 5: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
      currNode->dataPtrs.insert(vp);
    }
  }
#ifdef DEBUG_RP
  blame_info<<"Finishing up resolveLocalAliases for "<<exitCand->name<<" "<<currNode->name<<endl;
#endif 
}


void FunctionBFC::resolveAliases2(NodeProps * exitCand, NodeProps * currNode,
                  set<int> & visited, NodeProps * exitV)
{
#ifdef DEBUG_RP
  blame_info<<"In resolveAliases for "<<exitCand->name<<"("<<exitCand->eStatus<<")";
  blame_info<<" "<<currNode->name<<"("<<currNode->eStatus<<")"<<endl;
  
  blame_info<<"Node Props for curr node "<<currNode->name<<": ";
  for (int a = 0; a < NODE_PROPS_SIZE; a++)
    blame_info<<currNode->nStatus[a]<<" ";
  blame_info<<endl;
#endif   
  
  if (visited.count(currNode->number) > 0)
    return;
  visited.insert(currNode->number);
  
  set<NodeProps *>::iterator vec_vp_i;
  
  if (currNode->nStatus[EXIT_VAR]) {
    // Looking at the aliases in
    for (vec_vp_i = currNode->aliasesOut.begin(); vec_vp_i != currNode->aliasesOut.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[EXIT_VAR_ALIAS] = true;
    vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(1) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(1) - exitV "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
        continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(2) - exitV "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
#ifdef DEBUG_RP
      blame_info<<"Inserting alias(60) straight up "<<vp->name<<" into "<<exitCand->name<<endl;
#endif 
      exitCand->aliases.insert(vp);
        continue;
    }
      
    //Following: vp->exitV==NULL  
#ifdef DEBUG_RP
    blame_info<<"Inserting alias(6) straight up "<<vp->name<<" into "<<exitCand->name<<endl;
#endif 
    exitCand->aliases.insert(vp);
    
        resolveAliases2(exitCand, vp, visited, exitV);
    }
    
      // Looking at the aliases out
    for (vec_vp_i = currNode->aliasesIn.begin(); vec_vp_i != currNode->aliasesIn.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[EXIT_VAR_ALIAS] = true;
    vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(2) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(3) - exitV "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
          continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(4) - exitV "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
#ifdef DEBUG_RP
      blame_info<<"Inserting alias(61) straight up "<<vp->name<<" into "<<exitCand->name<<endl;
#endif         
        exitCand->aliases.insert(vp);
      continue;
    }  
#ifdef DEBUG_RP
        blame_info<<"Inserting alias(7) straight up "<<vp->name<<" into "<<exitCand->name<<endl;
#endif       
    //Following: vp->exitV==NULL  
    exitCand->aliases.insert(vp);
    resolveAliases2(exitCand, vp, visited, exitV);
    }
    
    for (vec_vp_i = currNode->almostAlias.begin(); vec_vp_i != currNode->almostAlias.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);  
    vp->nStatus[EXIT_VAR_A_ALIAS] = true;
    vp->pointsTo = exitCand; //what's pointsTo really mean?? and used for?
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(3) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(5) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(6) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
      continue;
    }
      
#ifdef DEBUG_RP
    blame_info<<"Inserting dfAlias(1) "<<vp->name<<"into set for "<<exitCand->name<<endl;
#endif 
    exitCand->dfAliases.insert(vp);
    vp->dfaUpPtr = exitCand;
    resolveAliases2(exitCand, vp, visited, exitV);
    }//almostAliases
    }//currNode is EXIT_VAR
  
  if (currNode->nStatus[EXIT_VAR_ALIAS]) {
    // Looking at the aliases in
    for (vec_vp_i = currNode->aliasesOut.begin(); vec_vp_i != currNode->aliasesOut.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);  
    if (!vp->nStatus[EXIT_VAR]) {
      vp->nStatus[EXIT_VAR_ALIAS] = true;
      vp->pointsTo = exitCand;
      if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(4) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
      }
      else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(7) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
        }  
      else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(8) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
      continue;
      }    
#ifdef DEBUG_RP
      blame_info<<"Inserting alias(8) straight up "<<vp->name<<" into "<<exitCand->name<<endl;
#endif 
          exitCand->aliases.insert(vp);
      resolveAliases2(exitCand, vp, visited, exitV);
    }
    }
    
      // Looking at the aliases out
    for (vec_vp_i = currNode->aliasesIn.begin(); vec_vp_i != currNode->aliasesIn.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    if (!vp->nStatus[EXIT_VAR]) {
      vp->nStatus[EXIT_VAR_ALIAS] = true;
      vp->pointsTo = exitCand;
      if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(5) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
      }
      else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(9) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
      }  
      else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(10) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
      continue;
      }    
#ifdef DEBUG_RP
          blame_info<<"Inserting alias(9) straight up "<<vp->name<<" into "<<exitCand->name<<endl;
#endif 
        exitCand->aliases.insert(vp);
      resolveAliases2(exitCand, vp, visited, exitV);
    }
    }
    
    for (vec_vp_i = currNode->almostAlias.begin(); vec_vp_i != currNode->almostAlias.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);  
    vp->nStatus[EXIT_VAR_A_ALIAS] = true;
    vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(6) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(11) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(12) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
        continue;
    }
      
#ifdef DEBUG_RP
      blame_info<<"Inserting dfAlias(2) "<<vp->name<<"into set for "<<exitCand->name<<endl;
#endif       
        exitCand->dfAliases.insert(vp);
    vp->dfaUpPtr = exitCand;
    resolveAliases2(exitCand, vp, visited, exitV);  
    }  
  }
  
  for (vec_vp_i = currNode->nonAliasStores.begin(); vec_vp_i != currNode->nonAliasStores.end(); vec_vp_i++) {
    NodeProps * vp = (*vec_vp_i);
    vp->nStatus[EXIT_VAR_PTR] = true;
    vp->pointsTo = currNode;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
    blame_info<<"PTRS__(resolveAliases)(7) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
    vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
    blame_info<<"PTRS__(resolveAliases)(13) "<<exitV->name<<" already there for ";
    blame_info<<vp->name<<endl;
#endif 
    continue;
    }  
    else {
#ifdef DEBUG_ERROR
    blame_info<<"PTRS__(resolveAliases)(14) "<<exitV->name<<" already there  (";
    blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
    continue;
    }  
  
      vp->dpUpPtr = currNode;  
      //Hui dataPtrs.insert test 12/04/15
      blame_info<<"dataPtrs.insert 6: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
    currNode->dataPtrs.insert(vp);
    resolveAliases2(exitCand, vp, visited, exitV);
  }
  
  if (currNode->nStatus[EXIT_VAR_FIELD] || currNode->nStatus[EXIT_VAR_FIELD_ALIAS]) {
    for (vec_vp_i = currNode->aliasesOut.begin(); vec_vp_i != currNode->aliasesOut.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[EXIT_VAR_FIELD_ALIAS] = true;
    vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(8) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(15) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(16) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif
      continue;
    }
#ifdef DEBUG_RP
    blame_info<<"Inserting alias(10) straight up "<<vp->name<<" into "<<currNode->name<<endl;
#endif       
    currNode->aliases.insert(vp);
    resolveAliases2(currNode, vp, visited, exitV);
    }
    
    for (vec_vp_i = currNode->aliasesIn.begin(); vec_vp_i != currNode->aliasesIn.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[EXIT_VAR_FIELD_ALIAS] = true;
    vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(9) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(17) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(18) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
        continue;
    }
#ifdef DEBUG_RP
    blame_info<<"Inserting alias(11) straight up "<<vp->name<<" into "<<exitCand->name<<endl;
#endif       

        currNode->aliases.insert(vp);
    resolveAliases2(currNode, vp, visited, exitV);
    }
      
      for (vec_vp_i = currNode->almostAlias.begin(); vec_vp_i != currNode->almostAlias.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(3) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(5) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(6) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
      continue;
    }
      
#ifdef DEBUG_RP
      blame_info<<"Inserting dfAlias(1) "<<vp->name<<"into set for "<<exitCand->name<<endl;
#endif 
    exitCand->dfAliases.insert(vp);
    vp->dfaUpPtr = exitCand;
    resolveAliases2(exitCand, vp, visited, exitV);  
    }  
  }
  
  if (currNode->nStatus[EXIT_VAR] || currNode->nStatus[EXIT_VAR_ALIAS] ||   //cond EXIT_VAR_ALIAS is added by Hui
        currNode->nStatus[EXIT_VAR_FIELD] || currNode->nStatus[EXIT_VAR_FIELD_ALIAS]) {
    for (vec_vp_i = currNode->fields.begin(); vec_vp_i != currNode->fields.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[EXIT_VAR_FIELD] = true;
        //TODO: change to be made by Hui 01/26/15 EVFA's field  should just be EVFA, not EVF
    vp->pointsTo = exitCand;
      if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(10) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(19) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR  
      blame_info<<"PTRS__(resolveAliases)(20) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
    }
    resolveAliases2(vp, vp, visited, exitV);
    }
  }
  
  if (currNode->nStatus[EXIT_VAR_ALIAS]) {
    for (vec_vp_i = currNode->fields.begin(); vec_vp_i != currNode->fields.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    vp->nStatus[EXIT_VAR_FIELD_ALIAS] = true;
    vp->pointsTo = exitCand;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(11) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(21) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(22) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
        continue;
    }
    
        resolveAliases2(currNode, vp, visited, exitV);
    }    
  }
  
  for (vec_vp_i = currNode->GEPs.begin(); vec_vp_i != currNode->GEPs.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
    
    // TODO: 10/30/10 investigate the out_d > 1 thing
    int out_d = out_degree(vp->number, G);
    if (out_d > 1) {
    vp->nStatus[EXIT_VAR_PTR] = true;
    vp->pointsTo = currNode;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(12) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(23) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(24) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
        continue;
    }
      
    vp->dpUpPtr = currNode;  
        //Hui dataPtrs.insert test 12/04/15
        blame_info<<"dataPtrs.insert 7: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
    currNode->dataPtrs.insert(vp);
    }
  }
  
#ifdef DATAPTRS_FROM_FIELDS2
  for (vec_vp_i = currNode->fields.begin(); vec_vp_i != currNode->fields.end(); vec_vp_i++) {
    NodeProps * vp = (*vec_vp_i);
    // TODO: 10/30/10 investigate the out_d > 1 thing
    int out_d = out_degree(vp->number, G);
      if (out_d > 1) {
    vp->nStatus[EXIT_VAR_PTR] = true;
    vp->pointsTo = currNode;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(12.2) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(23.2) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
          continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(24.2) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
        continue;
    }
      
    vp->dpUpPtr = currNode;  
        //Hui dataPtrs.insert test 12/04/15
        blame_info<<"dataPtrs.insert 7(fields): insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
    currNode->dataPtrs.insert(vp);
    }
  }
#endif
  
  for (vec_vp_i = currNode->arrayAccess.begin(); vec_vp_i != currNode->arrayAccess.end(); vec_vp_i++) {
    NodeProps * vp = (*vec_vp_i);
    int out_d = out_degree(vp->number, G);
    if (out_d > 1) {
    vp->nStatus[EXIT_VAR_PTR] = true;
    vp->pointsTo = currNode;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
        blame_info<<"PTRS__(resolveAliases)(13) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif   
          vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
        blame_info<<"PTRS__(resolveAliases)(25) "<<exitV->name<<" already there for ";
        blame_info<<vp->name<<endl;
#endif   
          continue;
    }  
    else {
#ifdef DEBUG_ERROR
        blame_info<<"PTRS__(resolveAliases)(26) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
        continue;
    }
      
    vp->dpUpPtr = currNode;
        //Hui dataPtrs.insert test 12/04/15
        blame_info<<"dataPtrs.insert 8: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
    currNode->dataPtrs.insert(vp);
    }  
  }
  
  for (vec_vp_i = currNode->resolvedLS.begin(); vec_vp_i != currNode->resolvedLS.end(); vec_vp_i++) {
    NodeProps *vp = (*vec_vp_i);
#ifdef DEBUG_RP
    blame_info<<"RA (RLS) - Current Node "<<currNode->name<<" Looking at "<<vp->name<<endl;
#endif 
    int out_d = out_degree(vp->number, G);
    if (out_d > 1 || vp->GEPs.size() == 0) {
    vp->nStatus[EXIT_VAR_PTR] = true;
    vp->pointsTo = currNode;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(15) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(29) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(30) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
      continue;
    }
      
    vp->dpUpPtr = currNode;
        //Hui dataPtrs.insert test 12/04/15
        blame_info<<"dataPtrs.insert 9: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
      currNode->dataPtrs.insert(vp);
    }
  }

  for (vec_vp_i = currNode->loads.begin(); vec_vp_i != currNode->loads.end(); vec_vp_i++) {
      NodeProps *vp = (*vec_vp_i);
    int out_d = out_degree(vp->number, G);
    if (out_d > 1 || vp->GEPs.size() == 0) {    
    vp->nStatus[EXIT_VAR_PTR] = true;
    vp->pointsTo = currNode;
    if (vp->exitV == NULL) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(14) - Assigning exitV "<<exitV->name<<" for "<<vp->name<<endl;
#endif 
      vp->exitV = exitV;
    }
    else if (vp->exitV == exitV) {
#ifdef DEBUG_RP
      blame_info<<"PTRS__(resolveAliases)(27) "<<exitV->name<<" already there for ";
      blame_info<<vp->name<<endl;
#endif 
      continue;
    }  
    else {
#ifdef DEBUG_ERROR
      blame_info<<"PTRS__(resolveAliases)(28) "<<exitV->name<<" already there  (";
      blame_info<<vp->exitV->name<<") for "<<vp->name<<endl;
#endif 
        continue;
    }
      
    vp->dpUpPtr = currNode;
        //Hui dataPtrs.insert test 12/04/15
        blame_info<<"dataPtrs.insert 10: insert to : "<<currNode->name<<" of: "<<vp->name<<endl;
        currNode->dataPtrs.insert(vp);
    }
  }
#ifdef DEBUG_RP
  blame_info<<"Finishing up resolveAliases for "<<exitCand->name<<" "<<currNode->name<<endl;
#endif 
}


void FunctionBFC::resolveArrays(NodeProps *origV, NodeProps *v, set<NodeProps *> &tempPointers)
{
  // At this point, we don't care if the pointer is written to or not,
  // we'll worry about that later
#ifdef DEBUG_ARRAYS
  blame_info<<"In resolveArrays: Inserting pointer(4) "<<v->name<<endl;
#endif
  tempPointers.insert(v);
  
  boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
  
  e_beg = boost::in_edges(v->number, G).first;// edge iterator begin
  e_end = boost::in_edges(v->number, G).second;// edge iterator end
  
  // iterate through the edges trying to find a Store between pointers
  for (; e_beg != e_end; ++e_beg) {
    int opCode = get(get(edge_iore, G),*e_beg);
    NodeProps *targetV = get(get(vertex_props,G), source(*e_beg,G));
    
    if (opCode == GEP_BASE_OP) {
#ifdef DEBUG_ARRAYS
        blame_info<<"Inserting arrayAccess for "<<origV->name<<" of "<<targetV->name<<"\n";
#endif
        origV->arrayAccess.insert(targetV);
#ifdef DEBUG_ARRAYS
        blame_info<<"Inserting pointer(5) "<<targetV->name<<endl;
#endif
        tempPointers.insert(targetV);
      
      boost::graph_traits<MyGraphType>::out_edge_iterator e_beg2, e_end2;
    
      e_beg2 = boost::out_edges(targetV->number, G).first;    // edge iterator begin
      e_end2 = boost::out_edges(targetV->number, G).second;    // edge iterator end  
      // iterate through the edges trying to find relationship between pointers
      for (; e_beg2 != e_end2; ++e_beg2) {
        int opCodeForField = get(get(edge_iore, G),*e_beg2);
      //blame_info<<"Opcode "<<opCodeForField<<endl;
      //int fNum = 0;
      if (opCodeForField == Instruction::Store) {
#ifdef DEBUG_ARRAYS
      blame_info<<"Vertex "<<v->name<<" is written(1)"<<endl;
#endif 
      v->isWritten = true;
      }
    }
      
    resolveArrays(origV, targetV, tempPointers);  
    }
    
      else if (opCode == Instruction::Store) {
#ifdef DEBUG_RP
    blame_info<<"How could it be possible ? "<<targetV->name<<endl;
#endif
    //tempPointers.insert(targetV);
    }
  }
}

void FunctionBFC::resolvePointersForNode2(NodeProps *v, set<NodeProps *> &tempPointers)
{
  if (v->resolved == true)
    return;
  else
    v->resolved = true;
  
#ifdef DEBUG_RP
  blame_info<<"\nIn resolvePointersForNode for "<<v->name<<endl;
#endif 
  int origPointerLevel = 0, collapsedPointerLevel = 0;
  const llvm::Type *origT,  *collapsedT;    
  
  if (v->llvm_inst != NULL) {
    if (isa<Instruction>(v->llvm_inst)) {
#ifdef DEBUG_RP
    blame_info<<v->name<<" is Instruction"<<endl;
#endif   
    Instruction *pi = cast<Instruction>(v->llvm_inst);  
    origT = pi->getType();  
    origPointerLevel = pointerLevel(origT,0);
    }
    else if (isa<ConstantExpr>(v->llvm_inst)) {
#ifdef DEBUG_RP
    blame_info<<v->name<<" is ConstantExpr"<<endl;
#endif   
      ConstantExpr *ce = cast<ConstantExpr>(v->llvm_inst);
    origT = ce->getType();
    origPointerLevel = pointerLevel(origT, 0);
    }
    else if (isa<ConstantAggregateZero> (v->llvm_inst)) {
#ifdef DEBUG_RP
    blame_info<<v->name<<" is ConstantAggregateZero"<<endl;
#endif   
      ConstantAggregateZero *caz = cast<ConstantAggregateZero>(v->llvm_inst);
    origT = caz->getType();
    origPointerLevel = pointerLevel(origT, 0);
    }
    else if (isa<ConstantArray> (v->llvm_inst)) {
#ifdef DEBUG_RP
    blame_info<<v->name<<" is ConstantArray"<<endl;
#endif   
    ConstantArray *ca = cast<ConstantArray>(v->llvm_inst);
    origT = ca->getType();
    origPointerLevel = pointerLevel(origT, 0);
    }
      //llvm_inst is CPN only for global vars
    else if (isa<ConstantPointerNull> (v->llvm_inst)) {
#ifdef DEBUG_RP
    blame_info<<v->name<<" is ConstantPointerNullVal, ptrlevel++"<<endl;
#endif
    ConstantPointerNull *cpn = cast<ConstantPointerNull>(v->llvm_inst);
    origT = cpn->getType();
    origPointerLevel = pointerLevel(origT, 0);    
    origPointerLevel++; //TC: WHY ++ ? make sure global vars processed as pointers
    }
    else if (v->isFormalArg){
#ifdef DEaBUG_RP
    blame_info<<v->name<<" is formal arg"<<endl;
#endif
    origT = v->llvm_inst->getType();  
    origPointerLevel = pointerLevel(origT,0);
    }
    else if (isa<ConstantStruct> (v->llvm_inst)) {
#ifdef DEBUG_RP
    blame_info<<"Leaving resolvePointersForNode for "<<v->name;
    blame_info<<"  -- ConstantStruct"<<endl;
#endif
    return;
    }
    else if (isa<ConstantVector> (v->llvm_inst)) {
#ifdef DEBUG_RP
    blame_info<<"Leaving resolvePointersForNode for "<<v->name;
    blame_info<<"  -- ConstantVectorVal"<<endl;
#endif
    return;      
    }  
    else {
#ifdef DEBUG_RP
    blame_info<<"Leaving resolvePointersForNode for "<<v->name;
#endif
    return;
    }
  }
    //v->llvm_inst == NULL
  else{
    return;
  }
  
    //if b = bitcast a, t; (here v=a, a->collapsed_inst = b->inst, 
    //if ptrLevel(b)>=1 and type(b)==array, then v(a) is bitCastArray
  bool bitCastArray = false;
  if (v->collapsed_inst != NULL) {
    bool proceed = true;
  
    if (isa<Instruction>(v->collapsed_inst)) {
      Instruction *pi = cast<Instruction>(v->collapsed_inst);  
      collapsedT = pi->getType();  
      collapsedPointerLevel = pointerLevel(collapsedT,0);
    }
    else if (isa<ConstantExpr>(v->collapsed_inst)) {
      ConstantExpr *ce = cast<ConstantExpr>(v->collapsed_inst);
      collapsedT = ce->getType();
      collapsedPointerLevel = pointerLevel(collapsedT, 0);
    }
    else if (isa<ConstantAggregateZero> (v->collapsed_inst)) {
      ConstantAggregateZero *caz = cast<ConstantAggregateZero>(v->collapsed_inst);
      collapsedT = caz->getType();
      collapsedPointerLevel = pointerLevel(collapsedT, 0);
    }
    else if (isa<ConstantArray> (v->collapsed_inst)) {
      ConstantArray *ca = cast<ConstantArray>(v->collapsed_inst);
      collapsedT = ca->getType();
      collapsedPointerLevel = pointerLevel(collapsedT, 0);
    }
    else {
      proceed = false;
    }
    
    if (proceed) { //for case that v came from a bitcast instruction
      string colTStr = returnTypeName(collapsedT, string(""));    
#ifdef DEBUG_RP
      blame_info<<"COL "<<collapsedPointerLevel<<" "<<colTStr<<endl;
#endif
      if (collapsedPointerLevel == 1 && colTStr.find("Array") != string::npos) {
        bitCastArray = true;
        
#ifdef DEBUG_RP
        blame_info<<v->name<<" has a bitcast array "<<v->collapsed_inst->getName().data()<<endl;
#endif
      }
    }
  }
  
  bool isGlobalField = false;
  
  NodeProps *fieldParent = v->fieldUpPtr;
  if (fieldParent != NULL) {
    if (fieldParent->isGlobal)
      isGlobalField = true;
  }
  
  string origTStr = returnTypeName(origT, string(""));
  
#ifdef DEBUG_RP
  blame_info<<v->name<<"'s pointer level is "<<origPointerLevel<<endl;
  blame_info<<v->name<<" has pointer type "<<origTStr<<endl;
#endif 
  if (origPointerLevel >1) //added by Hui 12/16/15, doesn't matter later
    v->isPtr = true;
  
  // A pointer level of 1 is just the location in memory for a standard primitive
  // We do make an exception here for structs since we treat fields and pointers
  // as interchangeable for this context
  
  // Changed 7/1/2010
  // and again 7/7/2010 
  // and again 7/12/2010 (bitCastArray, isGlobalField)
  if (origPointerLevel>1 || bitCastArray || isGlobalField || !v->storeLines.empty() 
            || (origPointerLevel==1 && origTStr.find("Array")!=string::npos)) {
    //  ||  (origPointerLevel == 1 && v->isBlamedExternCallParam))
    //if (origPointerLevel > 0 || v->storeLines.size() > 0 )
    //At this point, we don't care if the pointer is written to or not,
    //we'll worry about that later
#ifdef DEBUG_RP
    blame_info<<"Inserting pointer(1) "<<v->name<<endl;
#endif 
    tempPointers.insert(v);
    
    boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
    
    e_beg = boost::in_edges(v->number, G).first;  // edge iterator begin
    e_end = boost::in_edges(v->number, G).second;   // edge iterator end
    
    set<int> visited; // don't know if the way we're traversing the graph,
    // a loop is even possible, but just in case
    visited.insert(v->number);
    
    // iterate through the edges trying to find a Store between pointers
    for (; e_beg != e_end; ++e_beg) {
      int opCode = get(get(edge_iore, G),*e_beg);
      NodeProps *targetV = get(get(vertex_props,G), source(*e_beg,G));
      if (!targetV) continue;

      if (opCode == Instruction::Store) {  
#ifdef DEBUG_RP
        blame_info<<"STORE operation from "<<v->name<<" to "<<targetV->name<<endl;
#endif 
      }
      else if (opCode == GEP_BASE_OP) {
#ifdef DEBUG_RP
        blame_info<<"GEPB operation between "<<v->name<<" and "<<targetV->name<<endl;
#endif    
        //here, targetV is the field ptr,e.g. a = GEP b, 0, 1 then a=targetV
        boost::graph_traits<MyGraphType>::out_edge_iterator o_beg, o_end;
        o_beg = boost::out_edges(targetV->number, G).first;//edge iter begin
        o_end = boost::out_edges(targetV->number, G).second;//edge iter end
        
        bool circumVent = true;
        // iterate through the edges trying to find a Store between pointers
        for (; o_beg != o_end; ++o_beg) {
          int opCode2 = get(get(edge_iore, G),*o_beg);
          if (opCode2 == RESOLVED_L_S_OP)
          circumVent = false;
        }  
        
        if (circumVent) {
          v->GEPs.insert(targetV); //a = GEP b, then b.GEPs.insert(a)
#ifdef DEBUG_RP
          blame_info<<"Adding GEP(3) "<<targetV->name<<" to "<<v->name<<endl;
#endif 
          resolvePointersHelper2(v, origPointerLevel, targetV, visited, 
                            tempPointers, NULL, opCode);
        }//if circumvent  
        else {
#ifdef DEBUG_ERROR
          cerr<<"Was this supposed to happen?!"<<endl;
#endif 
        }
      } //if opCode==GEP_BASE

      else if (opCode == Instruction::Load ) {
        if (!targetV) continue;
#ifdef DEBUG_RP
        blame_info<<"LOAD operation between "<<v->name<<" and "<<targetV->name<<endl;
#endif
        boost::graph_traits<MyGraphType>::out_edge_iterator o_beg, o_end;
        o_beg = boost::out_edges(targetV->number, G).first;//edge iter begin
        o_end = boost::out_edges(targetV->number, G).second;//edge iter end
        
        bool circumVent = true;
        NodeProps *lsTarget = NULL;
        //added by Hui 04/26/16: for cases like: targetV = load v; call myFunc(targetV,..);
        //we need to add target to v's loads, further to its dataPtrs, then descParams, no matter
        //targetV has a R_L_S edge or not:TODO this modification is proven to be harmful, but we need a better way 
        bool isParam = false;

        for (; o_beg != o_end; ++o_beg) {
          int opCode2 = get(get(edge_iore, G),*o_beg);
          if (opCode2 == RESOLVED_L_S_OP) {
            circumVent = false;
            lsTarget = get(get(vertex_props,G), target(*o_beg,G));
          }
          else if (opCode2 == Instruction::Call || opCode2 == Instruction::Invoke || opCode2 == RESOLVED_EXTERN_OP) {
            isParam = true; //above last opCode added for blamed params that were fed to externFuncs after adjustment
          }
        }  

        if (circumVent) {
          v->loads.insert(targetV);
          resolvePointersHelper2(v, origPointerLevel, targetV, visited, tempPointers, NULL, opCode);
        }
        else {
#ifdef DEBUG_RP
          blame_info<<"Load not put in because there was a R_L_S from targetV "<<targetV->name<<endl;
#endif
#ifdef TEMP_FOR_MINIMD
          if (isParam) { //added by Hui 04/26/16, for
            v->loadForCalls.insert(targetV);
          } //TODO: old addition as 'loads' is proven to be harmful, study later to see if it's necessary
#endif
          if (lsTarget) {
#ifdef DEBUG_RP
            blame_info<<"R_L_S op between "<<lsTarget->name<<" and "<<targetV->name<<endl;
#endif 
            lsTarget->resolvedLS.insert(targetV);
            targetV->resolvedLSFrom.insert(lsTarget);
            resolvePointersHelper2(v, origPointerLevel, targetV, visited, tempPointers, lsTarget, opCode);
            //resolvePointersHelper2(lsTarget, origPointerLevel, targetV, visited, tempPointers);
          }
        }
      }

      else if (opCode == RESOLVED_L_S_OP) {
#ifdef DEBUG_RP
        blame_info<<"RLS operation between "<<v->name<<" and "<<targetV->name<<endl;
#endif 
        v->resolvedLS.insert(targetV);
        targetV->resolvedLSFrom.insert(v);
        resolvePointersHelper2(v, origPointerLevel, targetV, visited, tempPointers, NULL, opCode);
      }
    }//all in_edges of v
  }//if isPtr...

  else if (origTStr.find("Struct") != string::npos) { //TOCONTINUE: 01/30/16
    //At this point, we don't care if the pointer is written to or not,
    //we'll worry about that later
#ifdef DEBUG_RP
    blame_info<<"Inserting pointer(3) "<<v->name<<endl;
#endif 
    tempPointers.insert(v);
    boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
    
    e_beg = boost::in_edges(v->number, G).first;    // edge iterator begin
    e_end = boost::in_edges(v->number, G).second;    // edge iterator end
    
    // iterate through the edges trying to find a Store between pointers
    for (; e_beg != e_end; ++e_beg) {
      int opCode = get(get(edge_iore, G),*e_beg);
      //here,targetV is the coming node for v, not v itself
      NodeProps * targetV = get(get(vertex_props,G), source(*e_beg,G));
      if (!targetV) continue;
#ifdef DEBUG_RP
      blame_info<<"For the struct pointer: "<<v->name<<", targetV of this in_edge is "<<targetV->name<<endl;
#endif
      if (opCode == GEP_BASE_OP) {
        boost::graph_traits<MyGraphType>::out_edge_iterator e_beg2, e_end2;
        e_beg2 = boost::out_edges(targetV->number,G).first;//edge iter begin
        e_end2 = boost::out_edges(targetV->number,G).second;//edge iter  end
        
        // iterate through the edges trying to find relationship between pointers
        for (; e_beg2 != e_end2; ++e_beg2) {
          int opCodeForField = get(get(edge_iore, G),*e_beg2);
          int fNum = 0;
          if (opCodeForField >= GEP_S_FIELD_OFFSET_OP) {
            fNum = opCodeForField - GEP_S_FIELD_OFFSET_OP;//=which field
            if (isa<Value>(v->llvm_inst)) {
              Value *val = cast<Value>(v->llvm_inst);  
#ifdef DEBUG_RP
              blame_info<<"Calling structResolve for "<<v->name<<" of type "<<origTStr<<endl;
#endif 
              structResolve(val, fNum, targetV);//assign StructField to targetV
              break;
            }  
          }
          else if (opCodeForField == Instruction::Store) { //field is written
            targetV->isWritten = true;
            v->isWritten = true; //targetV is a filed of v
#ifdef DEBUG_RP
            blame_info<<"Vertices "<<targetV->name<<" "
                  <<v->name<<" are written(2)"<<endl;
#endif 
          }
        }
#ifdef DEBUG_RP
        blame_info<<"Transferring sBFC from(sf->ps) "<<targetV->name<<" to "<<v->name<<endl;
#endif 
        if (targetV->sField != NULL) {
          v->sBFC = targetV->sField->parentStruct;
#ifdef DEBUG_RP
          blame_info<<"sBFC added for "<<v->name<<": "<<v->sBFC->structName<<endl;
#endif
        }
        else {
#ifdef DEBUG_RP
          blame_info<<"Failed adding sBFC for "<<v->name<<endl;
#endif
        }

        if (v != targetV) { //condition added by Hui 01/20/16
#ifdef DEBUG_RP
          blame_info<<"Adding "<<targetV->name<<" to "<<v->name<<" as a field(2)"<<endl;
#endif
          v->fields.insert(targetV);
          targetV->fieldUpPtr = v;
        }
        //TOCHECK: why only Global's field is added to tempPointers for further check? 
        if (v->isGlobal) {
          tempPointers.insert(targetV);
        }
      }

#ifdef HUI_CHPL
      else if (opCode == Instruction::Store) {
        boost::graph_traits<MyGraphType>::in_edge_iterator e_beg2, e_end2;
        e_beg2 = boost::in_edges(targetV->number, G).first;
        e_end2 = boost::in_edges(targetV->number, G).second;

        for (; e_beg2 != e_end2; ++e_beg2) {
          int opCode2 = get(get(edge_iore, G), *e_beg2);
          NodeProps *targetV2 = get(get(vertex_props,G),source(*e_beg2,G));
          if (!targetV2) continue;
#ifdef DEBUG_RP
          blame_info<<"For "<<targetV->name<<", targetV2 of this in_edge is "<<targetV2->name<<endl;
#endif
          if (opCode2 == Instruction::Load) {
            boost::graph_traits<MyGraphType>::in_edge_iterator e_beg3, e_end3;
            e_beg3 = boost::in_edges(targetV2->number, G).first;
            e_end3 = boost::in_edges(targetV2->number, G).second;
            for (; e_beg3 != e_end3; ++e_beg3) {
              int opCode3 = get(get(edge_iore, G), *e_beg3);
              NodeProps *targetV3 = get(get(vertex_props, G), source(*e_beg3, G));
              if (!targetV3) continue;
#ifdef DEBUG_RP
              blame_info<<"For "<<targetV2->name<<", targetV3 of this in_edge is "<<targetV3->name<<endl;

#endif
              if (opCode3 == GEP_BASE_OP) {
                boost::graph_traits<MyGraphType>::out_edge_iterator e_beg4, e_end4;
                e_beg4 = boost::out_edges(targetV3->number,G).first;//edge iter begin
                e_end4 = boost::out_edges(targetV3->number,G).second;//edge iter  end
                // iterate through the edges trying to find relationship between pointers
                for (; e_beg4 != e_end4; ++e_beg4) {
                  int opCodeForField = get(get(edge_iore, G),*e_beg4);
                  int fNum = 0;
                  if (opCodeForField >= GEP_S_FIELD_OFFSET_OP) {
                    fNum = opCodeForField - GEP_S_FIELD_OFFSET_OP; 
                    if (isa<Value>(v->llvm_inst)) {
                      Value *val = cast<Value>(v->llvm_inst);  
#ifdef DEBUG_RP
                      blame_info<<"Calling structResolve for "<<v->name<<" of type "<<origTStr<<endl;
#endif                //TO BE CHECKED: 08/22/15 structResolve
                      structResolve(val, fNum, targetV3);//assign StructField to targetV3
                      break;
                    }
                  }
            
                  if (opCodeForField == Instruction::Store) {
                    targetV3->isWritten = true;
                    targetV2->isWritten = true;//tV3 is field of tV2
                    v->isWritten = true;//tV2 --RLS--> v
#ifdef DEBUG_RP
                    blame_info<<"Vertices "<<targetV3->name<<" "
                        <<targetV2->name<<" "<<v->name<<" are written(7)"<<endl;
#endif 
                  }
                }
#ifdef DEBUG_RP
                blame_info<<"Transferring sBFC from(sf->ps) "<<targetV3->name<<" to "<<v->name<<endl;
#endif 
                if (targetV3->sField != NULL) {
                  v->sBFC = targetV3->sField->parentStruct;
#ifdef DEBUG_RP
                  blame_info<<"sBFC added for "<<v->name<<": "
                            <<v->sBFC->structName<<endl;
#endif
                }
                else {
#ifdef DEBUG_RP
                  blame_info<<"Failed adding sBFC for "<<v->name<<endl;
#endif
                }

                if (v != targetV3) { //condition added by Hui 01/20/16
                  blame_info<<"Adding "<<targetV3->name<<" to "<<v->name<<" as a field(3)"<<endl;
                  v->fields.insert(targetV3);
                  targetV3->fieldUpPtr = v;
                }
                if (v->isGlobal) {
                  tempPointers.insert(targetV3);
                }
              }// if opCode3=GEP_BASE_OP
            }// for each in_edge of targetV2
          }// if opCode2=Load
        }// for each in_edge of targetV
      }// else if opCode=Store
#endif
    }// for each in_edge of v
  }// origTstr.find("Struct")
  
  //This case baraly happen since usually a variable with its type having 
  //"Array" has ptrlevel at least 1, so it's been dealt with in Case 1
  else if (origTStr.find("Array") != string::npos) { //ptrlevel == 0
    resolveArrays(v, v, tempPointers);
  }
    //added by Hui :currently doesn't give much help, so delete it, TO CHECK later
    /*
    else if(origPointerLevel ==1) { //we take care of the basic 1-level ptr
#ifdef DEBUG_RP
    blame_info<<"Inserting pointer(new) "<<v->name<<endl;
#endif 
    //pointers.insert(v);
    tempPointers.insert(v);
    
    boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
    
    e_beg = boost::in_edges(v->number, G).first;  // edge iterator begin
    e_end = boost::in_edges(v->number, G).second;   // edge iterator end
    
    set<int> visited; // don't know if the way we're traversing the graph,
    // a loop is even possible, but just in case
    visited.insert(v->number);
    
    // iterate through the edges trying to find a Store between pointers
    for(; e_beg != e_end; ++e_beg) {
      int opCode = get(get(edge_iore, G),*e_beg);
#ifdef DEBUG_RP
        blame_info<<"in_edge opCode = "<<opCode<<endl;
#endif 
      NodeProps *targetV = get(get(vertex_props,G), source(*e_beg,G));
      if (opCode == Instruction::Store) {  
#ifdef DEBUG_RP
        blame_info<<"STORE operation(new) between "<<v->name<<" and "<<targetV->name<<endl;
#endif 
      }
      else if (opCode == GEP_BASE_OP){
#ifdef DEBUG_RP
        blame_info<<"GEPB operation(new) between "<<v->name<<" and "<<targetV->name<<endl;
#endif          //here, targetV is the field ptr,e.g. a = GEP b, 0, 1 then a=targetV
        boost::graph_traits<MyGraphType>::out_edge_iterator o_beg, o_end;
        o_beg = boost::out_edges(targetV->number, G).first;//edge iter begin
        o_end = boost::out_edges(targetV->number, G).second;//edge iter end
        
        bool circumVent = true;
        // iterate through the edges trying to find a Store between pointers
        for(; o_beg != o_end; ++o_beg) {
          int opCode = get(get(edge_iore, G),*o_beg);
          if (opCode == RESOLVED_L_S_OP)
            circumVent = false;
        }  
        
        if (circumVent) {
          v->GEPs.insert(targetV); //a = GEP b, then b.GEPs.insert(a)
#ifdef DEBUG_RP
          blame_info<<"Adding GEP(new) "<<targetV->name<<" to load "<<v->name<<endl;
#endif 
          //resolvePointersHelper2(v, origPointerLevel, targetV, visited, tempPointers, NULL);
        }  
        else {
#ifdef DEBUG_ERROR
          cerr<<"Was this supposed to happen(new)?!"<<endl;
#endif 
        }
      }
      else if (opCode == Instruction::Load ) {
#ifdef DEBUG_RP
        blame_info<<"LOAD operation(new) between "<<v->name<<" and "<<targetV->name<<endl;
#endif
        boost::graph_traits<MyGraphType>::out_edge_iterator o_beg, o_end;
        o_beg = boost::out_edges(targetV->number, G).first;//edge iter begin
        o_end = boost::out_edges(targetV->number, G).second;//edge iter end
        
        bool circumVent = true;
        NodeProps *lsTarget = NULL;
        
        for(; o_beg != o_end; ++o_beg) {
          int opCode = get(get(edge_iore, G),*o_beg);
          if (opCode == RESOLVED_L_S_OP) {
            circumVent = false;
            lsTarget = get(get(vertex_props,G), target(*o_beg,G));
          }
        }  
        
        if (circumVent) {
          v->loads.insert(targetV);
          //resolvePointersHelper2(v, origPointerLevel, targetV, visited, tempPointers, NULL);
        }
        else {
#ifdef DEBUG_RP
          blame_info<<"Load not put in(new) because there was a R_L_S from targetV "<<targetV->name<<endl;
#endif 
          if (lsTarget) {
#ifdef DEBUG_RP
            blame_info<<"R_L_S op(new) between "<<lsTarget->name<<" and "<<targetV->name<<endl;
#endif 
            lsTarget->resolvedLS.insert(targetV);
            targetV->resolvedLSFrom.insert(lsTarget);
            //resolvePointersHelper2(v, origPointerLevel, targetV, visited, tempPointers, lsTarget);
          }
        }
      }
      else if (opCode == RESOLVED_L_S_OP) {
#ifdef DEBUG_RP
        blame_info<<"RLS operation(new) between "<<v->name<<" and "<<targetV->name<<endl;
#endif 
        v->resolvedLS.insert(targetV);
        targetV->resolvedLSFrom.insert(v);
        //resolvePointersHelper2(v, origPointerLevel, targetV, visited, tempPointers, NULL);
      }
    }
    }*/
  
#ifdef DEBUG_RP_SUMMARY
  
  blame_info<<endl;
  blame_info<<"For PTR "<<v->name<<endl;
  blame_info<<"Is Pointer "<<v->isPtr<<endl; //TC: Should it be set to 1?
  blame_info<<"Is Written "<<v->isWritten<<endl;
  set<NodeProps *>::iterator vec_vp_i;
  set<NodeProps *>::iterator set_vp_i;
  
  
  blame_info<<"Aliases in ";
  for (vec_vp_i = v->aliasesIn.begin(); vec_vp_i != v->aliasesIn.end(); vec_vp_i++) {
    blame_info<<(*vec_vp_i)->name<<" ";
  }
  blame_info<<endl;
  
  blame_info<<"Aliases out ";
  for (vec_vp_i = v->aliasesOut.begin(); vec_vp_i != v->aliasesOut.end(); vec_vp_i++) {
    blame_info<<(*vec_vp_i)->name<<" ";
  }
  blame_info<<endl;
  
  blame_info<<"Fields ";
  for (vec_vp_i = v->fields.begin(); vec_vp_i != v->fields.end(); vec_vp_i++) {
    blame_info<<(*vec_vp_i)->name<<" ";
  }
  blame_info<<endl;
  
  blame_info<<"GEPs ";
  for (vec_vp_i = v->GEPs.begin(); vec_vp_i != v->GEPs.end(); vec_vp_i++){
    blame_info<<(*vec_vp_i)->name<<" ";
  }
  blame_info<<endl;
  
  blame_info<<"Non-Alias-Stores ";
  for (vec_vp_i = v->nonAliasStores.begin(); vec_vp_i != v->nonAliasStores.end(); vec_vp_i++) {
    blame_info<<(*vec_vp_i)->name<<" ";
  }
  blame_info<<endl;
  
  blame_info<<"Loads ";
  for (vec_vp_i = v->loads.begin(); vec_vp_i != v->loads.end(); vec_vp_i++) {
    blame_info<<(*vec_vp_i)->name<<" ";
  }
  blame_info<<endl;
  
  blame_info<<"Almost Aliases ";
  for (vec_vp_i = v->almostAlias.begin(); vec_vp_i != v->almostAlias.end(); vec_vp_i++) {
    blame_info<<(*vec_vp_i)->name<<" ";
  }
  blame_info<<endl;
  
  blame_info<<"Resolved LS ";
  for (vec_vp_i = v->resolvedLS.begin(); vec_vp_i != v->resolvedLS.end(); vec_vp_i++) {
    blame_info<<(*vec_vp_i)->name<<" ";
  }
  blame_info<<endl;
  
  blame_info<<"StoresTo ";
  for (set_vp_i = v->storesTo.begin(); set_vp_i != v->storesTo.end(); set_vp_i++){
    blame_info<<(*set_vp_i)->name<<" ";
  }
  blame_info<<endl;
  
  blame_info<<"StoreLines"<<endl;
  set<int>::iterator set_i_i;
  for (set_i_i = v->storeLines.begin(); set_i_i != v->storeLines.end(); set_i_i++){
    blame_info<<*set_i_i<<" ";
  }
  blame_info<<endl;
  
#endif 
  
}

void FunctionBFC::resolvePointers2()
{
  graph_traits < MyGraphType >::edge_descriptor ed;
    property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
    graph_traits<MyGraphType>::vertex_iterator i, v_end;
  set<NodeProps *> tempPointers;
  
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    NodeProps *v = get(get(vertex_props, G),*i);
    //int v_index = get(get(vertex_index, G),*i);
    if (!v) {
#ifdef DEBUG_ERROR
    cerr<<"Null V in resolvePointers2\n";
#endif 
    continue;
    }
    // First we want to resolve the local variables and exit variables
    if (v->isLocalVar || v->isFormalArg || v->isGlobal) {
      resolvePointersForNode2(v, tempPointers);
    }
  }//end of for loop
  
  set<NodeProps *>::iterator set_vp_i;  
  for (set_vp_i = tempPointers.begin(); set_vp_i != tempPointers.end(); set_vp_i++)
    pointers.insert(*set_vp_i);
  
#ifdef DEBUG_RP
  blame_info<<"At this point the pointers are: ";
  for (set_vp_i = pointers.begin(); set_vp_i != pointers.end(); set_vp_i++)
    blame_info<<(*set_vp_i)->name<<" ";
  blame_info<<endl;
#endif 
  
  bool keepGoing = true;
  while (keepGoing) {
    tempPointers.clear();
    for (set_vp_i = pointers.begin(); set_vp_i != pointers.end(); set_vp_i++) {
    NodeProps *v = (*set_vp_i);
    resolvePointersForNode2(v, tempPointers);
    set<NodeProps *>::iterator vec_vp_i;
    for (vec_vp_i = v->fields.begin(); vec_vp_i != v->fields.end(); vec_vp_i++) {
       resolvePointersForNode2((*vec_vp_i), tempPointers);
    }
    }
    
    if (tempPointers.size() > 0) {
    for (set_vp_i = tempPointers.begin(); set_vp_i != tempPointers.end(); set_vp_i++)
      pointers.insert(*set_vp_i);
      
      keepGoing = true;
    }
    else {
    keepGoing = false;
    }
  }
#ifdef DEBUG_RP
  blame_info<<"At this point(2) the pointers are: ";
  for (set_vp_i = pointers.begin(); set_vp_i != pointers.end(); set_vp_i++)
    blame_info<<(*set_vp_i)->name<<" ";
  blame_info<<endl;
#endif 
  
  //Now assign the EXIT_VAR, EXIT_VAR_ALIAS, EXIT_VAR_PTR distinctions
  for (set_vp_i = pointers.begin(); set_vp_i != pointers.end(); set_vp_i++) {
    NodeProps *vp = (*set_vp_i);
    if ((vp->isFormalArg && !vp->isLocalVar) 
          || vp->isGlobal || vp->name.find("retval") != string::npos) {
      // First we need to make sure that it is actually written in at least
      // one of the aliases
      set<int> visited;
      if (vp->isWritten || checkIfWritten2(vp, visited)) {
#ifdef DEBUG_RP
      blame_info<<"For ev: "<<vp->name<<" one of the aliases was written to!"<<endl;
#endif 
      vp->isWritten = true;
      vp->nStatus[EXIT_VAR] = true;
      vp->exitV = vp;
      visited.clear();
      resolveAliases2(vp, vp, visited, vp);
      //visited.clear();
      //resolveDataPtrs(vp, vp, visited);
    }
    else {
#ifdef DEBUG_RP
      blame_info<<"For ev: "<<vp->name<<" one of the aliases was NOT written to, NO resolveAliases on it"<<endl;
#endif
    }
    }  
  }
  // Now assign the LOCAL_VAR, LOCAL_VAR_ALIAS, LOCAL_VAR_PTR distinctions
  for (set_vp_i = pointers.begin(); set_vp_i != pointers.end(); set_vp_i++) {
    NodeProps *vp = (*set_vp_i);
    if (vp->isLocalVar == true) {
    // First we need to make sure that it is actually written in at least
    // one of the aliases
    set<int> visited;
      
    if (vp->isWritten || checkIfWritten2(vp, visited)) {
#ifdef DEBUG_RP
      blame_info<<"For lv: "<<vp->name<<" one of the aliases was written to!"<<endl;
#endif 
      vp->isWritten = true;
      vp->nStatus[LOCAL_VAR] = true;
      
      if (vp->exitV == NULL)
      vp->exitV = vp;
      visited.clear();
      resolveLocalAliases2(vp, vp, visited, vp);
      //visited.clear();
      //resolveDataPtrs(vp, vp, visited);
    }
    else {
#ifdef DEBUG_RP
      blame_info<<"For "<<vp->name<<
              " not written to (that we know) but still going to resolve aliases"<<endl;
#endif        
      vp->nStatus[LOCAL_VAR] = true;
    
      if (vp->exitV == NULL)
      vp->exitV = vp;
      visited.clear();
      resolveLocalAliases2(vp, vp, visited, vp);  
    }//vp not written
    }//if isLocalVar == true
    }//for all pointers

  // Finally, assign the LOCAL_VAR_A_ALIAS for those that don't have it yet
  // Mainly, for those non-ptr local variables
  // Commented out on 04/11/16: storesTo shouldn't be included into dfAliases
    /*
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    NodeProps *v = get(get(vertex_props, G),*i);
    //int v_index = get(get(vertex_index, G),*i);
    if (!v) {
#ifdef DEBUG_ERROR
      cerr<<"Null V in resolvePointers2\n";
#endif 
      continue;
    }
    // First we want to resolve the local variables and function parameters
    if (v->isLocalVar) {
    //resolveLocalDFA(v, pointers); //removed by Hui 04/11/16: storesTo shouldn't be included into dfAliases
    }
  }*/
}

void FunctionBFC::againCheckIfWrittenForAllNodes()
{
#ifdef DEBUG_AGAINCEHCK
    blame_info<<"In againCheckIfWrittenForAllNodes"<<endl;
#endif
  graph_traits < MyGraphType >::edge_descriptor ed;
    property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
    graph_traits<MyGraphType>::vertex_iterator i, v_end;
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    NodeProps *v = get(get(vertex_props, G),*i);
    //int v_index = get(get(vertex_index, G),*i);
    if (!v) {
#ifdef DEBUG_AGAINCHECK
     blame_info<<"Null V in againCheckIfWrittenForAllNodes\n";
#endif 
    continue;
    }
    
      int origPointerLevel = 0;
      const llvm::Type *origT;
      if (isa<Instruction>(v->llvm_inst)) {
    Instruction *pi = cast<Instruction>(v->llvm_inst);  
    origT = pi->getType();  
    origPointerLevel = pointerLevel(origT,0);
    }
    else if (isa<ConstantExpr>(v->llvm_inst)) {
      ConstantExpr *ce = cast<ConstantExpr>(v->llvm_inst);
    origT = ce->getType();
    origPointerLevel = pointerLevel(origT, 0);
    }
      else if (v->isFormalArg) {
        origT = v->llvm_inst->getType();
        origPointerLevel = pointerLevel(origT, 0);
      }
        
      if (origPointerLevel >=1) { //should we only care about >1 (src-level ptr)?
        set<int> visited;
        bool writtenTo = checkIfWritten2(v, visited);
      }
    }
}

void FunctionBFC::resolveLocalDFA(NodeProps * v, set<NodeProps *> & pointers)
{
  set<NodeProps *>::iterator set_vp_i;
  
  for (set_vp_i = v->storesTo.begin(); set_vp_i != v->storesTo.end(); set_vp_i++)
  {
    NodeProps *vp = *set_vp_i;
    if (!vp) continue;

    if (vp->storeLines.size() == 0)
      continue;
    
    if (!vp->nStatus[EXIT_VAR_A_ALIAS])
    {
      //pointers.insert(vp);
      vp->nStatus[LOCAL_VAR_A_ALIAS] = true;
      
      #ifdef DEBUG_RP
      blame_info<<"Inserting dfAlias(3) "<<vp->name<<" into set for "<<v->name<<endl;
      #endif 
      
      v->dfAliases.insert(vp);
      vp->dfaUpPtr = v;
#ifdef DEBUG_RP
      blame_info<<"Vertex "<<v->name<<" is written(3)"<<endl;
#endif 
      v->isWritten = true; //changed from vp to v, by Hui 04/11/16: we can't say vp->isWritten=true from here
    }
  }
}

//not used anymore 10/23/17
void FunctionBFC::collapseAutoCopyPairs()
{
  vector<CollapsePair *>::iterator vec_cp_i;
  
  for (vec_cp_i = autoCopyCollapsePairs.begin(); vec_cp_i != autoCopyCollapsePairs.end(); vec_cp_i++) {
    CollapsePair *cp = *vec_cp_i;
#ifdef DEBUG_GRAPH_COLLAPSE
    blame_info<<"Transferring CAP edges from "<<cp->collapseVertex->name<<" to "<<cp->destVertex->name<<endl;
#endif 
    transferEdgesAndDeleteNode(cp->collapseVertex, cp->destVertex);
  }
}

void FunctionBFC::collapseRedundantFields()
{
  vector<CollapsePair *>::iterator vec_cp_i;
  
  for (vec_cp_i = collapsePairs.begin(); vec_cp_i != collapsePairs.end(); vec_cp_i++) {
    CollapsePair *cp = *vec_cp_i;
#ifdef DEBUG_GRAPH_COLLAPSE
    blame_info<<"Transferring CRF edges from field "<<cp->collapseVertex->name<<" to "<<cp->destVertex->name;
    blame_info<<" for "<<cp->nameFieldCombo<<endl;
#endif 
    transferEdgesAndDeleteNode(cp->collapseVertex, cp->destVertex, false, true);
  }
}


void FunctionBFC::resolveDataReads()
{
#ifdef DEBUG_CFG
  blame_info<<"Before assignBBGenKIll"<<endl;
#endif 
  cfg->assignPTRBBGenKill();
  
#ifdef DEBUG_CFG
  blame_info<<"Before reachingDefs"<<endl;
#endif 
  cfg->reachingPTRDefs();
  
#ifdef DEBUG_CFG
  blame_info<<"Before calcStoreLines"<<endl;
#endif 
  cfg->calcPTRStoreLines();
  
#ifdef DEBUG_CFG
  blame_info<<"Before printCFG"<<endl;
  printCFG();
#endif 
  
}



void FunctionBFC::resolveTransitiveAliases()
{
    graph_traits<MyGraphType>::vertex_iterator i, v_end;
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    NodeProps *v = get(get(vertex_props, G),*i);
    if (!v) {
#ifdef DEBUG_ERROR      
    cerr<<"Null V in resolveTransitiveAlias\n";
#endif 
    continue;
    }
    set<NodeProps *>::iterator set_vp_i, set_vp_i2;
    for (set_vp_i = v->aliases.begin(); set_vp_i!=v->aliases.end(); set_vp_i++){
    NodeProps *al1 = *set_vp_i;
    for (set_vp_i2 = set_vp_i; set_vp_i2 != v->aliases.end(); set_vp_i2++) {
        NodeProps *al2 = *set_vp_i2;
      if (al1 == al2 )
      continue;
        
      if (al1 != v) {
#ifdef DEBUG_RP
      blame_info<<"Transitive alias between "<<al1->name<<" and "<<al2->name<<endl;
#endif 
      al1->aliases.insert(al2);
      }
      if (al2 != v) {
#ifdef DEBUG_RP
      blame_info<<"Transitive alias(2) between "<<al2->name<<" and "<<al1->name<<endl;
#endif 
      al2->aliases.insert(al1);
          }
    }
    }
  }
}

// TODO: Probably should account for the fact that a field may be aliased to more than one thing
void FunctionBFC::resolveFieldAliases()
{
    graph_traits<MyGraphType>::vertex_iterator i, v_end;

    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    NodeProps *v= get(get(vertex_props, G),*i);
    if (!v){
#ifdef DEBUG_ERROR      
    cerr<<"Null V in resolveFieldAliases\n";
#endif 
    continue;
    }
    
    if (v->nStatus[EXIT_VAR_FIELD_ALIAS]) {
    if (v->fieldUpPtr != NULL){
#ifdef DEBUG_RP
          blame_info<<"IMPORTANT EVFA "<<v->name<<" alread has fUpPtr "
            <<v->fieldUpPtr->name<<endl;
#endif 
          return;
    }
#ifdef DEBUG_RP
        blame_info<<"In resolveFieldAliases for EVFA: "<<v->name<<endl;
#endif 
    set<NodeProps *>::iterator set_vp_i, set_vp_i2;
      for (set_vp_i = v->aliases.begin(); set_vp_i != v->aliases.end(); 
             set_vp_i++) {
      NodeProps *al = *set_vp_i;
      // we might be our own alias probably should make it never happens
      // TODO: we should never be our own alias
      if (al == v)
      continue;      
      if (al->nStatus[EXIT_VAR_FIELD] && al->fieldUpPtr != NULL && 
              al->fieldUpPtr != v) {//last cond added by Hui 01/20/16
      v->fieldUpPtr = al->fieldUpPtr;
      v->fieldAlias = al; //TC: you can have >=1 alias right ?
          
#ifdef DEBUG_RP
            blame_info<<"Adding "<<v->name<<" to "<<al->name<<"'s fieldUpPtr:"
              <<al->fieldUpPtr->name<<" as a field(4)"<<endl;
#endif 
            al->fieldUpPtr->fields.insert(v);
        v->nStatus[EXIT_VAR_FIELD] = true;
#ifdef DEBUG_RP
      blame_info<<"Making EVFA variable an EVF for "<<v->name<<" to "
              <<v->fieldUpPtr->name<<endl;
#endif 
            break; //once we found fieldAlias we quit for loop 
      }  
    }
    }
    
      else if (v->nStatus[LOCAL_VAR_FIELD_ALIAS]) {
    if (v->fieldUpPtr != NULL) {
#ifdef DEBUG_RP
          blame_info<<"IMPORTANT LVFA "<<v->name<<" alread has fUpPtr "
            <<v->fieldUpPtr->name<<endl;
#endif 
          return;
    }
#ifdef DEBUG_RP  
        blame_info<<"In resolveFieldAliases for LVFA: "<<v->name<<endl;
#endif
        set<NodeProps *>::iterator set_vp_i, set_vp_i2;
      for (set_vp_i = v->aliases.begin(); set_vp_i != v->aliases.end(); 
             set_vp_i++) {
      NodeProps *al = *set_vp_i;
        //we might be our own alias probably should make never happens
      // TODO: we should never be our own alias
      if (al == v)
      continue;
          if (al->nStatus[LOCAL_VAR_FIELD] && al->fieldUpPtr != NULL && 
              al->fieldUpPtr != v) {//last cond added by Hui 01/20/16
      v->fieldUpPtr = al->fieldUpPtr;
      v->fieldAlias = al; //TC: what if v has >1 alias ?
          
#ifdef DEBUG_RP
            blame_info<<"Adding "<<v->name<<" to "<<al->name<<"'s fieldUpPtr:"
              <<al->fieldUpPtr->name<<" as a field(5)"<<endl;
#endif 
            al->fieldUpPtr->fields.insert(v);
      v->nStatus[LOCAL_VAR_FIELD] = true;
#ifdef DEBUG_RP
      blame_info<<"Making LVFA variable an LVF for "<<v->name<<" to "
              <<v->fieldUpPtr->name<<endl;
#endif
            break; //once we found fieldAlias we quit the for loop
      }   
    }
    }
    }//for all vertices
}

// Truncate genGraph for internal module functions
void FunctionBFC::genGraphTrunc(ExternFuncBFCHash &efInfo)
{
    // Create graph with the exact size of the number of symbols for the function
    MyGraphType G_new(variables.size()); //variables stores all the blame nodes
    G.swap(G_new); //switch G and G_new
  
    // Create property map for node properies for each register
    property_map<MyGraphType, vertex_props_t>::type props = get(vertex_props, G);  
  
    // Create property map for edge properties for each blame relationship
    property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
    RegHashProps::iterator begin, end;
    /* Iterate through variables and label them with their node IDs (integers)
   -> first is const char * associated with their name
   -> second is NodeProps * for node
  */
    for (begin = variables.begin(), end = variables.end(); begin != end; begin++) {
    string name(begin->first);
    NodeProps *v = begin->second;
      if (v) {
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"Putting node "<<v->number<<"("<<name<<") into graph"<<endl;
#endif 
    put(props, v->number, v);
      }
      else {
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"Weird: not putting node since NodeProps is empty "<<name<<endl;
#endif 
      }
  }
  //every block has a iSet, is a set of implicit blame nodes:Condition Names
    set<const char*, ltstr> iSet;  
  
    //int varCount = 0, currentLineNum = 0;
  
    int currentLineNum = 0;
#ifdef DEBUG_GRAPH_BUILD
  blame_info<<"Starting to Gen Edges "<<endl;
#endif 
    // We iterate through all the instructions again to create the edges (explicit and implicit)
    // generated between the registers  
  // Some times there are duplicate calls on lines, we want to make sure we don't reuse one
  set<NodeProps *> seenCall;
    for (Function::iterator b = func->begin(), be = func->end(); b != be; ++b) {
      iSet = iReg[(*b).getName().str()];
      for (BasicBlock::iterator i = (*b).begin(), ie = (*b).end(); i != ie; ++i) {
        //cout<<"CurrentLineNum is "<<currentLineNum<<endl;
        genEdges(&*i, iSet, props, edge_type, currentLineNum, seenCall);
      }
    }
  
#ifdef DEBUG_GRAPH_BUILD
  blame_info<<"Finished generating edges "<<endl;
#endif 
  
  // TODO: Make this a verbose or runtime decision
    //printDotFiles("_noCompress.dot", false);
    if (getenv("GEN_DOT") != NULL) {
    printDotFiles("_noCompressImp.dot", true);
    }
  
    collapseGraph(); 

    //added by Hui 05/09/16
    //collapseAutoCopyPairs();
  
#ifdef ENABLE_FORTRAN
  collapseRedundantFields();
#endif 
    if (getenv("GEN_DOT") != NULL) {
    printDotFiles("_afterCompressImp.dot", true);
    }
  //Make the edges incoming to nodes that have stores based on
  //the control flow
  resolveStores();  
  //get all resolvedLS
    adjustMainGraph();
  
  identifyExternCalls(efInfo); 
  
  resolvePointers2();
  
    // the alias of my alias is my friend
  // this needs to be after resolvePointers2
  resolveTransitiveAliases(); 
  
  // make field aliases associated with the proper local variable/exit variable in terms of fieldUpPtrs
  resolveFieldAliases();
  
    //added by Hui 05/08/16: to make up for all important registers
    againCheckIfWrittenForAllNodes();

  // Most stuff concentrates on data writes, we need to figure out
  // for any given data read the reaching definition for the data write
  // that affects it and factor that into things 
  //resolveDataReads();
#ifdef DEBUG_GRAPH_BUILD
  blame_info<<"Calling DBHL(trunc)"<<endl; 
#endif 
  determineBFCHoldersTrunc();
  
}

// Normal genGraph for user functions
void FunctionBFC::genGraph(ExternFuncBFCHash &efInfo)
{
  // Create graph with the exact size of the number of symbols for the function
  MyGraphType G_new(variables.size()); //variables stores all the blame nodes
  G.swap(G_new); //switch G and G_new
  
  // Create property map for node properies for each register
  property_map<MyGraphType, vertex_props_t>::type props = get(vertex_props, G);  
  
  // Create property map for edge properties for each blame relationship
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
  RegHashProps::iterator begin, end;
  /* Iterate through variables and label them with their node IDs (integers)
   -> first is const char * associated with their name
   -> second is NodeProps * for node
  */
  for (begin = variables.begin(), end = variables.end(); begin != end; begin++) {
    string name(begin->first);
    NodeProps *v = begin->second;
    if (v) {
#ifdef DEBUG_GRAPH_BUILD
      blame_info<<"Putting node "<<v->number<<" ("<<v->name<<") into graph"<<endl;
#endif 
      put(props, v->number, v);
    }
    else {
#ifdef DEBUG_GRAPH_BUILD
      blame_info<<"Weird: not putting node since NodeProps is empty "<<name<<endl;
#endif 
    }
  }
  //every block has a iSet, is a set of implicit blame nodes:Condition Names
  set<const char*, ltstr> iSet;  
  
  //int varCount = 0, currentLineNum = 0;
  int currentLineNum = 0;
#ifdef DEBUG_GRAPH_BUILD
  blame_info<<"Starting to Gen Edges "<<endl;
#endif 
  // We iterate through all the instructions again to create the edges (explicit and implicit)
  // generated between the registers  
  // Some times there are duplicate calls on lines, we want to make sure we don't reuse one
  set<NodeProps *> seenCall;
  for (Function::iterator b = func->begin(), be = func->end(); b != be; ++b) {
    iSet = iReg[(*b).getName().str()];
    for (BasicBlock::iterator i = (*b).begin(), ie = (*b).end(); i != ie; ++i) {
      //cout<<"CurrentLineNum is "<<currentLineNum<<endl;
      genEdges(&*i, iSet, props, edge_type, currentLineNum, seenCall);
    }
  }
  
#ifdef DEBUG_GRAPH_BUILD
  blame_info<<"Finished generating edges "<<endl;
#endif 
  
  // TODO: Make this a verbose or runtime decision
  //printDotFiles("_noCompress.dot", false);
  if (getenv("GEN_DOT") != NULL) {
    printDotFiles("_noCompressImp.dot", true);
  }

  collapseGraph(); 

  //added by Hui 05/09/16
  //collapseAutoCopyPairs();
 
#ifdef ENABLE_FORTRAN
  collapseRedundantFields();
#endif 
  if (getenv("GEN_DOT") != NULL) {
    printDotFiles("_afterCompressImp.dot", true);
  }
  //Has to be called before resolveStores,for pid, NOT used for extern funcs! 03/06/17
  //resolvePidAliases();
  //resolvePPA(); //keep potential pidAliases for EVs
  //Make the edges incoming to nodes that have stores based on
  //the control flow
  resolveStores();  
  //get all resolvedLS
  adjustMainGraph();
  
  identifyExternCalls(efInfo); 
  
  resolvePointers2(); //TODO: START FROM HERE 10/23/17
  
  // the alias of my alias is my friend
  // this needs to be after resolvePointers2
  resolveTransitiveAliases(); 
  
  // make field aliases associated with the proper local variable/exit variable in terms of fieldUpPtrs
  resolveFieldAliases();
  
  // For obj, we put it because we wanna take advantage of aliases(may move upward) 03/05/17, NOT used for extern funcs!
  //resolveObjAliases();
    
  //added by Hui 05/08/16: to make up for all important registers
  againCheckIfWrittenForAllNodes();
  
  // Most stuff concentrates on data writes, we need to figure out
  // for any given data read the reaching definition for the data write
  // that affects it and factor that into things 
  //resolveDataReads();
#ifdef DEBUG_GRAPH_BUILD
  blame_info<<"Calling DBHL "<<endl; 
#endif
  // Add EVs for args that are pid value
  //recheckExitVars(); //For Chapel1.15, Pid arg isn't integer, so it has been added into EVs already
  determineBFCHoldersLite();
  
  resolveCallsDomLines(); 
  
#ifdef DEBUG_GRAPH_BUILD
  blame_info<<"Finished DBHL , now going to print _trunc.dot file "<<endl;
#endif
  if (getenv("GEN_DOT") != NULL) {
    printTruncDotFiles("_trunc.dot", true);
  }
}


void FunctionBFC::printCFG()
{
  cfg->printCFG(blame_info);
}


void FunctionBFC::resolveStores()
{
    property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
    //bool inserted;
    graph_traits < MyGraphType >::edge_descriptor ed;
    graph_traits<MyGraphType>::vertex_iterator i, v_end; 
  
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    NodeProps *v = get(get(vertex_props, G),*i);
    int v_index = get(get(vertex_index, G),*i);
    
    if (!v) {
#ifdef DEBUG_ERROR      
    cerr<<"Null V in resolveStores\n";
#endif 
    continue;
    }  
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"In resolveStores for "<<v->name<<endl;
#endif 
    boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
    
    e_beg = boost::out_edges(v_index, G).first;    // edge iterator begin
    e_end = boost::out_edges(v_index, G).second;    // edge iterator end  
    //int storeCount = 0;  
      //store a, b
    set<NodeProps *> stores;  //store value: a
    set<NodeProps *> storeSources; //value holder: b
    
    // iterate through the edges to find matching opcode
    for (; e_beg != e_end; ++e_beg) {
      int opCode = get(get(edge_iore, G),*e_beg);
    if (opCode == Instruction::Store) {
        NodeProps *sourceV = get(get(vertex_props,G), source(*e_beg,G));//v==sourceV since it's out_edge
        NodeProps *targetV = get(get(vertex_props,G), target(*e_beg,G));
        if (!targetV) continue;

#ifdef DEBUG_GRAPH_BUILD
      blame_info<<"Store between "<<sourceV->name<<" and "<<targetV->name<<endl;
      blame_info<<"Vertex "<<sourceV->name<<" is written(4)"<<endl;
#endif 
#ifdef DEBUG_LINE_NUMS
      blame_info<<"Inserting line number(1a) "<<targetV->line_num<<" to "<<sourceV->name<<endl;
#endif 
        sourceV->lineNumbers.insert(targetV->line_num);
          //we don't consider 1st store for argHolder as a written since it's just copy the param
          //but we still treat it as a store to the argHolder for now 10/23/17
          if (!(sourceV->name == (targetV->name + ".addr")))
        sourceV->isWritten = true;
                        
#ifdef DEBUG_GRAPH_BUILD
          blame_info<<"We have a storeFrom inserted: "<<targetV->name<<"->storeFrom="<<sourceV->name<<endl;
#endif 
          targetV->storeFrom = sourceV;
      sourceV->storesTo.insert(targetV);
      stores.insert(targetV);
      storeSources.insert(sourceV);
    }
    }
    
      //Since resolveDataRead is no longer used, so we put all stores into relevantInstructions
      //no singleStores anymore !!! storeVP->fbb->singleStores.push_back(storeVP) if stores.size()==1
    if (stores.size() >= 1) {//store a *b, then a is in stores, b is in storesSource..
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"Stores for V - "<<v->name<<" number "<<stores.size()<<endl;
#endif 
    set<NodeProps *>::iterator set_vp_i = stores.begin();
    for ( ; set_vp_i != stores.end(); set_vp_i++) {
      NodeProps *storeVP = (*set_vp_i);
      if (storeVP->fbb == NULL)
      continue;

#ifdef DEBUG_GRAPH_BUILD
      blame_info<<"Insert relevantInst - "<<storeVP->name<<" in fbb "<<storeVP->fbb->getName()<<endl;
#endif 
          storeVP->fbb->relevantInstructions.push_back(storeVP);
    }
    }
  }//for all vertices

#ifdef DEBUG_CFG
  blame_info<<"Before CFG sort"<<endl;
#endif 
  cfg->sortCFG(); //for each BB, sort the edges
  
#ifdef DEBUG_CFG
  blame_info<<"Before assignBBGenKIll"<<endl;
#endif 
  cfg->assignBBGenKill();
  
#ifdef DEBUG_CFG
  blame_info<<"Before reachingDefs"<<endl;
#endif 
  cfg->reachingDefs();
  
#ifdef DEBUG_CFG
  blame_info<<"Before calcStoreLines"<<endl;
#endif 
  cfg->calcStoreLines();
  
#ifdef DEBUG_CFG
  blame_info<<"Before printCFG"<<endl;
  printCFG();
#endif 
}


void FunctionBFC::adjustMainGraph()
{
  bool inserted;
    graph_traits < MyGraphType >::edge_descriptor ed;
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  graph_traits<MyGraphType>::vertex_iterator i, v_end; 
  
  for (tie(i,v_end) = vertices(G); i != v_end; ++i)  {
    NodeProps *vp = get(get(vertex_props, G),*i);
    // We have stores coming in
    if (vp && vp->storesTo.size() > 0) { //eg. store a1 b; store a2 b (b=vp)
    boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
      
    e_beg = boost::in_edges(vp->number, G).first;  // edge iterator begin
    e_end = boost::in_edges(vp->number, G).second;  // edge iterator end
      
    set<int> deleteMe;
    // iterate through the edges to find matching opcode
    for (; e_beg != e_end; ++e_beg) {
      int opCode = get(get(edge_iore, G),*e_beg);
        
      if (opCode == Instruction::Load) {
      NodeProps *sourceVP = get(get(vertex_props, G), source(*e_beg, G));
            if (!sourceVP) continue;
      int sourceLine = sourceVP->line_num;
#ifdef DEBUG_GRAPH_COLLAPSE
      blame_info<<"For sourceVP(Load): "<<sourceVP->name<<" "<<sourceLine<<endl;
      blame_info<<"---"<<endl;
#endif 
      set<NodeProps *>::iterator set_vp_i;
        for (set_vp_i = vp->storesTo.begin(); set_vp_i != vp->storesTo.end(); set_vp_i++) {
        NodeProps *storeVP = (*set_vp_i);
#ifdef DEBUG_GRAPH_COLLAPSE
        blame_info<<"storeVP: "<<storeVP->name<<" - Store Lines - ";
              set<int>::iterator set_i_i;
            for (set_i_i = storeVP->storeLines.begin(); set_i_i != storeVP->storeLines.end(); set_i_i++)
            blame_info<<*set_i_i<<" ";
  
        blame_info<<" ; Border Lines - ";
            for (set_i_i = storeVP->borderLines.begin(); set_i_i != storeVP->borderLines.end(); set_i_i++)
            blame_info<<*set_i_i<<" ";

        blame_info<<endl;
#endif 
        if ((storeVP->storeLines.count(sourceLine)>0 && resolveStoreLine(storeVP, sourceVP)) ||
          (storeVP->borderLines.count(sourceLine)>0 && resolveBorderLine(storeVP, sourceVP, vp, sourceLine))) {
        if (sourceVP->llvm_inst != NULL) {
          int newPtrLevel = 99;
          if (isa<Instruction>(sourceVP->llvm_inst)) {
          const llvm::Type *origT = 0;    
          Instruction *pi = cast<Instruction>(sourceVP->llvm_inst);  
          origT = pi->getType();  
          newPtrLevel = pointerLevel(origT,0);
          }
          else if (isa<ConstantExpr>(sourceVP->llvm_inst)) {
          const llvm::Type *origT = 0;    
          ConstantExpr *ce = cast<ConstantExpr>(sourceVP->llvm_inst);
            origT = ce->getType();
          newPtrLevel = pointerLevel(origT, 0);
          }
                
                  //example:store C, b; a = load b;
                  //changed by Hui 03/24/16:we shouldn't remove a->b if C is a constant, 
                  //since it could be just an initialization of b.
                  //only delete load edge if sourceVP is a non-ptr and non-const
                  if (newPtrLevel==0 && storeVP->name.find("Constant+")==string::npos) {
          deleteMe.insert(sourceVP->number);
#ifdef DEBUG_GRAPH_COLLAPSE
                  blame_info<<"Need to remove edge(hui) from "<<sourceVP->name<<" to "<<vp->name<<" [8]"<<endl;
#endif 
                  }
        }
              
        tie(ed, inserted) = add_edge(sourceVP->number, storeVP->number, G);
        if (inserted){
                  blame_info<<"RESOLVED_L_S_OP added between "<<sourceVP->name<<" and "<<storeVP->name<<endl;
          edge_type[ed] = RESOLVED_L_S_OP;//load_store op
                  //TOCHECK: added by Hui 03/24/16: if sourceVP's Ptrlevel==0, we still want to keep the RLS relation
                  //However, later resolvePointers won't add them to resolvedLSFrom/resolvedLS, so we add it here in advance
                  storeVP->resolvedLS.insert(sourceVP);
                  sourceVP->resolvedLSFrom.insert(storeVP);
                }
        }//end of satisfying resolveStoreLines condition
        }//for all storesTo of vp
#ifdef DEBUG_GRAPH_COLLAPSE
      blame_info<<"---"<<endl;
      blame_info<<endl;
#endif 
      }
    }
    //remove edges from the sourceVp to vp if it's in deleteMe  
    set<int>::iterator set_i_i;
    for (set_i_i = deleteMe.begin(); set_i_i != deleteMe.end(); set_i_i++) {
      remove_edge((*set_i_i), vp->number, G);
    }

    }//if vp->storesTo.size() >0
  }//for all vertices
}

bool FunctionBFC::resolveStoreLine(NodeProps *storeVP, NodeProps *sourceVP)
{
#ifdef DEBUG_GRAPH_COLLAPSE
  blame_info<<"In resolveStoreLine storeVP:"<<storeVP->name<<" ln="<<storeVP->line_num<<" lNO="<<storeVP->lineNumOrder
        <<"  sourceVP:"<<sourceVP->name<<" ln="<<sourceVP->line_num<<" lNO="<<sourceVP->lineNumOrder<<endl;
#endif 
    if(storeVP->line_num < sourceVP->line_num)
        return true;
    else if(storeVP->line_num == sourceVP->line_num){ //cases like: findme = findme*10; both load and store
        
        //if(sourceVP->lineNumOrder > storeVP->lineNumOrder) //are in the same line, but THE load shouldn't depend on
        //   return true;                                   //THE store since the store comes after the load
        //
        //Changed by Hui 05/10/16: the above condition only effect on cases when both
        //are reigisters, when one of them is variable, then the lineNumOrder isn't
        //reliable, we need to find out the corresponding Store/Load, and use their
        //lineNumOrder, here storeVP->line_num == sourceVP->line_num
        int commLine = sourceVP->line_num;
        if ((sourceVP->loadLineNumOrder).find(commLine) != (sourceVP->loadLineNumOrder).end() &&
            (storeVP->storeLineNumOrder).find(commLine) != (storeVP->storeLineNumOrder).end() &&
            (sourceVP->loadLineNumOrder)[commLine] >= (storeVP->storeLineNumOrder)[commLine])
            return true;
        else{ 
            blame_info<<"resolveStoreLine fails because lineNumOrder"<<endl;
            return false; //the store comes after the load
        }
    }
    else {// when storeVP->line_num > sourceVP->line_num, clearly false
        blame_info<<"resolveStoreLine fails because line_num"<<endl;
        return false;
    }
}

bool FunctionBFC::resolveBorderLine(NodeProps *storeVP, NodeProps *sourceVP, \
                           NodeProps * origStoreVP, int sourceLine)
{
#ifdef DEBUG_GRAPH_COLLAPSE
  blame_info<<endl;
  blame_info<<"In resolveBorderLine "<<storeVP->name<<" "<<sourceVP->name<<" "<<origStoreVP->name<<endl;
#endif 
    //int storeVPNum = storeVP->lineNumOrder;
  vector<FuncStores *>::iterator vec_fs_i;
  
  for (vec_fs_i = allStores.begin(); vec_fs_i != allStores.end(); vec_fs_i++) {
    FuncStores *fs = *vec_fs_i;
#ifdef DEBUG_GRAPH_COLLAPSE    
    blame_info<<"FS: ";
    if (fs->receiver)
      blame_info<<"Receiver "<<fs->receiver->name<<" ";
    else
      blame_info<<"Receiver NULL ";
    if (fs->contents)
      blame_info<<"Contents "<<fs->contents->name<<" ";
    else
      blame_info<<"Contents NULL ";
    
    blame_info<<fs->line_num<<" "<<fs->lineNumOrder<<endl;
#endif 
    
    if (fs->receiver == origStoreVP && fs->line_num == sourceLine) {    
#ifdef DEBUG_GRAPH_COLLAPSE
      blame_info<<"Border Line between "<<storeVP->name<<" "<<sourceVP->name<<" "<<origStoreVP->name<<endl;
#endif 
            //Here, the fs is the store that kills the previous store, so
            //it should come after the load(sourceVP) so that the previous store
            //would be effect on sourceVP
      if (fs->lineNumOrder > sourceVP->lineNumOrder) {
#ifdef DEBUG_GRAPH_COLLAPSE
        blame_info<<"Border Cond resolved"<<endl;
#endif 
        return true;
      }
    }
  }
  
#ifdef DEBUG_GRAPH_COLLAPSE
  blame_info<<"Border Cond NOT resolved"<<endl;
#endif 
  
  return false;
}

// get original typeID of a pointer
unsigned FunctionBFC::getPointedTypeID(const llvm::Type *t)
{
  unsigned typeVal = t->getTypeID();
  if (typeVal == Type::PointerTyID)
    return getPointedTypeID(cast<PointerType>(t)->getElementType());
  else
    return typeVal;
}


// get original type of a pointer(if t is)
const Type* FunctionBFC::getPointedType(const Type *t)
{
  unsigned typeVal = t->getTypeID();
  if (typeVal == Type::PointerTyID)
    return getPointedType(cast<PointerType>(t)->getElementType());
  else
    return t;
}


void FunctionBFC::addImplicitEdges(Value *v, set<const char*, ltstr> &iSet,                           
        property_map<MyGraphType, edge_iore_t>::type edge_type, string vName, bool useName)
{
    if (exclusive_blame) //In exclusive blame, the loop iteration line should not be included in blameSet of it
      return;

    bool inserted;
    graph_traits < MyGraphType >::edge_descriptor ed;
  
  string impName;
  char tempBuf[18];
  
  if (useName) {
    impName.insert(0, vName);
  }
  else if (v->hasName()) {
    impName = v->getName().str();
  }
  else {
    //sprintf(tempBuf, "0x%x", (unsigned)v);
      sprintf(tempBuf, "0x%x", v);
    string tempStr(tempBuf);
    impName.insert(0, tempStr);      
  }

  if (variables.count(impName) <= 0) 
        return;

  // Take Care of Implicit Edges
  for (set<const char*, ltstr>::iterator s_iter = iSet.begin(); s_iter != iSet.end(); s_iter++) { 
        string s_iterString(*s_iter);
    if (variables.count(s_iterString)>0) {
#ifdef DEBUG_GRAPH_IMPLICIT    
      blame_info<<"Adding implicit edges between "<<impName<<" and "<<*s_iter<<endl;
#endif 
      tie(ed, inserted) = add_edge(variables[impName]->number, variables[s_iterString]->number, G); 
      if (inserted)
        edge_type[ed] = IMPLICIT_OP;  
      else {
#ifdef DEBUG_ERROR
        blame_info<<"ERROR__(genEdges) - Insertion fail for implicit edge"<<endl;
#endif
        //cerr<<"Insertion fail for implicit edge\n";
      }
    }

    else
            continue;
      //return;
  }
}

// Almost the same as geDefault, except we don deal with the last operand, which is the call node
void FunctionBFC::geDefaultPTXIntrinsic(Instruction *pi, set<const char*, ltstr> &iSet, 
              property_map<MyGraphType, vertex_props_t>::type props,
              property_map<MyGraphType, edge_iore_t>::type edge_type,
              int &currentLineNum)

{
    //bool inserted;
    graph_traits < MyGraphType >::edge_descriptor ed; 
  
  // Take care of implicit edges for default case 
  addImplicitEdges(pi, iSet, edge_type, "", false);
  
  string instName;
  char tempBuf[18];
  
  if (pi->hasName()) {
    instName = pi->getName().str();
  }
  else {
    sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
    string tempStr(tempBuf);
    instName.insert(0, tempStr);      
  }
  
    int opCount = 0, callNameIdx = pi->getNumOperands()-1; 
  // Take care of Explicit Edges for non-load operations with 2+ operands
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
      if (opCount == callNameIdx) //we don't deal with the call node for ptx intrinsics
          continue;

    if (v->hasName()) {
      addEdge(instName, v->getName().str(), pi, 1);
    }
    else {
        string opName;
        //added by Hui : deal with constant operands
        if (isa<ConstantInt>(v)) {
          ConstantInt *cv = (ConstantInt *)v;
          int number = cv->getSExtValue();

          char tempBuf[64];
          sprintf(tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, 0, pi->getOpcode());    
          char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));

          strcpy(vN,tempBuf);
          vN[strlen(tempBuf)]='\0';
          const char *vName = vN;
          opName.insert(0, vName);
        } 
        else if (isa<ConstantFP>(v)) {
          char tempBuf[70];
          ConstantFP *cfp = (ConstantFP *)v;
          const APFloat apf = cfp->getValueAPF();

          if(APFloat::semanticsPrecision(apf.getSemantics()) == 24) {
            float floatNum = apf.convertToFloat();
            sprintf (tempBuf, "Constant+%g+%i+%i+%i", floatNum, currentLineNum, 0, pi->getOpcode());    
          }
          else if(APFloat::semanticsPrecision(apf.getSemantics()) == 53) {
            double floatNum = apf.convertToDouble();
            sprintf (tempBuf, "Constant+%g2.2+%i+%i+%i", floatNum, currentLineNum, 0, pi->getOpcode());    
          }
          else {
#ifdef DEBUG_ERROR
            blame_info<<"Not a float or a double FPVal"<<endl;
#endif 
            return;
          }

          char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
          strcpy(vN,tempBuf);
          vN[strlen(tempBuf)]='\0';
          const char * vName = vN;
          opName.insert(0, vName);
        }
        //registers
        else{
          sprintf(tempBuf, "0x%x", /*(unsigned)*/v);
          string tempStr(tempBuf);
          opName.insert(0, tempStr);
        }

      addEdge(instName, opName, pi, 2);
      }//operand has no name
  
      opCount++;
    }//for all operands
}


void FunctionBFC::geDefault(Instruction *pi, set<const char*, ltstr> &iSet, 
              property_map<MyGraphType, vertex_props_t>::type props,
              property_map<MyGraphType, edge_iore_t>::type edge_type,
              int &currentLineNum)

{
    //bool inserted;
    graph_traits < MyGraphType >::edge_descriptor ed; 
  
  // Take care of implicit edges for default case 
  addImplicitEdges(pi, iSet, edge_type, "", false);
  
  string instName;
  char tempBuf[18];
  
  if (pi->hasName()) {
    instName = pi->getName().str();
  }
  else {
    sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
    string tempStr(tempBuf);
    instName.insert(0, tempStr);      
  }
  
  // Take care of Explicit Edges for non-load operations with 2+ operands
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
    if (v->hasName()) {
      addEdge(instName, v->getName().str(), pi, 1);
    }
    else {
        string opName;
        //added by Hui : deal with constant operands
        if (isa<ConstantInt>(v)) {
          ConstantInt *cv = (ConstantInt *)v;
          int number = cv->getSExtValue();

          char tempBuf[64];
          sprintf(tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, 0, pi->getOpcode());    
          char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));

          strcpy(vN,tempBuf);
          vN[strlen(tempBuf)]='\0';
          const char *vName = vN;
          opName.insert(0, vName);
        } 
        else if (isa<ConstantFP>(v)) {
          char tempBuf[70];
          ConstantFP *cfp = (ConstantFP *)v;
          const APFloat apf = cfp->getValueAPF();

          if(APFloat::semanticsPrecision(apf.getSemantics()) == 24) {
            float floatNum = apf.convertToFloat();
            sprintf (tempBuf, "Constant+%g+%i+%i+%i", floatNum, currentLineNum, 0, pi->getOpcode());    
          }
          else if(APFloat::semanticsPrecision(apf.getSemantics()) == 53) {
            double floatNum = apf.convertToDouble();
            sprintf (tempBuf, "Constant+%g2.2+%i+%i+%i", floatNum, currentLineNum, 0, pi->getOpcode());    
          }
          else {
#ifdef DEBUG_ERROR
            blame_info<<"Not a float or a double FPVal"<<endl;
#endif 
            return;
          }

          char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
          strcpy(vN,tempBuf);
          vN[strlen(tempBuf)]='\0';
          const char * vName = vN;
          opName.insert(0, vName);
        }
        //registers
        else{
          sprintf(tempBuf, "0x%x", /*(unsigned)*/v);
          string tempStr(tempBuf);
          opName.insert(0, tempStr);
        }

      addEdge(instName, opName, pi, 2);
      }//operand has no name
  }//for all operands
}
//added by Hui 08/20/15  Temporary solution for multi-level structures (2-level)
string FunctionBFC::getRealStructName(string rawStructVarName, Value *v, User *pi,
                                                                string instName)
{
    if (isa<ConstantInt>(v)) {
        blame_info<<"The instName of GEP(1): "<<instName<<endl;
        /////step1 done/////////////NOT very useful///////////////////////
        //if(rawStructVarName.find("localActor") != string::npos)
        //    return rawStructVarName; //temporary scheme, for the first level GEP

        Value *vLoad = *(pi->op_begin());
        if (isa<Instruction>(vLoad)) {
            blame_info<<"vLoad is an instruction\n";
            Instruction *vInstLoad = cast<Instruction>(vLoad);
            if(vInstLoad->getOpcode() == Instruction::Load){
                blame_info<<"vInstLoad is a Load instruction\n";
                Value *midTemp = vInstLoad->getOperand(0);
                /*if(midTemp->hasName())
                    NodeProps *nodeTemp=variables[midTemp->getName().str().c_str()];
                    if(nodeTemp->storeFrom != NULL)
                        blame_info<<"We have nodeStore now !\n";
                        NodeProps *nodeStore = nodeTemp->storeFrom;
                */
                Value::use_iterator ui, ue;
                for(ui=midTemp->use_begin(),ue=midTemp->use_end(); ui!=ue; ++ui){
                    if(Instruction *useInst = dyn_cast<Instruction>((*ui).getUser())){
                        if(useInst->getOpcode()==Instruction::Store){
                            blame_info<<"We find the store inst for call_tmp\n";
                            Value *storeFrom = *(useInst->op_begin());
                            if(isa<Instruction>(storeFrom)){
                                Instruction *instGEP = cast<Instruction>(storeFrom);
                                if(instGEP->getOpcode()==Instruction::GetElementPtr){
                                    blame_info<<"We find GEP inst !\n";
                                    Value *topStruct = *(instGEP->op_begin());
                                    string tempStr; //hold the return string
                                    if(topStruct->hasName()){ //TO CONTINUE: 12/28/15
                                      tempStr.insert(0,topStruct->getName().str());
                                      blame_info<<"GEP topStruct hasName: "<<tempStr<<endl;
                                      User::op_iterator op_i = instGEP->op_begin();
                                      if(instGEP->getNumOperands()>2){
                                        ++op_i; //first index
                                        ++op_i; //second index
                                        Value *offset = op_i->get();
                                        if(isa<ConstantInt>(offset)){
                                          ConstantInt *cO = (ConstantInt *)offset;
                                          int offNum = cO->getSExtValue();
                                tempStr.insert(0, ".P.");
          
                                char fNumStr[3];
                                sprintf(fNumStr, "%d", offNum);
                                tempStr.insert(0, fNumStr);
                                          return tempStr;
                                        }
                                        else {
                                          blame_info<<"Fail in Cond 0\n";
                                          return rawStructVarName; //to check: should I return x.P.tempStr ?
                                        }
                                      }
                                      else{
                                        blame_info<<"We are accesing one element in an array(topStruct hasName: "<<tempStr<<")\n";
                                        return tempStr;
                                      } 
                                    }
                                    else { //topStruct doesn't have a name(still a register)
                                      User::op_iterator op_i = instGEP->op_begin();
                                      Value *vLoad2 = op_i->get(); //vLoad2=topStruct
                                      char tempBuff2[18];
                                      sprintf(tempBuff2, "0x%x",vLoad2);
                                      string tempStr2(tempBuff2);
                                      tempStr.clear();
                                      tempStr.insert(0,tempStr2);  
                                      if(instGEP->getNumOperands()>2){
                                        ++op_i; //first index
                                        ++op_i; //second index
                                        Value *offset = *op_i;
                                        if(isa<ConstantInt>(offset)){
                                          ConstantInt *cO = (ConstantInt *)offset;
                                          int offNum = cO->getSExtValue();
                                tempStr.insert(0, ".P.");
          
                                char fNumStr[3];
                                sprintf(fNumStr, "%d", offNum);
                                tempStr.insert(0, fNumStr);
                                          return tempStr;
                                        }
                                        else {
                                          blame_info<<"Fail in Cond -1\n";
                                          return rawStructVarName; //to check: should I return x.P.tempStr ?
                                        }
                                      }
                                      else{
                                        blame_info<<"We are accesing one element in an array(topStruct doesn't hasName\n";
                                        return tempStr;
                                      } 
                                    }//end of !topStruct->hasName()
                                }//end of upper level GEP inst   
                                else {
                                    blame_info<<"Fail in Cond 2\n";
                                    return rawStructVarName;
                                }
                            }
                            else {
                                blame_info<<"Fail in Cond 3\n";
                                return rawStructVarName;
                            }
                        }
                        else {
                            blame_info<<"Fail in Cond4\n";
                            continue; //if the Instruction isn't Store
                        }
                    }
                    else {
                        blame_info<<"Fail in Cond5\n";
                        continue; //if the use isn't an Instruction
                    }
                }
                blame_info<<"Searched all uses, didn't find Store\n";
                return rawStructVarName;
            }
            else {
                blame_info<<"Fail in Cond6\n";
                return rawStructVarName;
            }
        }
        else {
            blame_info<<"Fail in Cond7\n";
            return rawStructVarName;
        }
    }
    else {
        blame_info<<"Fail in Cond8\n";
        return rawStructVarName;
    }
}


string FunctionBFC::getUniqueNameAsFieldForNode(Value *val, int errNo, string rawStructVarName)
{
  //If val has name, then we simply return either its uniqueNameAsField if existed, not its own name(it's the top level struct)
  string nodeName = getNameForVal(val);
    
  if(variables.count(nodeName)>0){
    if(!(variables[nodeName]->uniqueNameAsField.empty())){
      string retName(variables[nodeName]->uniqueNameAsField);
      return retName;
    }
    else {
      blame_info<<errNo<<" Error: Var: "<<nodeName<<" didn't have uniqueNameAsField"<<endl;
      return rawStructVarName;
    }
  }
  else { //node not in variables, error
    blame_info<<errNo<<" Error: node not found in variables: "<<nodeName<<endl;
    return rawStructVarName;
  }
}


//added by Hui 01/14/16: get the original value back-tracing the S->L chain
Value* FunctionBFC::getValueFromOrig(Instruction *vInstLoad) 
{
  Value *loadFrom = *(vInstLoad->op_begin());
  if(Instruction *loadFromInst = dyn_cast<Instruction>(loadFrom)){
    if(loadFromInst->getOpcode()==Instruction::GetElementPtr ||
       loadFromInst->getOpcode()==Instruction::ExtractValue) //It means there's no more store->load chains before
      return vInstLoad;                                      //we reach to GEP,simply return this load inst
    //else: store r1, tmp;  r2=load tmp;(tmp is just Alloca, we need to do store-load backtrace)
    Value::use_iterator ui, ue;
    for(ui=loadFrom->use_begin(),ue=loadFrom->use_end(); ui!=ue; ++ui){
      if(Instruction *useInst = dyn_cast<Instruction>((*ui).getUser())){
        if(useInst->getOpcode()==Instruction::Store){
          Value *valueFrom = *(useInst->op_begin());
          if(Instruction *valueFromInst = dyn_cast<Instruction>(valueFrom)){
            if(valueFromInst->getOpcode()==Instruction::Load){
              blame_info<<"Start recursion inside getValueFromOrig"<<endl;
              return getValueFromOrig(valueFromInst); //recursively call this func
            }
            else return valueFrom;//valueFromInst is the GEP retVal,simply return it
          }
          else return valueFrom;//valueFrom isn't inst,so it's an exitVar(param/gv)
        }
      }
    }
    blame_info<<"Weird: shouldn't be here!"<<endl;
    return vInstLoad;
  }
  //not an inst, it could be a CE or globalVar or ArgVar
  blame_info<<"loadFrom isn't an inst(even not alloca), we deal it after return"<<endl;
  return vInstLoad;
}


// Helper function, given an llvm::value, return a string name 
string FunctionBFC::getNameForVal(Value *val) 
{
  string valName;
  if (val->hasName())
    valName = val->getName().str();
    
  else {
    char tempBuff[18];
    sprintf(tempBuff, "0x%x",val);
    string tempStr(tempBuff);
    valName = tempStr;
  }

  return valName;
}


//Helper function in getUpperLevelFieldName
string FunctionBFC::resolveGEPBaseFromLoad(Instruction *vInst, NodeProps *vBaseNode, string rawStructVarName)
{
    string baseName = vBaseNode->name; //vBaseNode has to be there for sure since that's the vBase!
    if(vInst->getOpcode() == Instruction::Load){
      //we need backtrace the (a=LD b)->SL(store a, tmp)->LD(c=load tmp) chain to get 'a'
      Value *valueFrom = getValueFromOrig(vInst);
      blame_info<<"After getValueFromOrig, we have "<<valueFrom<<endl;

      if(Instruction *valueFromInst = dyn_cast<Instruction>(valueFrom)){
        if(valueFromInst->getOpcode()==Instruction::GetElementPtr){//1st case: GEP->(ST->LD)->GEP
          //found GEPFather for vBase
          Value *gf = *(valueFromInst->op_begin());      
          string gfName = getNameForVal(gf); 
          if (variables.count(gfName)>0 && vBaseNode) {
            NodeProps *gfNode = variables[gfName];
            gfNode->GEPChildren.insert(vBaseNode);
            vBaseNode->GEPFather = gfNode;
          }
          else blame_info<<"1 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;
          //return the uniqueNameAsField of this GEP retVal
          return getUniqueNameAsFieldForNode(valueFrom, 1, rawStructVarName);
        }
        else if(valueFromInst->getOpcode()==Instruction::ExtractValue){//2nd case: extractval->(ST->LD)->GEP
          //found GEPFather for vBase
          Value *gf = *(valueFromInst->op_begin());      
          string gfName = getNameForVal(gf); 
          if (variables.count(gfName)>0 && vBaseNode) {
            NodeProps *gfNode = variables[gfName];
            gfNode->GEPChildren.insert(vBaseNode);
            vBaseNode->GEPFather = gfNode;
          }
          else blame_info<<"2 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;
          //return the uniqueNameAsField of this extractvalue retVal
          return getUniqueNameAsFieldForNode(valueFrom, 2, rawStructVarName);
        }
        else if(valueFromInst->getOpcode()==Instruction::Load){//3rd case: (*)-LD->(ST->LD)->GEP
          Value *loadFrom = *(valueFromInst->op_begin());

          if(Instruction *loadFromInst = dyn_cast<Instruction>(loadFrom)){
            if(loadFromInst->getOpcode()==Instruction::GetElementPtr){//2nd most common case: GEP->LD->(ST->LD)->GEP
              //found GEPFather for vBase
              Value *gf = *(loadFromInst->op_begin());      
              string gfName = getNameForVal(gf); 
              if (variables.count(gfName)>0 && vBaseNode) {
                NodeProps *gfNode = variables[gfName];
                gfNode->GEPChildren.insert(vBaseNode);
                vBaseNode->GEPFather = gfNode;
              }
              else blame_info<<"3 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;
              //return the uniqueNameAsField of this GEP retVal 
              return getUniqueNameAsFieldForNode(loadFrom, 3, rawStructVarName);
            }
            else if(loadFromInst->getOpcode()==Instruction::ExtractValue){//2nd most common case: GEP->load->(SL...)->GEP
              //found GEPFather for vBase
              Value *gf = *(loadFromInst->op_begin());      
              string gfName = getNameForVal(gf); 
              if (variables.count(gfName)>0 && vBaseNode) {
                NodeProps *gfNode = variables[gfName];
                gfNode->GEPChildren.insert(vBaseNode);
                vBaseNode->GEPFather = gfNode;
              }
              else blame_info<<"4 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;
              //return the uniqueNameAsField of this extractvalue retVal   
              return getUniqueNameAsFieldForNode(loadFrom, 4, rawStructVarName);
            }
            else if(loadFromInst->getOpcode()==Instruction::Alloca && loadFromInst->hasName()) {
              //we need to add GEPFather/child relation between vBase and loadFromInst in order to propagate line# later
              //found GEPFather for vBase
              Value *gf =(Value*)loadFromInst;      
              string gfName = getNameForVal(gf); 
              if (variables.count(gfName)>0 && vBaseNode) {
                NodeProps *gfNode = variables[gfName];
                gfNode->GEPChildren.insert(vBaseNode);
                vBaseNode->GEPFather = gfNode;
              }
              else blame_info<<"5 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;
              //simply return loadFromInst's name
              return loadFromInst->getName().str();
            }
            else {
              blame_info<<"what's the inst of loadFrom??"<<endl;
              return rawStructVarName;
            }
          } 
          // when we have: a = load (GEP b, 0..)
          else if(isa<ConstantExpr>(loadFrom)){
            //return variables[0x330b2d8.CE.44]->uniqueNameAsField
            blame_info<<"The loadFrom is a ConstantExpr, there is an embedded GEP!"<<endl;
            ConstantExpr *ceGEP = cast<ConstantExpr>(loadFrom);
            if(ceGEP->getOpcode() == Instruction::GetElementPtr){
              //we need to add GEP father/child relationship between ceBase and vBase
              Value *gf = *(ceGEP->op_begin());
              string gfName = getNameForVal(gf);
              if (variables.count(gfName)>0 && vBaseNode) {
                NodeProps *gfNode = variables[gfName];
                gfNode->GEPChildren.insert(vBaseNode);
                vBaseNode->GEPFather = gfNode;
              }
              else blame_info<<"6 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;
              //since we don't know the currentlinenum of this ceGEP node when this func is called
              //so we can't directly get its uniqueNameAsField, have to recompute it from the base
              string gfUniqueNameAsField = getUpperLevelFieldName(gfName, ceGEP, "ceGEP");
              //now start to manully get loadFrom's uniqueNameAsField
              if(ceGEP->getNumOperands() == 3){
                Value *fieldNum = ceGEP->getOperand(2);
                if(isa<ConstantInt>(fieldNum)){
                  ConstantInt *cv = (ConstantInt *)fieldNum;
                  int number = cv->getSExtValue();
                  
                  string fieldName = gfUniqueNameAsField;
                  fieldName.insert(0, ".P.");
                  char fieldNumStr[3];
                  sprintf(fieldNumStr, "%d", number);
                  fieldName.insert(0, fieldNumStr);
                  //we've computed the uniqueNameAsField for this ce/loadFrom, which is also the one for vBase
                  return fieldName;
                }
              }
              else if(ceGEP->getNumOperands() == 2){
                string fieldName = gfUniqueNameAsField;
                fieldName.insert(0, "I.");
                return fieldName;
              }
              else
                blame_info<<"Error: ceGEP has(!=2/3) "<<ceGEP->getNumOperands()<<" Operands"<<endl;
            }
            else if(ceGEP->getOpcode() == Instruction::ExtractValue){ //place holder, not necessary will happen
              //we need to add GEP father/child relationship between ceBase and vBase
              Value *gf = *(ceGEP->op_begin());
              string gfName = getNameForVal(gf);
              if (variables.count(gfName)>0 && vBaseNode) {
                NodeProps *gfNode = variables[gfName];
                gfNode->GEPChildren.insert(vBaseNode);
                vBaseNode->GEPFather = gfNode;
              }
              else blame_info<<"7 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;
              //since we don't know the currentlinenum of this ceExtractValue node when it's called
              //so we can't directly get its uniqueNameAsField, have to recompute it from the base
              string gfUniqueNameAsField = getUpperLevelFieldName(gfName, ceGEP, "ceExtVal");
              //now start to manully get loadFrom's uniqueNameAsField
              if(ceGEP->getNumOperands() == 2){ //For now, we only take care of 1st index
                Value *fieldNum = ceGEP->getOperand(1);
                if(isa<ConstantInt>(fieldNum)){
                  ConstantInt *cv = (ConstantInt *)fieldNum;
                  int number = cv->getSExtValue();
                    
                  string fieldName = gfUniqueNameAsField;
                  fieldName.insert(0, ".P.");
                  char fieldNumStr[3];
                  sprintf(fieldNumStr, "%d", number);
                  fieldName.insert(0, fieldNumStr);
                  //we've computed the uniqueNameAsField for this ce/loadFrom, which is also the one for vBase
                  return fieldName;
                }
              }
              else {
                blame_info<<"Error: ceExtVal has(!=2) "<<ceGEP->getNumOperands()<<" Operands"<<endl;
                return rawStructVarName;
              }
            } 
            else {
              blame_info<<"What is this constantExpr of loadFrom?? opcode="<<ceGEP->getOpcode()<<endl;
              return rawStructVarName;
            }
          }
          //It's an exitVar(globalVar): load->(ST->LD)->GEP (TOCHECK)
          else if(loadFrom->hasName()){
            //we need to add GEPFather/child relation between vBase and loadFrom in order to propagate line# later
            //found GEPFather for vBase
            Value *gf =(Value*)loadFrom;      
            string gfName = getNameForVal(gf); 
            if (variables.count(gfName)>0 && vBaseNode) {
              NodeProps *gfNode = variables[gfName];
              gfNode->GEPChildren.insert(vBaseNode);
              vBaseNode->GEPFather = gfNode;
            }
            else blame_info<<"8 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;
            //return loadFrom->name (gv's name as the vBase's uniqueNameAsField)
            return loadFrom->getName().str();
          }
          else {
            blame_info<<"Fail in case 1"<<endl;
            return rawStructVarName;
          }
        }//valueFromInst==load
        else {
          blame_info<<"Fail in case 2, not likely"<<endl;
          return rawStructVarName;
        }
      }//valueFrom is inst
      else if (isa<ConstantExpr>(valueFrom)){
        blame_info<<"Not dealing with bazzard nested GEP case"<<endl;
        return rawStructVarName;
      }
      else if (valueFrom->hasName()) {
        //we need to add GEPFather/child relation between vBase and loadFrom in order to propagate line# later
        //found GEPFather for vBase
        Value *gf =(Value*)valueFrom;      
        string gfName = getNameForVal(gf); 
        if (variables.count(gfName)>0 && vBaseNode) {
          NodeProps *gfNode = variables[gfName];
          gfNode->GEPChildren.insert(vBaseNode);
          vBaseNode->GEPFather = gfNode;
        }
        else blame_info<<"9 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;
        //directly return the name of arg/gv as vBase's uniqueNameAsField
        return valueFrom->getName().str();
      }
      else {
        blame_info<<"Fail in case 3, not likely"<<endl;
        return rawStructVarName;
      }
    }//vBase is Load
    
    else {
      blame_info<<"WRONG func call! vInst isn't load"<<endl;
      return rawStructVarName;
    }
}

//added by Hui 01/14/16: try to get the uniqueNameAsField for a GEP field node
//rawStructVarName is the structVarNameGlobal,which is its own name for base, and base's uniqueNameAsField of base for others
string FunctionBFC::getUpperLevelFieldName(string rawStructVarName, User *pi, string instName)
{
  Value *vBase = *(pi->op_begin()); 
  NodeProps *vBaseNode = NULL;
  string baseName = getNameForVal(vBase);
  blame_info<<"In getUpperLevelFieldName for inst "<<instName<<" base "<<baseName<<endl;
  //if GEP base already has its uniqueNameAsField, simply return it
  if(variables.count(baseName)>0){
    vBaseNode = variables[baseName]; //should be there
    if(!(variables[baseName]->uniqueNameAsField.empty())){
      string retName(variables[baseName]->uniqueNameAsField);
      blame_info<<"GEP base: "<<baseName<<"  already has the uniqueNameAsField("
          <<retName<<"), simply return it!"<<endl;
      return retName;
    }
  }
  //else: GEP base doesn't have a uniqueNameAsField yet, we back trace
  if (Instruction *vInst = dyn_cast<Instruction>(vBase)) {
    if (vInst->getOpcode() == Instruction::Load) {
      return resolveGEPBaseFromLoad(vInst, vBaseNode, rawStructVarName);
    }//vBase is Load
    else { //we need to resolve GEP->(LD->ST)->GEP case, we have to put it here since vInst is also Alloca
      boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
    e_beg = boost::out_edges(vBaseNode->number, G).first;  //edge iterator begin
    e_end = boost::out_edges(vBaseNode->number, G).second; // edge iterator end
      for (; e_beg!=e_end; e_beg++) {
        int opCode = get(get(edge_iore, G), *e_beg);
        // if the bw propagation follows (LD->ST) chain
        if (opCode == Instruction::Store) {  
          NodeProps *sourceV = get(get(vertex_props,G), target(*e_beg,G));
          if (sourceV && sourceV->llvm_inst) {
            if (Instruction *sourceVInst = dyn_cast<Instruction>(sourceV->llvm_inst)) {
              if (sourceVInst->getOpcode() == Instruction::Load) {
                return resolveGEPBaseFromLoad(sourceVInst, vBaseNode, rawStructVarName);
              }
            }
          }
        }
      }
    }

    //for consistant GEP/extracvals, a=GEP b; c=GEP a: a should already have its uniqueNameAsField, we shouldn't be here
    if(vInst->getOpcode()==Instruction::GetElementPtr || vInst->getOpcode()==Instruction::ExtractValue) {
      //we need to add GEP father/child relationship between a and b
      Value *gf = *(vInst->op_begin());
      string gfName = getNameForVal(gf);
      if (variables.count(gfName)>0 && vBaseNode) {
        NodeProps *gfNode = variables[gfName];
        gfNode->GEPChildren.insert(vBaseNode);
        vBaseNode->GEPFather = gfNode;
      }
      else 
        blame_info<<"10 Fail adding GEPFather("<<gfName<<") and Child("<<baseName<<")"<<endl;  
      blame_info<<"Check: "<<baseName<<", we are not supposed to be here"<<endl;
      return rawStructVarName;
    }
    
    //ruling out the previous else case, now it's only for the top GEP, its base is the LV (local var)
    //if (LD->ST)->GEP, it should return alreay in the previous else case
    //since now vBase=GEPFather, so we don't need to add GEPFather/child relationship for it
    else if(vInst->getOpcode() == Instruction::Alloca && vInst->hasName()) {
      return vInst->getName().str();
    }
  }//vBase is inst
  
  //the top GEP base isn't inst or hasName, it's a CE?
  else if (isa<ConstantExpr>(vBase)) {
    blame_info<<"We currently not deal with nested GEP operations"<<endl;
    return rawStructVarName;
  }
  
  //for the top GEP, its base is the EV (globaVar or argVar), so it's not inst, but should have name
  else if (vBase->hasName()) { 
    //since now vBase=GEPFather, so we don't need to add GEPFather/child relationship for it
    return vBase->getName().str(); //baseName
  }
    
  else
    blame_info<<"Check, what the hell is vBase??? "<<rawStructVarName<<endl;
  return rawStructVarName;
}
        

string FunctionBFC::geGetElementPtr(User *pi, set<const char*, ltstr> &iSet,property_map<MyGraphType, 
        vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type,int &currentLineNum)
{
  bool inserted;
  graph_traits < MyGraphType >::edge_descriptor ed;
  
  addImplicitEdges(pi, iSet, edge_type, "", false);
  
  string instName;
  char tempBuf[18];
  if (pi->hasName()) {
    instName = pi->getName().str();
  }
  else {
    sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
    string tempStr(tempBuf);
    instName.insert(0, tempStr);      
    
    if (isa<ConstantExpr>(pi)) {
      instName += ".CE";
      char tempBuf2[10];
      sprintf(tempBuf2, ".%d", currentLineNum);
      instName.append(tempBuf2);
    }
  }
  
  // Name of struct variable name (essentially op0)
  string structVarNameGlobal;
  int opCount = 0;
  // Go through the operands
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
    if (v->hasName() && variables[instName] && variables[v->getName().str()]) {
      tie(ed, inserted) = add_edge(variables[instName]->number,variables[v->getName().str()]->number,G);
      if (inserted) {
        if (opCount == 0) {
          string baseName = v->getName().str();
          edge_type[ed] = GEP_BASE_OP;//pi->getOpcode();  previously existed
          variables[instName]->pointsTo = variables[baseName];
          variables[baseName]->pointedTo.insert(variables[instName]);
#ifdef DEBUG_GRAPH_BUILD
          blame_info<<"GRAPH_(genEdge) - GEP "<<instName<<" points to "<<baseName<<endl;
#endif
          if (variables[baseName]->uniqueNameAsField.empty())
            variables[baseName]->uniqueNameAsField = getUpperLevelFieldName(baseName, pi, instName);
          else
            blame_info<<"Check(v already has uniqueNameAsField): old="<<variables[baseName]->uniqueNameAsField<<endl;
          
          structVarNameGlobal = variables[baseName]->uniqueNameAsField;
          blame_info<<"(0)base "<<baseName<<" has uniqueNmAsFld: "<<structVarNameGlobal<<endl;
        }
        else if (opCount == 1) {
          edge_type[ed] = GEP_OFFSET_OP;
#ifdef DEBUG_GRAPH_BUILD
          blame_info<<"GEP_OFFSET_OP: "<<v->getName().str()<<" to inst: "<<instName<<endl;
#endif
          ////added by Hui 01/14/16 not likely to happen
          if (pi->getNumOperands()==2) {//we are accessing an element in an array
            string upperStructName = getUpperLevelFieldName(structVarNameGlobal,pi,instName);
            upperStructName.insert(0,"I.");
            string strAlloc(upperStructName);
            variables[instName]->uniqueNameAsField = strAlloc;
            blame_info<<"(1)Inst "<<instName<<" has uniqueNameAsField: "<<strAlloc<<endl;
                      
            if (cpHash.count(strAlloc)) {
#ifdef DEBUG_GRAPH_COLLAPSE
              blame_info<<"Collapsable field already exists(1). Add inst name "<<instName<<" to collapsable pairs."<<endl;
#endif 
              CollapsePair *cp = new CollapsePair();
              cp->nameFieldCombo = strAlloc;
              cp->collapseVertex = variables[instName];
              cp->destVertex = cpHash[strAlloc];
              collapsePairs.push_back(cp);

              cp->collapseVertex->collapseTo = cp->destVertex;
              cp->destVertex->collapseNodes.insert(cp->collapseVertex);
   
#ifdef DEBUG_GRAPH_COLLAPSE
              blame_info<<"Remove edge(hui) from "<<instName<<" to "<<v->getName().data()<<" [3]"<<endl;
#endif 
              remove_edge(variables[instName]->number, variables[v->getName().str()]->number, G);//TOCHECK:do we need to remove_edge
            }
            else {
              //collapsableFields.insert(structVarName);
              cpHash[strAlloc] = variables[instName]; //cpHash only insert here
#ifdef DEBUG_GRAPH_COLLAPSE
              blame_info<<"Collapsable field doesn't exist(1). Create field and make inst name "<<instName<<" dest node."<<endl;
#endif 
            }
          }
        }
        else if (opCount == 2) {//barely happen in this case 
          edge_type[ed] = GEP_S_FIELD_VAR_OFF_OP;
#ifdef DEBUG_GRAPH_BUILD
          blame_info<<"GEP_S_FIELD_VAR_OFF_OP: "<<v->getName().data()<<" to inst: "<<instName<<endl;
#endif
        }
      }
      else {
#ifdef DEBUG_ERROR      
        cerr<<"Insertion fail in genEges for "<<instName<<" to "<<v->getName().data()<<endl;
#endif 
      }
    } //v hasName() = true
  
    else if (isa<ConstantInt>(v)) {
      if (opCount == 1) {
        if(pi->getNumOperands()==2){//we are accessing an element in an array, rarely happen in this case
          string upperStructName = getUpperLevelFieldName(structVarNameGlobal,pi,instName);
          upperStructName.insert(0,"I.");
          string strAlloc(upperStructName);
          variables[instName]->uniqueNameAsField = strAlloc;
          blame_info<<"(2)Inst "<<instName<<" has uniqueNameAsField: "<<strAlloc<<endl;

          if (cpHash.count(strAlloc)) {
#ifdef DEBUG_GRAPH_COLLAPSE
            blame_info<<"Collapsable field already exists(2).  Add inst name "<<instName<<" to collapsable pairs."<<endl;
#endif 
            CollapsePair *cp = new CollapsePair();
            cp->nameFieldCombo = strAlloc;
            cp->collapseVertex = variables[instName];
            cp->destVertex = cpHash[strAlloc];
            collapsePairs.push_back(cp);

            cp->collapseVertex->collapseTo = cp->destVertex;
            cp->destVertex->collapseNodes.insert(cp->collapseVertex);
          }
          else {
            cpHash[strAlloc] = variables[instName]; //cpHash only insert here
#ifdef DEBUG_GRAPH_COLLAPSE
            blame_info<<"Collapsable field doesn't exist(2).  Create field and make inst name "<<instName<<" dest node."<<endl;
#endif 
          }
        }
      } //we only deal with the 1st int operand if we are accessing array.

      else if (opCount == 2) {
        ConstantInt *cv = (ConstantInt *)v;
        int number = cv->getSExtValue();
        char tempBuf[64];
        sprintf (tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, opCount, Instruction::GetElementPtr);    
        char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
        strcpy(vN,tempBuf);
        vN[strlen(tempBuf)]='\0';
        string vNameStr(vN);
      
        if (variables.count(instName) && variables.count(vNameStr)) {
#ifdef DEBUG_GRAPH_BUILD
          blame_info<<"Adding edge from "<<instName<<" to "<<vNameStr<<endl;
#endif 
          tie(ed, inserted) = add_edge(variables[instName]->number,variables[vNameStr]->number,G);  
          if (inserted) {
            edge_type[ed] = GEP_S_FIELD_OFFSET_OP + number;
            //base should already have its uniqueNameAsField
            string structVarName = getUpperLevelFieldName(structVarNameGlobal,pi,instName);
            structVarName.insert(0, ".P.");
            char fieldNumStr[3];
            sprintf(fieldNumStr, "%d", number);
            structVarName.insert(0, fieldNumStr);
              
            string strAlloc(structVarName);
#ifdef DEBUG_GRAPH_COLLAPSE
            blame_info<<"Name of collapsable field candidate is "<<strAlloc<<" for "<<instName<<endl;
#endif 
            NodeProps *instNode = variables[instName];
            instNode->uniqueNameAsField = strAlloc;
#ifdef DEBUG_GRAPH_COLLAPSE
            blame_info<<"(3)Inst "<<instName<<" has uniqueNameAsField: "<<strAlloc<<endl;
#endif 

            if (cpHash.count(strAlloc)) {
#ifdef DEBUG_GRAPH_COLLAPSE
              blame_info<<"Collapsable field already exists.  Add inst name "<<instName<<" to collapsable pairs."<<endl;
#endif             
              CollapsePair *cp = new CollapsePair();
              cp->nameFieldCombo = strAlloc;
              cp->collapseVertex = variables[instName];
              cp->destVertex = cpHash[strAlloc];
              collapsePairs.push_back(cp);
            
              cp->collapseVertex->collapseTo = cp->destVertex;
              cp->destVertex->collapseNodes.insert(cp->collapseVertex);
#ifdef DEBUG_GRAPH_COLLAPSE //added remove[4.5]: when field redundant, we need to remove the edge from it to the base, either
              blame_info<<"Remove edge(hui) from "<<instName<<" to "<<vNameStr<<" [4]"<<endl;
#endif 
              remove_edge(variables[instName]->number,variables[vNameStr]->number,G);
            }
            else {
              cpHash[strAlloc] = variables[instName]; //cpHash only insert here
#ifdef DEBUG_GRAPH_COLLAPSE
              blame_info<<"Collapsable field doesn't exist.Create field and make inst name "<<instName<<" dest node."<<endl;
#endif 
            }
          }//insert edge succeed
          else {
#ifdef DEBUG_ERROR      
            cerr<<"Insertion fail, GEP struct field const(F)"<<" for "<<instName<<endl;
#endif 
          }
        }
    
        else {
#ifdef DEBUG_ERROR
          blame_info<<"Error adding edge with(unfound in variables) "<<vNameStr<<" "
            <<variables.count(vNameStr)<<" and "<<instName<<" "<<variables.count(instName)<<endl;
#endif 
          return instName;
        }        
      }//opCount=2
    }//if v is constantInt
  
    else if (isa<ConstantExpr>(v)) {
      ConstantExpr *ce = cast<ConstantExpr>(v);
      if (ce->getOpcode() == Instruction::GetElementPtr) {
        string opName = geGetElementPtr(ce, iSet, props, edge_type, currentLineNum);
        if (opName.empty()) 
          continue;
        
        // then we can treat v exactly as a register as the following
        if (variables.count(instName) > 0 && variables.count(opName) > 0) {
          tie(ed, inserted) = add_edge(variables[instName]->number,variables[opName]->number,G);
          if (inserted) {
            if (opCount == 0) {//here opName is the GEP baseName
              edge_type[ed] = GEP_BASE_OP;//pi->getOpcode();  
              variables[instName]->pointsTo = variables[opName];
              variables[opName]->pointedTo.insert(variables[instName]);
#ifdef DEBUG_GRAPH_BUILD
              blame_info<<"GRAPH_(genEdge) - GEP(2b) "<<instName<<" points to "<<opName<<endl;
#endif
              string upperStructName = getUpperLevelFieldName(opName,pi,instName);
              string strAlloc(upperStructName);
  
              if (variables[opName]->uniqueNameAsField.empty()) {//very likely, unless in 2nd GEP of two successive GEPs
                blame_info<<"(4b)base "<<opName<<" has uniqueNameAsField: "<<strAlloc<<endl;
                variables[opName]->uniqueNameAsField = strAlloc; 
              }
              else{
                blame_info<<"Base had uniqueNameAsField)(2b): old="<<variables[opName]->uniqueNameAsField<<" new="<<strAlloc<<endl;
                if(variables[opName]->uniqueNameAsField.compare(strAlloc)!=0) //shouldn't happen
                  blame_info<<"(ce)Error: register can't hold retval from 2 diff GEPs!"<<endl; 
              }    
              //set structVarNameGlobal here so later operand can see
              structVarNameGlobal = variables[opName]->uniqueNameAsField;
            }
            else if (opCount == 1) {
              edge_type[ed] = GEP_OFFSET_OP;
#ifdef DEBUG_GRAPH_BUILD
              blame_info<<"(ce)GEP_OFFSET_OP(2): "<<opName<<" to inst: "<<instName<<endl;
#endif
              if (pi->getNumOperands()==2){//we are accessing an element in an array
                string upperStructName = getUpperLevelFieldName(structVarNameGlobal,pi,instName);
                upperStructName.insert(0,"I.");
                string strAlloc(upperStructName);
                variables[instName]->uniqueNameAsField = strAlloc;
                blame_info<<"(5ce)Inst "<<instName<<" has uniqueNameAsField: "<<strAlloc<<endl;

                if (cpHash.count(strAlloc)) {
#ifdef DEBUG_GRAPH_COLLAPSE
                  blame_info<<"(ce)Collapsable field already exists(3).  Add inst name "<<instName<<" to collapsable pairs."<<endl;
#endif 
                  CollapsePair *cp = new CollapsePair();
                  cp->nameFieldCombo = strAlloc;
                  cp->collapseVertex = variables[instName];
                  cp->destVertex = cpHash[strAlloc];
                  collapsePairs.push_back(cp);
                     
                  cp->collapseVertex->collapseTo = cp->destVertex;
                  cp->destVertex->collapseNodes.insert(cp->collapseVertex);
#ifdef DEBUG_GRAPH_COLLAPSE
                  blame_info<<"(ce)Remove edge(hui) from "<<instName<<" to "<<opName<<" [5]"<<endl;
#endif 
                  remove_edge(variables[instName]->number,variables[opName]->number,G);//TOCHECK:whether we need to remove_edge
                }
                else {
                  cpHash[strAlloc] = variables[instName]; //cpHash only insert here
#ifdef DEBUG_GRAPH_COLLAPSE
                  blame_info<<"(ce)Collapsable field doesn't exist(1).  Create field and make inst name "<<instName<<" dest node."<<endl;
#endif 
                }
              }
            }
            else if (opCount == 2) {//barely happen in this case
              edge_type[ed] = GEP_S_FIELD_VAR_OFF_OP;
#ifdef DEBUG_GRAPH_BUILD
              blame_info<<"(ce)GEP_S_FIELD_VAR_OFF_OP(2): "<<opName<<" to inst: "<<instName<<endl;
#endif
            }
          }//insertion succeeds
          else { //not inserted
#ifdef DEBUG_ERROR      
            cerr<<"(ce)Insertion fail in genEges for "<<instName<<" to "<<opName<<endl;
#endif 
          }
        }//inst and op node found in variables
        else {  
#ifdef DEBUG_ERROR
          blame_info<<"(ce)Variables can't find value of "<<instName<<" "<<variables.count(instName);
          blame_info<<" or "<<opName<<" "<<variables.count(opName)<<endl;
#endif       
        }
      } //ce/v is GetElementPtr

      else if (ce->getOpcode() == Instruction::BitCast ||
               ce->getOpcode() == Instruction::AddrSpaceCast) {
        Value *vRepl = ce->getOperand(0);
        //Then we process the v exactly as if it's normal variable/register
        if (vRepl == NULL)
          continue;
        if (vRepl->hasName() && variables[instName] && variables[vRepl->getName().str()]) {
          tie(ed, inserted) = add_edge(variables[instName]->number,variables[vRepl->getName().str()]->number,G);
          if (inserted) {
            if (opCount == 0) {
              string baseName = vRepl->getName().str();
              edge_type[ed] = GEP_BASE_OP;//pi->getOpcode();  previously existed
              variables[instName]->pointsTo = variables[baseName];
              variables[baseName]->pointedTo.insert(variables[instName]);
#ifdef DEBUG_GRAPH_BUILD
              blame_info<<"(ce2)GRAPH_(genEdge) - GEP "<<instName<<" points to "<<baseName<<endl;
#endif
              if (variables[baseName]->uniqueNameAsField.empty())
                variables[baseName]->uniqueNameAsField = getUpperLevelFieldName(baseName, pi, instName);
              else
                blame_info<<"(ce2)Check(vRepl already has uniqueNameAsField): old="<<variables[baseName]->uniqueNameAsField<<endl;
              
              structVarNameGlobal = variables[baseName]->uniqueNameAsField;
              blame_info<<"(ce20)base "<<baseName<<" has uniqueNmAsFld: "<<structVarNameGlobal<<endl;
            }
            else if (opCount == 1) {
              edge_type[ed] = GEP_OFFSET_OP;
#ifdef DEBUG_GRAPH_BUILD
              blame_info<<"(ce2)GEP_OFFSET_OP: "<<vRepl->getName().str()<<" to inst: "<<instName<<endl;
#endif
              ////added by Hui 01/14/16 not likely to happen
              if (pi->getNumOperands()==2) {//we are accessing an element in an array
                string upperStructName = getUpperLevelFieldName(structVarNameGlobal,pi,instName);
                upperStructName.insert(0,"I.");
                string strAlloc(upperStructName);
                variables[instName]->uniqueNameAsField = strAlloc;
                blame_info<<"(1ce2)Inst "<<instName<<" has uniqueNameAsField: "<<strAlloc<<endl;
                          
                if (cpHash.count(strAlloc)) {
#ifdef DEBUG_GRAPH_COLLAPSE
                  blame_info<<"(ce2)Collapsable field already exists(1).  Add inst name "<<instName<<" to collapsable pairs."<<endl;
#endif 
                  CollapsePair *cp = new CollapsePair();
                  cp->nameFieldCombo = strAlloc;
                  cp->collapseVertex = variables[instName];
                  cp->destVertex = cpHash[strAlloc];
                  collapsePairs.push_back(cp);

                  cp->collapseVertex->collapseTo = cp->destVertex;
                  cp->destVertex->collapseNodes.insert(cp->collapseVertex);
       
#ifdef DEBUG_GRAPH_COLLAPSE
                  blame_info<<"(ce2)Remove edge(hui) from "<<instName<<" to "<<vRepl->getName().data()<<" [3]"<<endl;
#endif 
                  remove_edge(variables[instName]->number,variables[vRepl->getName().str()]->number,G);//TOCHECK:do we need to remove_edge
                }
                else {
                  //collapsableFields.insert(structVarName);
                  cpHash[strAlloc] = variables[instName]; //cpHash only insert here
#ifdef DEBUG_GRAPH_COLLAPSE
                  blame_info<<"(ce2)Collapsable field doesn't exist(1). Create field and make inst name "<<instName<<" dest node."<<endl;
#endif 
                }
              }
            }
            else if (opCount == 2) {//barely happen in this case 
              edge_type[ed] = GEP_S_FIELD_VAR_OFF_OP;
#ifdef DEBUG_GRAPH_BUILD
              blame_info<<"(ce2)GEP_S_FIELD_VAR_OFF_OP: "<<vRepl->getName().data()<<" to inst: "<<instName<<endl;
#endif
            }
          }
          else {
#ifdef DEBUG_ERROR      
            cerr<<"(ce2)Insertion fail in genEges for "<<instName<<" to "<<vRepl->getName().data()<<endl;
#endif 
          }
        } //vRepl hasName() = true
        else if (isa<ConstantInt>(vRepl)) {
#ifdef DEBUG_GRAPH_BUILD
          blame_info<<"sooo weird, vRepl in bit is constant int"<<endl;
#endif
          continue;
        } //shouldn't happen
        else { // vRepl is register
          char tempBuff[18];
          sprintf(tempBuff, "0x%x", /*(unsigned)*/vRepl);
          string opName(tempBuff);
          
          if (variables.count(instName) > 0 && variables.count(opName) > 0) {
            tie(ed, inserted) = add_edge(variables[instName]->number,variables[opName]->number,G);
            if (inserted) {
              if (opCount == 0) {//here opName is the GEP baseName
                edge_type[ed] = GEP_BASE_OP;//pi->getOpcode();  
                variables[instName]->pointsTo = variables[opName];
                variables[opName]->pointedTo.insert(variables[instName]);
#ifdef DEBUG_GRAPH_BUILD
                blame_info<<"(ce2)GRAPH_(genEdge) - GEP(2) "<<instName<<" points to "<<opName<<endl;
#endif
                string upperStructName = getUpperLevelFieldName(opName,pi,instName);
                string strAlloc(upperStructName);
      
                if (variables[opName]->uniqueNameAsField.empty()) {//very likely, unless in 2nd GEP of two successive GEPs
                  blame_info<<"(ce24)base "<<opName<<" has uniqueNameAsField: "<<strAlloc<<endl;
                  variables[opName]->uniqueNameAsField = strAlloc; 
                }
                else{
                  blame_info<<"(ce2)Base had uniqueNameAsField)(2): old="<<variables[opName]->uniqueNameAsField<<" new="<<strAlloc<<endl;
                  if(variables[opName]->uniqueNameAsField.compare(strAlloc)!=0) //shouldn't happen
                    blame_info<<"(ce2)Error: register can't hold retval from 2 diff GEPs!"<<endl; 
                }    
                //set structVarNameGlobal here so later operand can see
                structVarNameGlobal = variables[opName]->uniqueNameAsField;
              }
              else if (opCount == 1) {
                edge_type[ed] = GEP_OFFSET_OP;
#ifdef DEBUG_GRAPH_BUILD
                blame_info<<"(ce2)GEP_OFFSET_OP(2): "<<opName<<" to inst: "<<instName<<endl;
#endif
                if (pi->getNumOperands()==2){//we are accessing an element in an array
                  string upperStructName = getUpperLevelFieldName(structVarNameGlobal,pi,instName);
                  upperStructName.insert(0,"I.");
                  string strAlloc(upperStructName);
                  variables[instName]->uniqueNameAsField = strAlloc;
                  blame_info<<"(ce25)Inst "<<instName<<" has uniqueNameAsField: "<<strAlloc<<endl;

                  if (cpHash.count(strAlloc)) {
#ifdef DEBUG_GRAPH_COLLAPSE
                    blame_info<<"(ce2)Collapsable field already exists(3).  Add inst name "<<instName<<" to collapsable pairs."<<endl;
#endif 
                    CollapsePair *cp = new CollapsePair();
                    cp->nameFieldCombo = strAlloc;
                    cp->collapseVertex = variables[instName];
                    cp->destVertex = cpHash[strAlloc];
                    collapsePairs.push_back(cp);
                         
                    cp->collapseVertex->collapseTo = cp->destVertex;
                    cp->destVertex->collapseNodes.insert(cp->collapseVertex);
#ifdef DEBUG_GRAPH_COLLAPSE
                    blame_info<<"(ce2)Remove edge(hui) from "<<instName<<" to "<<opName<<" [5]"<<endl;
#endif 
                    remove_edge(variables[instName]->number,variables[opName]->number,G);//TOCHECK:whether we need to remove_edge
                  }
                  else {
                    cpHash[strAlloc] = variables[instName]; //cpHash only insert here
#ifdef DEBUG_GRAPH_COLLAPSE
                    blame_info<<"(ce2)Collapsable field doesn't exist(1).  Create field and make inst name "<<instName<<" dest node."<<endl;
#endif 
                  }
                }
              }
              else if (opCount == 2) {//barely happen in this case
                edge_type[ed] = GEP_S_FIELD_VAR_OFF_OP;
#ifdef DEBUG_GRAPH_BUILD
                blame_info<<"(ce2)GEP_S_FIELD_VAR_OFF_OP(2): "<<opName<<" to inst: "<<instName<<endl;
#endif
              }
            }//insertion succeeds
            else { //not inserted
#ifdef DEBUG_ERROR      
              cerr<<"(ce2)Insertion fail in genEges for "<<instName<<" to "<<opName<<endl;
#endif 
            }
          }//inst and op node found in variables
          else {  
#ifdef DEBUG_ERROR
            blame_info<<"(ce2)Variables can't find value of "<<instName<<" "<<variables.count(instName);
            blame_info<<" or "<<opName<<" "<<variables.count(opName)<<endl;
#endif       
          }
        }// vRepl is register 
      } // v is BitCast or AddrSpaceCast
    } // v is ConstantExpr
    
    else { //if v is register
      char tempBuff[18];
      sprintf(tempBuff, "0x%x", /*(unsigned)*/v);
      string opName(tempBuff);
      
      if (variables.count(instName) > 0 && variables.count(opName) > 0) {
        tie(ed, inserted) = add_edge(variables[instName]->number,variables[opName]->number,G);
        if (inserted) {
          if (opCount == 0) {//here opName is the GEP baseName
            edge_type[ed] = GEP_BASE_OP;//pi->getOpcode();  
            variables[instName]->pointsTo = variables[opName];
            variables[opName]->pointedTo.insert(variables[instName]);
#ifdef DEBUG_GRAPH_BUILD
            blame_info<<"GRAPH_(genEdge) - GEP(2) "<<instName<<" points to "<<opName<<endl;
#endif
            string upperStructName = getUpperLevelFieldName(opName,pi,instName);
            string strAlloc(upperStructName);
  
            if (variables[opName]->uniqueNameAsField.empty()) {//very likely, unless in 2nd GEP of two successive GEPs
              blame_info<<"(4)base "<<opName<<" has uniqueNameAsField: "<<strAlloc<<endl;
              variables[opName]->uniqueNameAsField = strAlloc; 
            }
            else{
              blame_info<<"Base had uniqueNameAsField)(2): old="<<variables[opName]->uniqueNameAsField<<" new="<<strAlloc<<endl;
              if(variables[opName]->uniqueNameAsField.compare(strAlloc)!=0) //shouldn't happen
                blame_info<<"Error: register can't hold retval from 2 diff GEPs!"<<endl; 
            }    
            //set structVarNameGlobal here so later operand can see
            structVarNameGlobal = variables[opName]->uniqueNameAsField;
          }
          else if (opCount == 1) {
            edge_type[ed] = GEP_OFFSET_OP;
#ifdef DEBUG_GRAPH_BUILD
            blame_info<<"GEP_OFFSET_OP(2): "<<opName<<" to inst: "<<instName<<endl;
#endif
            if (pi->getNumOperands()==2){//we are accessing an element in an array
              string upperStructName = getUpperLevelFieldName(structVarNameGlobal,pi,instName);
              upperStructName.insert(0,"I.");
              string strAlloc(upperStructName);
              variables[instName]->uniqueNameAsField = strAlloc;
              blame_info<<"(5)Inst "<<instName<<" has uniqueNameAsField: "<<strAlloc<<endl;

              if (cpHash.count(strAlloc)) {
#ifdef DEBUG_GRAPH_COLLAPSE
                blame_info<<"Collapsable field already exists(3).  Add inst name "<<instName<<" to collapsable pairs."<<endl;
#endif 
                CollapsePair *cp = new CollapsePair();
                cp->nameFieldCombo = strAlloc;
                cp->collapseVertex = variables[instName];
                cp->destVertex = cpHash[strAlloc];
                collapsePairs.push_back(cp);
                     
                cp->collapseVertex->collapseTo = cp->destVertex;
                cp->destVertex->collapseNodes.insert(cp->collapseVertex);
#ifdef DEBUG_GRAPH_COLLAPSE
                blame_info<<"Remove edge(hui) from "<<instName<<" to "<<opName<<" [5]"<<endl;
#endif 
                remove_edge(variables[instName]->number,variables[opName]->number,G);//TOCHECK:whether we need to remove_edge
              }
              else {
                cpHash[strAlloc] = variables[instName]; //cpHash only insert here
#ifdef DEBUG_GRAPH_COLLAPSE
                blame_info<<"Collapsable field doesn't exist(1).  Create field and make inst name "<<instName<<" dest node."<<endl;
#endif 
              }
            }
          }
          else if (opCount == 2) {//barely happen in this case
            edge_type[ed] = GEP_S_FIELD_VAR_OFF_OP;
#ifdef DEBUG_GRAPH_BUILD
            blame_info<<"GEP_S_FIELD_VAR_OFF_OP(2): "<<opName<<" to inst: "<<instName<<endl;
#endif
          }
        }//insertion succeeds
        else { //not inserted
#ifdef DEBUG_ERROR      
          cerr<<"Insertion fail in genEges for "<<instName<<" to "<<opName<<endl;
#endif 
        }
      }//inst and op node found in variables
      else {  
#ifdef DEBUG_ERROR
        blame_info<<"Variables can't find value of "<<instName<<" "<<variables.count(instName);
        blame_info<<" or "<<opName<<" "<<variables.count(opName)<<endl;
#endif       
      }
    }// op is register 
      
    //update opCount here once
    opCount++;
  } // end for 
  
  return instName;
} 


// For now (07/07/17) We just treat extractvalue inst exactly the same as GEP, including collapsable pairs
// and uniqueFieldName construction. Basically we need to treat (ld+extracValue) equal to GEP
string FunctionBFC::geExtractValue(User *pi, set<const char*, ltstr> &iSet,property_map<MyGraphType, 
        vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type,int &currentLineNum)
{
  bool inserted;
  graph_traits < MyGraphType >::edge_descriptor ed;
  
  addImplicitEdges(pi, iSet, edge_type, "", false);
  
  string instName;
  char tempBuf[18];
  if (pi->hasName()) {
    instName = pi->getName().str();
  }
  else {
    sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
    string tempStr(tempBuf);
    instName.insert(0, tempStr);      
    
    if (isa<ConstantExpr>(pi)) {
      instName += ".CE";
      char tempBuf2[10];
      sprintf(tempBuf2, ".%d", currentLineNum);
      instName.append(tempBuf2);
    }
  }
  
  // Name of struct variable name (essentially op0)
  string structVarNameGlobal;
  int opCount = 0;
  // Go through the operands
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
    if (v->hasName() && variables[instName] && variables[v->getName().str()]) {
      tie(ed, inserted) = add_edge(variables[instName]->number,variables[v->getName().str()]->number,G);
      if (inserted) {
        if (opCount == 0) {
          string baseName = v->getName().str();
          edge_type[ed] = GEP_BASE_OP;//pi->getOpcode();  previously existed
          variables[instName]->pointsTo = variables[baseName];
          variables[baseName]->pointedTo.insert(variables[instName]);
#ifdef DEBUG_GRAPH_BUILD
          blame_info<<"GRAPH_(genEdge) - exv "<<instName<<" points to "<<baseName<<endl;
#endif
          if(variables[baseName]->uniqueNameAsField.empty())
            variables[baseName]->uniqueNameAsField = getUpperLevelFieldName(baseName, pi, instName);
          else
            blame_info<<"exv Check(v already has uniqueNameAsField): old="<<variables[baseName]->uniqueNameAsField<<endl;
          
          structVarNameGlobal = variables[baseName]->uniqueNameAsField;
          blame_info<<"exv (0)base "<<baseName<<" has uniqNmAFld: "<<structVarNameGlobal<<endl;
        }
        else {//barely happen in this case: opCout>=1 and it has name
          edge_type[ed] = GEP_S_FIELD_VAR_OFF_OP;
#ifdef DEBUG_GRAPH_BUILD
          blame_info<<"exv GEP_S_FIELD_VAR_OFF_OP: "<<v->getName().data()<<" to inst: "<<instName<<endl;
#endif
        }
      }
      else {
#ifdef DEBUG_ERROR      
        cerr<<"exv Insertion fail in genEges for "<<instName<<" to "<<v->getName().data()<<endl;
#endif 
      }
    } //v hasName() = true
  
    else if (isa<ConstantInt>(v)) {
      if (opCount == 1) { //currently we don't handle more than 1 indice for exv
        ConstantInt *cv = (ConstantInt *)v;
        int number = cv->getSExtValue();
        char tempBuf[64];
        sprintf (tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, opCount, Instruction::GetElementPtr);    
        char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
        strcpy(vN,tempBuf);
        vN[strlen(tempBuf)]='\0';
        string vNameStr(vN);
      
        if (variables.count(instName) && variables.count(vNameStr)) {
          tie(ed, inserted) = add_edge(variables[instName]->number,variables[vNameStr]->number,G);  
          if (inserted) {
            edge_type[ed] = GEP_S_FIELD_OFFSET_OP + number;
            //base should already have its uniqueNameAsField
            string structVarName = getUpperLevelFieldName(structVarNameGlobal,pi,instName);
            structVarName.insert(0, ".P.");
            char fieldNumStr[3];
            sprintf(fieldNumStr, "%d", number);
            structVarName.insert(0, fieldNumStr);
              
            string strAlloc(structVarName);
#ifdef DEBUG_GRAPH_COLLAPSE
            blame_info<<"exv Name of collapsable field candidate is "<<strAlloc<<" for "<<instName<<endl;
#endif 
            NodeProps *instNode = variables[instName];
            instNode->uniqueNameAsField = strAlloc;
            blame_info<<"exv (3)Inst "<<instName<<" has uniqueNameAsField: "<<strAlloc<<endl;

            if (cpHash.count(strAlloc)) {
#ifdef DEBUG_GRAPH_COLLAPSE
              blame_info<<"exv Collapsable field already exists.  Add inst name "<<instName<<" to collapsable pairs."<<endl;
#endif             
              CollapsePair *cp = new CollapsePair();
              cp->nameFieldCombo = strAlloc;
              cp->collapseVertex = variables[instName];
              cp->destVertex = cpHash[strAlloc];
              collapsePairs.push_back(cp);
            
              cp->collapseVertex->collapseTo = cp->destVertex;
              cp->destVertex->collapseNodes.insert(cp->collapseVertex);
#ifdef DEBUG_GRAPH_COLLAPSE //added remove[4.5]: when field redundant, we need to remove the edge from it to the base, either
              blame_info<<"exv Remove edge(hui) from "<<instName<<" to "<<vNameStr<<" [4]"<<endl;
#endif 
              remove_edge(variables[instName]->number,variables[vNameStr]->number,G);
            }
            else {
              cpHash[strAlloc] = variables[instName]; //cpHash only insert here
#ifdef DEBUG_GRAPH_COLLAPSE
              blame_info<<"exv Collapsable field doesn't exist.Create field and make inst name "<<instName<<" dest node."<<endl;
#endif 
            }
          }//insert edge succeed
          else {
#ifdef DEBUG_ERROR      
           cerr<<"exv Insertion fail, GEP struct field const(F)"<<" for "<<instName<<endl;
#endif 
          }
        }
        else {
#ifdef DEBUG_ERROR
          blame_info<<"exv Error adding edge with(unfound in variables) "<<vNameStr<<" "
           <<variables.count(vNameStr)<<"/"<<instName<<" "<<variables.count(instName)<<endl;
#endif 
        }        
      }//opCount=1
    }//if v is constantInt

    else { //if v is register
      char tempBuff[18];
      sprintf(tempBuff, "0x%x", /*(unsigned)*/v);
      string opName(tempBuff);
     
      if (variables.count(instName) > 0 && variables.count(opName) > 0) {
        tie(ed, inserted) = add_edge(variables[instName]->number,variables[opName]->number,G);
        if (inserted) {
          if (opCount == 0) {
            edge_type[ed] = GEP_BASE_OP;//pi->getOpcode();  
            variables[instName]->pointsTo = variables[opName];
            variables[opName]->pointedTo.insert(variables[instName]);
#ifdef DEBUG_GRAPH_BUILD
            blame_info<<"exv GRAPH_(genEdge) - GEP(2) "<<instName<<" points to "<<opName.c_str()<<endl;
#endif
            string upperStructName = getUpperLevelFieldName(opName, pi, instName);
            string strAlloc(upperStructName);
  
            if (variables[opName]->uniqueNameAsField.empty()) {//very likely, unless in 2nd GEP of two successive GEPs
              blame_info<<"exv (4)base "<<opName<<" has uniqueNameAsField: "<<strAlloc<<endl;
              variables[opName]->uniqueNameAsField = strAlloc; 
            }
            else {
              blame_info<<"exv Base had uniNmeAsFld)(2): old="<<variables[opName]->uniqueNameAsField<<" new="<<strAlloc<<endl;
              if (variables[opName]->uniqueNameAsField.compare(strAlloc)!=0) //shouldn't happen
                blame_info<<"exv Error: register can't hold retval from 2 diff GEPs!"<<endl; 
            }    
            //set structVarNameGlobal here so other operand can see later
            structVarNameGlobal = variables[opName]->uniqueNameAsField;
          }
          else if (opCount >= 1) {//barely happen in this case
            edge_type[ed] = GEP_S_FIELD_VAR_OFF_OP;
#ifdef DEBUG_GRAPH_BUILD
            blame_info<<"exv GEP_S_FIELD_VAR_OFF_OP(2): "<<opName<<" to inst: "<<instName<<endl;
#endif
          }
        }//insertion succeeds
        else { //not inserted
#ifdef DEBUG_ERROR      
          cerr<<"exv Insertion fail in genEges for "<<instName<<" to "<<opName<<endl;
#endif 
        }
      }//inst and op node found in variables
      else {  
#ifdef DEBUG_ERROR
        blame_info<<"exv Variables can't find value of "<<instName<<" "<<variables.count(instName);
        blame_info<<" or "<<opName<<" "<<variables.count(opName)<<endl;
#endif       
      }
    }// op is register 
      
    //update opCount here once
    opCount++;
  } // end for 
  
  return instName;
} 


string FunctionBFC::geInsertValue(User *pi, set<const char*, ltstr> &iSet,property_map<MyGraphType, 
        vertex_props_t>::type props, property_map<MyGraphType, edge_iore_t>::type edge_type,int &currentLineNum)
{
  blame_info<<"HELP! Pull me out here cuz I don't know what to do!"<<endl;
  return string("InsertValue");
} 


void FunctionBFC::geCallWrapFunc(Instruction *pi, set<const char*, ltstr> &iSet,
              property_map<MyGraphType, vertex_props_t>::type props,
              property_map<MyGraphType, edge_iore_t>::type edge_type,
              int &currentLineNum, set<NodeProps *> &seenCall)
{
    
  Value *lastOp = pi->getOperand(pi->getNumOperands()-1);
  blame_info<<"Entering geCallWrapFunc for "<<lastOp->getName().str()<<endl;
  //First, we still need to parse executeON for fid and arg value
  int fidHolder = -1;
  NodeProps *argHolder = NULL;

  if (pi->getNumOperands() < 4) {
    blame_info<<"Error: pi has not enough params"<<endl;
    return;
  }
  //get the fid value first
  Value *fid = pi->getOperand(1);
  if (isa<ConstantInt>(fid)) {
    ConstantInt *fidVal = cast<ConstantInt>(fid);
    fidHolder = (int)(fidVal->getZExtValue());
  }
  if (fidHolder == -1) {
    blame_info<<"Error: can't retrive fid from "<<lastOp->getName().str()<<endl;
    return;
  }
  string wrapName = (this->getModuleBFC()->funcPtrTable)[fidHolder];
  string realName;
  if (wrapName.find("wrap") == 0) 
    realName = wrapName.substr(4); //start from 4th char (chopped "wrap" from head)
  else if (wrapName.find("_local_wrap") == 0) {
    realName = wrapName.substr(11); //get everything after _local_wrap
    realName.insert(0, "_local_");
  }
  else
    blame_info<<"Weird: check what's the wrapName: "<<wrapName<<endl;

  //get the function prototype
  if (knownFuncsInfo.count(realName) == 0) {
    blame_info<<realName<<" isn't from user module, we don't deal with this func call"<<endl;
    return;
  }
  FuncSignature *Func = knownFuncsInfo[realName];
  const int numArgs = Func->args.size();
  Value **params = new Value* [numArgs]; //storage for all real params for on/coforall_fn_chpl*
  for (int i = 0; i < numArgs; i++)
    params[i] = NULL;
  
  //get all params value second (!CORE of this function!)
  if (lastOp->getName().str().find("chpl_executeOn")==0)
    getParamsForOn(pi, params, numArgs, Func->args);
  
  else if (lastOp->getName().str().find("chpl_taskListAddBegin")==0 || 
           lastOp->getName().str().find("chpl_taskListAddCoStmt")==0) 
    getParamsForCoforall(pi, params, numArgs, Func->args);

  
  //check the completeness of params
  bool complete = true;
  for (int i=0; i<numArgs; i++) {
    if (params[i] == NULL) {
      blame_info<<"Error: params for on are incomplete: param#"<<i<<endl;
      complete = false;
    }
  }

  //get the names of call and arg
  int sizeSuffix = 0;
  string tempStr;

  while (1) {
    tempStr.clear();
    char tempBuf[1024];
    sprintf (tempBuf, "%s--%i", realName.c_str(), currentLineNum);    
    char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
   
    strcpy(vN,tempBuf);
    vN[strlen(tempBuf)]='\0';
    
    tempStr.insert(0,vN);
    for (int a = 0; a < sizeSuffix; a++) {
      tempStr.push_back('a');
    }
    
    //cout<<vNTemp<<" "<<sizeSuffix<<endl;
  if (variables.count(tempStr) > 0) {
    NodeProps *vpTemp = variables[tempStr];
  
    if (seenCall.count(vpTemp)) {
      sizeSuffix++;
      continue;
    }
    else {
      seenCall.insert(vpTemp);
      break;
    }
  }
  else
    break;
  }
  // tempStr is the mangled call name
  string callName = tempStr; //calculated callName, with line#
                                        //like: child--22a
  // add implicit edges for the call node
  if (variables.count(callName)) 
    addImplicitEdges(pi, iSet, edge_type, callName, true);
  else {
    blame_info<<"Error: can't find callName: "<<callName<<" in variables"<<endl;
    return;
  }
  // add edges between params and call node
  if (complete) {
    for (int i=0; i<numArgs; i++) {
      Value *param = params[i];
      string paramName;
      if (param->hasName()) // not likely
        paramName = param->getName().str();
      else {
        if (!isa<ConstantInt>(param) && !isa<ConstantFP>(param) && 
            !isa<UndefValue>(param) && !isa<ConstantPointerNull>(param)) {//v isn't a constant value
          char tempBuf2[20];
          sprintf(tempBuf2, "0x%x", /*(unsigned)*/param);
          paramName = string(tempBuf2);
        }
      }

    if (!paramName.empty() && variables.count(paramName) > 0) {  
        // Add edge between arg and wrapOn_fn_chpl*
        addEdge(paramName, callName, pi, 3);
      }
      else
        blame_info<<"Error: can't find param#"<<i<<" "<<paramName<<" in variables"<<endl;
    }
  }

  //delete the params' holder
  delete[] params;
}

void FunctionBFC::geCall(Instruction *pi, set<const char*, ltstr> &iSet,
              property_map<MyGraphType, vertex_props_t>::type props,
              property_map<MyGraphType, edge_iore_t>::type edge_type,
              int &currentLineNum, set<NodeProps *> &seenCall)
{
  //bool inserted;
  graph_traits < MyGraphType >::edge_descriptor ed;
  /////added by Hui,trying to get the called function///////////
  llvm::CallInst *cpi = cast<CallInst>(pi);
  llvm::Function *calledFunc = cpi->getCalledFunction();
  if (calledFunc != NULL && calledFunc->hasName())
    blame_info<<"In geCall, calledFunc's name = "<<calledFunc->getName().data();
 
  blame_info<<"  pi->getNumOperands()="<<pi->getNumOperands()<<endl;
  //////////////////////////////////////////////////////////
  int callNameIdx = pi->getNumOperands()-1; //called func is the last operand of this inst
  Value *call_name = pi->getOperand(callNameIdx);
  const char *callNameStr;
  bool isTradName = true;
  if (call_name->hasName()) {
    callNameStr = call_name->getName().data();
#ifdef SPECIAL_FUNC_PTR
    string fn = call_name->getName().str();
    if (fn.find("llvm.nvvm") == 0) {
      if (fn.find("llvm.nvvm.read") != string::npos) {
        blame_info<<"Not deal with special ptx registers"<<endl;
        return;
      }

      if (pi->hasNUsesOrMore(1)) {
        geDefaultPTXIntrinsic(pi, iSet, props, edge_type, currentLineNum);
      }
      else {
        blame_info<<fn<<" has no uses"<<endl;
      }
  
      return; //IMPORTANT 
    }
      
    //not process other llvm instrinsics
    else if (fn.find("llvm.dbg") != string::npos) {
      blame_info<<"Not deal with intrinsic func calls"<<endl;
      return;
    }
#endif
  }
  else if (isa<ConstantExpr>(call_name)) {
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"call_name is ConstantExpr"<<endl;
#endif
    ConstantExpr *ce = cast<ConstantExpr>(call_name);
    User::op_iterator op_i = ce->op_begin();
    for (; op_i != ce->op_end(); op_i++) {
      Value *funcVal = op_i->get();      
      if (isa<Function>(funcVal)) {
        Function *embFunc = cast<Function>(funcVal);
        //blame_info<<"Func "<<embFunc->getName()<<endl;
        isTradName = false;
        callNameStr = embFunc->getName().data();
        break;
      }
    }
  }
  //TODO: Following case is funcCall is a register, 
  //most likely is the object functions
  //NEED to figure out a way to handle
  // (!call_name->hasName() && !isa<ConstantExpr>(call_name)) 
  else {
    blame_info<<"We met a func call with no name !(register)"<<endl;
    return;
  }
  
  //bool foundFunc = true;
  int sizeSuffix = 0;
  string tempStr;
  
  while (1) {
    tempStr.clear();
    char tempBuf[1024];
    sprintf (tempBuf, "%s--%i", callNameStr, currentLineNum);    
    char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
    
    strcpy(vN,tempBuf);
    vN[strlen(tempBuf)]='\0';
  
    tempStr.insert(0,vN);
    for (int a = 0; a < sizeSuffix; a++) {
      tempStr.push_back('a');
    }

    //cout<<vNTemp<<" "<<sizeSuffix<<endl;
    if (variables.count(tempStr) > 0) {
      NodeProps *vpTemp = variables[tempStr];
      if (seenCall.count(vpTemp)) {
        sizeSuffix++;
        continue;
      }
      else {
        seenCall.insert(vpTemp);
        break;
      }
    }

    else
      break;
  }
  
  string vName = tempStr; //calculated callName, with line#
                         //like: child--22a
  // Add edge for the return value container 
  // call_name - function call name
  // pi - variable that receives return value, if none then function is void
  string retName;
  if (pi->hasName()) {
    retName = pi->getName().str();
    addEdge(retName, vName, pi, 4);
  }
  else if (pi->hasNUsesOrMore(1)) {
    char tempBuf2[20];
    sprintf(tempBuf2, "0x%x", /*(unsigned)*/pi);
    string name(tempBuf2);
    retName = name;
    addEdge(retName, vName, pi, 5);
  }
  
  // Take Care of Implicit Edges
  if ((call_name->hasName() || !isTradName) && variables.count(vName))   
    addImplicitEdges(pi, iSet, edge_type, vName, true);
  
  // Add edges for all parameters of the function          
  int opCount = 0;
  string opName;
  User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end();
  for ( ; op_i != op_e; ++op_i) {
    Value *v = op_i->get();
    //add edges from operands to the call node
    if (opCount != callNameIdx) {
#ifdef DEBUG_GRAPH_BUILD
      blame_info<<"op["<<opCount<<"]: "<<endl;
#endif
      // Normal case where the parameter has a name and we have a normal function
      if (v->hasName()) {            
        opName = v->getName().str();
        addEdge(opName, vName, pi, 6);
      }
      // corner case where parameter is a constantExpr (GEP or BitCast)
      else if (isa<ConstantExpr>(v)) {
#ifdef DEBUG_GRAPH_BUILD
        blame_info<<"Param "<<opCount<<" for "<<vName<<" is a constantExpr"<<endl;
#endif 
        ConstantExpr *ce = cast<ConstantExpr>(v);  
        if (ce->getOpcode() == Instruction::GetElementPtr) {
          Value *vReplace = ce->getOperand(0);
          opName = getNameForVal(vReplace);
            
          if (call_name->hasName() || !isTradName)
            addEdge(opName, vName, pi, 7);
        } // end check to see if param is GEP instruction
        else if (ce->getOpcode() == Instruction::BitCast ||
                 ce->getOpcode() == Instruction::AddrSpaceCast) {
          Value *vReplace = ce->getOperand(0);
          opName = getNameForVal(vReplace);
            
          if (call_name->hasName() || !isTradName)
            addEdge(opName, vName, pi, 8);
        } // end check to see if param is a BitCast Instruction
      } // end if ValueID is a ConstantExprVal
      // Case where params have no name (FORTRAN support)
      else if (!isa<ConstantInt>(v) && !isa<ConstantFP>(v) && 
               !isa<UndefValue>(v) && !isa<ConstantPointerNull>(v)) {
        opName = getNameForVal(v);   
        addEdge(opName, vName, pi, 11);
      }
    }
      
    opCount++;
  }// end for loop going through ops

  //Process Chapel runtime function calls, to be modified for CUDA
  string cN(callNameStr);
  int specialCall = needExProc(cN);
  if (specialCall) {
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"We need to take special take of the call: "<<cN<<endl;
#endif
    specialProcess(pi, specialCall, cN);
  }
}

void FunctionBFC::geLoad(Instruction *pi, set<const char*, ltstr> &iSet,
            property_map<MyGraphType, vertex_props_t>::type props,
            property_map<MyGraphType, edge_iore_t>::type edge_type,
            int & currentLineNum)
{  
  string instName;
  char tempBuf[18];
  
  if (pi->hasName()) {
    instName = pi->getName();
  }
  else {
    sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
    string tempStr(tempBuf);
    instName.insert(0, tempStr);      
  }
  
  //get the address from where you can load the value
  Value *v = pi->getOperand(0);
    
  if (v->hasName()) {
    if (variables.count(v->getName().str()) && variables.count(instName)) {
      addEdge(instName, v->getName().str(), pi, 12);
    }
  }

  else if (isa<ConstantExpr>(v)) { //v doesn't have a name
    ConstantExpr *ce = cast<ConstantExpr>(v);  
    if (ce->getOpcode() == Instruction::GetElementPtr) {
      string opName = geGetElementPtr(ce, iSet, props, edge_type, currentLineNum);
      if (opName.empty())
        return;
      addEdge(instName, opName, pi, 13);
    }
    else if (ce->getOpcode() == Instruction::BitCast ||
             ce->getOpcode() == Instruction::AddrSpaceCast) {
      Value *vRepl = ce->getOperand(0);
      if (vRepl == NULL) {
#ifdef DEBUG_ERROR
        blame_info<<"v bitcast get NULL in geLoad"<<endl;
#endif 
        return;
      }
            
      string vReplName;
      if (vRepl->hasName())
        vReplName = vRepl->getName().str();
      else {
        sprintf(tempBuf, "0x%x", vRepl);
        string tempStr(tempBuf);
        vReplName.insert(0, tempStr);
      }
      //we don't need to judge whether both nodes are in variables since it'll be judged inside "addEdge"
      addEdge(instName, vReplName, pi, 14);
    }  
  }

  else { //v doesn't have a name and not a constantExpr, most common
    string opName;  
    sprintf(tempBuf, "0x%x", /*(unsigned)*/v);
    string tempStr(tempBuf);    
    opName.insert(0, tempStr);
        
    addEdge(instName, opName, pi, 15);
  }
}    


void FunctionBFC::geMemAtomic(Instruction *pi, set<const char*, ltstr> &iSet, 
                            property_map<MyGraphType, vertex_props_t>::type props, 
                            property_map<MyGraphType, edge_iore_t>::type edge_type,  
                            int &currentLineNum)
{
  //TODO: Add edges between pi and ops: mem atomic = load+store
#ifdef DEBUG_GRAPH_BUILD
  blame_info<<"gen edges for MemAtomic Inst: "<<pi->getOpcodeName()<<" at line"<<currentLineNum<<endl;
#endif

  if (pi->getOpcode() == Instruction::AtomicCmpXchg) {
    User::op_iterator op_i = pi->op_begin();
    Value *first = op_i->get(); //mem location: ptr
    op_i++;
    Value *second = op_i->get(); //comp
    op_i++;
    Value *third = op_i->get(); //value
    
    geAtomicLoadPart(first, pi, iSet, props, edge_type, currentLineNum);
    geAtomicStorePart(first, third, pi, iSet, props, edge_type, currentLineNum);
        
    //add edge between comp and ptr
    char tempBuf[18];

    string cmpName;
    if (second->hasName())
      cmpName = second->getName().str();
    else if (isa<ConstantInt>(second)) {
      ConstantInt *cv = (ConstantInt *)second;
      int number = cv->getSExtValue();
            
      char tempBuf[64];
      sprintf(tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, 0, pi->getOpcode());    
      char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
        
      strcpy(vN,tempBuf);
      vN[strlen(tempBuf)]='\0';
      const char *vName = vN;
            
      cmpName.insert(0, vName);
    } 
    else {
      sprintf(tempBuf, "0x%x", /*(unsigned)*/second);
      string tempStr(tempBuf);
      cmpName.insert(0, tempStr);
    }
        
    // Now we just need to find the correct ptrName and addEdge !
    string ptrName;

    if (first->hasName()) {
      ptrName = first->getName().str();
      if (variables.count(ptrName) && variables.count(cmpName)) {
        addEdge(ptrName, cmpName, pi, 16);
      }
    }
    else if (isa<ConstantExpr>(first)) { //first/ptr doesn't have a name
      ConstantExpr *ce = cast<ConstantExpr>(first);  
      if (ce->getOpcode() == Instruction::GetElementPtr) {
        ptrName = geGetElementPtr(ce, iSet, props, edge_type, currentLineNum);
        if (ptrName.empty())
          return;
        if (variables.count(ptrName) && variables.count(cmpName)) {
          addEdge(ptrName, cmpName, pi, 17);
        }
      }
      else if (ce->getOpcode() == Instruction::BitCast || 
               ce->getOpcode() == Instruction::AddrSpaceCast) {
        Value *vRepl = ce->getOperand(0);
        if (vRepl == NULL) 
          return;
           
        if (vRepl->hasName())
          ptrName = vRepl->getName().str();
        else {
          sprintf(tempBuf, "0x%x", vRepl);
          string tempStr(tempBuf);
          ptrName.insert(0, tempStr);
        }

        if (variables.count(ptrName) && variables.count(cmpName)) {
          addEdge(ptrName, cmpName, pi, 18);
        }
      }  
    }
    else { //first doesn't have a name and not a constantExpr, most common
      sprintf(tempBuf, "0x%x", /*(unsigned)*/first);
      string tempStr(tempBuf);    
      ptrName.insert(0, tempStr);
            
      if (variables.count(ptrName) && variables.count(cmpName)) {
        addEdge(ptrName, cmpName, pi, 19);
      }
    }
  } //end of if Opcode = AtomicCmpXchg

  else if (pi->getOpcode() == Instruction::AtomicRMW) {
    User::op_iterator op_i = pi->op_begin();
    Value *first = op_i->get(); // ptr 
    op_i++;
    Value *second = op_i->get(); //value
    
    geAtomicLoadPart(first, pi, iSet, props, edge_type, currentLineNum);
    geAtomicStorePart(first, second, pi, iSet, props, edge_type, currentLineNum);
  }
}


void FunctionBFC::geAtomicLoadPart(Value *ptr, 
                            Instruction *pi, set<const char*, ltstr> &iSet, 
                            property_map<MyGraphType, vertex_props_t>::type props, 
                            property_map<MyGraphType, edge_iore_t>::type edge_type,  
                            int &currentLineNum)
{
  string instName;
  char tempBuf[18];
  
  if (pi->hasName()) {
    instName = pi->getName();
  }
  else {
    sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
    string tempStr(tempBuf);
    instName.insert(0, tempStr);      
  }
  
  //// start adding edges //////
  if (ptr->hasName()) {
    if (variables.count(ptr->getName().str()) && variables.count(instName)) {
      addEdge(instName, ptr->getName().str(), pi, 20);
    }
  }

  else if (isa<ConstantExpr>(ptr)) { //ptr doesn't have a name
    ConstantExpr *ce = cast<ConstantExpr>(ptr);  
    if (ce->getOpcode() == Instruction::GetElementPtr) {
      string opName = geGetElementPtr(ce, iSet, props, edge_type, currentLineNum);
      if (opName.length() == 0)
        return;
      if (variables.count(instName)&&variables.count(opName))
        addEdge(instName, opName, pi, 21);
    }

    else if (ce->getOpcode() == Instruction::BitCast || 
             ce->getOpcode() == Instruction::AddrSpaceCast) {
      Value *vRepl = ce->getOperand(0);
      if (vRepl == NULL) {
#ifdef DEBUG_ERROR
        blame_info<<"ptr bitcast get NULL in geAtomicLoadPart"<<endl;
#endif 
        return;
      }
            
      string vReplName;
      if (vRepl->hasName())
        vReplName = vRepl->getName().str();
      else {
        sprintf(tempBuf, "0x%x", vRepl);
        string tempStr(tempBuf);
        vReplName.insert(0, tempStr);
      }

      // TODO: Add debug integer param to addEdge so we know which addEdge it came from 
      if (variables.count(instName) && variables.count(vReplName)) {
        addEdge(instName, vReplName, pi, 22);
      }
      else {
#ifdef DEBUG_ERROR
        blame_info<<"Can't find "<<instName<<" or "<<vReplName<<" in variables in geAtomicLoadPart"<<endl;
#endif 
        return;
      }
    }  
  }

  else { //ptr doesn't have a name and not a constantExpr, most common
    string opName;  
    sprintf(tempBuf, "0x%x", /*(unsigned)*/ptr);
    string tempStr(tempBuf);    
    opName.insert(0, tempStr);
        
    if (variables.count(opName) && variables.count(instName)) {
      addEdge(instName, opName, pi, 23);
    }
  }
}  


void FunctionBFC::geAtomicStorePart(Value *ptr, Value *val, 
                            Instruction *pi, set<const char*, ltstr> &iSet, 
                            property_map<MyGraphType, vertex_props_t>::type props, 
                            property_map<MyGraphType, edge_iore_t>::type edge_type,  
                            int &currentLineNum)
{
  //first=value,second=location to store
  string firstName, secondName;
  char tempBuf[18];
  
  if (ptr->hasName()) {
    secondName = ptr->getName().str();
    addImplicitEdges(ptr, iSet, edge_type, "", false);
  }
  else if (isa<ConstantExpr>(ptr)) {
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"Second value in Store is ConstantExpr"<<endl;
#endif 
    // Just to make sure
    ConstantExpr *ce = cast<ConstantExpr>(ptr);  
    if (ce->getOpcode() == Instruction::BitCast || 
        ce->getOpcode() == Instruction::AddrSpaceCast) {
      // Overwriting 
#ifdef DEBUG_GRAPH_BUILD
      blame_info<<"Overwriting GEP Store second with "<<ce->getOperand(0)->getName().data()<<endl;
#endif 
      ptr = ce->getOperand(0);
      if (ptr->hasName()) 
        secondName= ptr->getName().str();
      else {
        sprintf(tempBuf, "0x%x", /*(unsigned)*/ptr);
        string tempStr(tempBuf);
        secondName.insert(0, tempStr);
      }
    }
    else if (ce->getOpcode() == Instruction::GetElementPtr)
      secondName= geGetElementPtr(ce, iSet, props, edge_type, currentLineNum);
  }
  else {
    sprintf(tempBuf, "0x%x", /*(unsigned)*/ptr);
    string tempStr(tempBuf);
    secondName.insert(0, tempStr);
  }

  // start dealing with val
  if (val->hasName()) {
    firstName = val->getName().str();
    //addImplicitEdges(val, iSet, edge_type, "", false);
  }
  else if (isa<ConstantExpr>(val)) {
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"First value in Store is ConstantExpr"<<endl;
#endif 
    ConstantExpr *ce = cast<ConstantExpr>(val);  
    if (ce->getOpcode() == Instruction::BitCast || 
        ce->getOpcode() == Instruction::AddrSpaceCast) {
      val = ce->getOperand(0);
      if (val->hasName()) 
        firstName= val->getName().str();
      else {
        sprintf(tempBuf, "0x%x", /*(unsigned)*/val);
        string tempStr(tempBuf);
        firstName.insert(0, tempStr);
      }
    }
    else if (ce->getOpcode() == Instruction::GetElementPtr) {
      firstName= geGetElementPtr(ce, iSet, props, edge_type, currentLineNum);
    }
  }  
  else if (isa<ConstantInt>(val)) {
    ConstantInt *cv = (ConstantInt *)val;
    int number = cv->getSExtValue();  
    char tempBuf[64];
    sprintf(tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, 0, pi->getOpcode());    
    char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
  
    strcpy(vN,tempBuf);
    vN[strlen(tempBuf)]='\0';
    const char *vName = vN;
  
    firstName.insert(0, vName);

    //addImplicitEdges(pi, iSet, edge_type, vName, true);
  } 
  else { // FORTRAN support
    sprintf(tempBuf, "0x%x", /*(unsigned)*/val);
    string tempStr(tempBuf);
    firstName.insert(0, tempStr);
  }
  
  if (variables.count(secondName) && variables.count(firstName))
    addEdge(secondName, firstName, pi, 24);
}


bool FunctionBFC::isPTXIntrinsic(Instruction *pi) {
    if (pi->getOpcode() == Instruction::Call) {
      Value *lastOp = pi->getOperand(pi->getNumOperands()-1);
      if (lastOp->hasName()) {
        string fn = lastOp->getName().str();
        if (fn.find("llvm.nvvm") == 0) {
          if (fn.find("llvm.nvvm.read") != string::npos)
            blame_info<<"WTH?? why "<<fn<<" here?"<<endl;
          
          if (!pi->hasNUsesOrMore(1)) 
            blame_info<<fn<<" has no uses"<<endl;
          
          return true;
        }
      }
    }

    return false;
}


void FunctionBFC::addEdge(string source, string dest, Instruction *pi, int place)
{
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  bool inserted, existed;
  graph_traits < MyGraphType >::edge_descriptor ed, oldEg;  
#ifdef DEBUG_GRAPH_BUILD_EDGES
  blame_info<<"Adding edge "<<place<<" between "<<source<<" and "<<dest<<" of type "<<pi->getOpcodeName()<<endl;
#endif
  
  if (variables.count(source) == 0 || variables.count(dest) == 0) {
#ifdef DEBUG_ERROR
    blame_info<<"Variables can't find value of "<<source<<" "<<variables.count(source);
    blame_info<<" or "<<dest<<" "<<variables.count(dest)<<endl;
#endif       
    return;
  }
  /* TO DEBUG 02/12/18: DON'T know why calling "edge" fail in some cases
  blame_info<<"I'm fine 1 "<<variables[source]->number<<"->"<<variables[dest]->number<<endl;
   
  tie(oldEg, existed) = edge(variables[source]->number, variables[dest]->number, G);
 
  blame_info<<"I'm fine 2"<<endl;
 
  if (existed) {
#ifdef DEBUG_ERROR
    blame_info<<"Old edge existed between "<<source<<" and "<<dest<<endl;
#endif       
    return;
  }*/
      
  tie(ed, inserted) = add_edge(variables[source]->number, variables[dest]->number, G);
 
  if (inserted) {
    if (isPTXIntrinsic(pi)) 
      edge_type[ed] = NVVM_PTX_INTRINSIC; //Defined in P.h
    else
      edge_type[ed] = pi->getOpcode();
  }
  else {
#ifdef DEBUG_ERROR
    blame_info<<"Insertion fail in genEdges for "<<source<<" to "<<dest<<endl;
    cerr<<"Insertion fail in genEdges for "<<source<<" to "<<dest<<endl;
#endif 
  }
}


void FunctionBFC::geStore(Instruction *pi, set<const char*, ltstr> &iSet,
      property_map<MyGraphType, vertex_props_t>::type props,
      property_map<MyGraphType, edge_iore_t>::type edge_type,
      int & currentLineNum)
{  
  //bool inserted;
  // graph_traits < MyGraphType >::edge_descriptor ed;
  User::op_iterator op_i = pi->op_begin();
  Value *first = op_i->get();
  op_i++;
  Value *second = op_i->get();
  //first=value,second=location to store
  string firstName, secondName;
  
  char tempBuf[18];
  
  if (second->hasName()) {
    secondName = second->getName();
    addImplicitEdges(second, iSet, edge_type, "", false);
  }
  else if (isa<ConstantExpr>(second)){
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"Second value in Store is ConstantExpr"<<endl;
#endif 
       
    ConstantExpr *ce = cast<ConstantExpr>(second);  
    if (ce->getOpcode() == Instruction::BitCast ||
        ce->getOpcode() == Instruction::AddrSpaceCast) {
      // Overwriting 
#ifdef DEBUG_GRAPH_BUILD
      blame_info<<"Overwriting Cast Store second with "<<ce->getOperand(0)->getName().data()<<endl;
#endif 
      second = ce->getOperand(0);
      if (second->hasName()) 
        secondName= second->getName().str();
      else {
        sprintf(tempBuf, "0x%x", /*(unsigned)*/second);//TC: old:first
        string tempStr(tempBuf);
        secondName.insert(0, tempStr);
      }
    }
    else if (ce->getOpcode() == Instruction::GetElementPtr)
      secondName= geGetElementPtr(ce, iSet, props, edge_type, currentLineNum);
  }  
  else if (!second->hasName() && !isa<Constant>(second)) {
    sprintf(tempBuf, "0x%x", /*(unsigned)*/second);
    string tempStr(tempBuf);
    secondName.insert(0, tempStr);
  }

  //08/02/17 We should never add implicit edges between first and iSet since that's READ, not WRITTEN
  if (first->hasName()){
    firstName = first->getName().str();
  }
  else if (isa<ConstantExpr>(first)) {
#ifdef DEBUG_GRAPH_BUILD
    blame_info<<"First value in Store is ConstantExpr"<<endl;
#endif 
    ConstantExpr *ce = cast<ConstantExpr>(first);  
    if (ce->getOpcode() == Instruction::BitCast ||
        ce->getOpcode() == Instruction::AddrSpaceCast) {
      // Overwriting 
#ifdef DEBUG_GRAPH_BUILD
      blame_info<<"Overwriting Cast Store first with "<<ce->getOperand(0)->getName().data()<<endl;
#endif 
      first = ce->getOperand(0);
      if (first->hasName()) 
        firstName= first->getName().str();
      else {
        sprintf(tempBuf, "0x%x", /*(unsigned)*/first);
        string tempStr(tempBuf);
        firstName.insert(0, tempStr);
      }
    }
    else if (ce->getOpcode() == Instruction::GetElementPtr) {
      firstName= geGetElementPtr(ce, iSet, props, edge_type, currentLineNum);
    }
    else {
      char tempBuf[18];
      sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
      string name(tempBuf); // Use the address of the instruction as its name ??
      name += ".CE";
          
      char tempBuf2[10];
      sprintf(tempBuf2, ".%d", currentLineNum);
      name.append(tempBuf2);
      firstName.insert(0, name);
    }
  }
  else if (isa<ConstantInt>(first)) {
    ConstantInt *cv = (ConstantInt *)first;
    int number = cv->getSExtValue();
    
    char tempBuf[64];
    sprintf(tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, 0, pi->getOpcode());    
    char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
  
    strcpy(vN,tempBuf);
    vN[strlen(tempBuf)]='\0';
    const char *vName = vN;
    
    firstName.insert(0, vName);
  } 
  else if (isa<ConstantFP>(first)) {
    char tempBuf[70];
    ConstantFP *cfp = (ConstantFP *)first;
    const APFloat apf = cfp->getValueAPF();
  
    if (APFloat::semanticsPrecision(apf.getSemantics()) == 24) {
      float floatNum = apf.convertToFloat();
      sprintf (tempBuf, "Constant+%g+%i+%i+%i", floatNum, currentLineNum, 0, pi->getOpcode());    
    }
    else if(APFloat::semanticsPrecision(apf.getSemantics()) == 53) {
      double floatNum = apf.convertToDouble();
      sprintf (tempBuf, "Constant+%g2.2+%i+%i+%i", floatNum, currentLineNum, 0, pi->getOpcode());    
    }
    else {
#ifdef DEBUG_ERROR
      blame_info<<"Not a float or a double FPVal"<<endl;
#endif 
      return;
    }
    
    char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
    strcpy(vN,tempBuf);
    vN[strlen(tempBuf)]='\0';
    const char * vName = vN;
    
    firstName.insert(0, vName);  
  }
  else if (!first->hasName() && !isa<Constant>(first)) { //for registers
    sprintf(tempBuf, "0x%x", /*(unsigned)*/first);
    string tempStr(tempBuf);
    firstName.insert(0, tempStr);
  }
  
  // store a, b: add edge  b->a
  addEdge(secondName, firstName, pi, 25);
}


void FunctionBFC::geBitCast(Instruction *pi, set<const char*, ltstr> &iSet,
              property_map<MyGraphType, vertex_props_t>::type props,
              property_map<MyGraphType, edge_iore_t>::type edge_type,
              int &currentLineNum)
{
  bool inserted;
  graph_traits < MyGraphType >::edge_descriptor ed;
  
  if (pi->getNumUses() == 1) {
    Value::use_iterator use_i = pi->use_begin();
    if (Instruction *usesBit = dyn_cast<Instruction>((*use_i).getUser())) {
      if (isa<CallInst>(usesBit)) {
        CallInst *ci = cast<CallInst>(usesBit);
        if (ci != NULL) {
          Function *calledFunc = ci->getCalledFunction();
          if (calledFunc != NULL)
            if (calledFunc->hasName())
              if (calledFunc->getName().str().find("llvm.dbg") != string::npos)
                return;
        }
      }            
    }
  } 

  string instName, opName;
  instName = getNameForVal(pi);
  opName = getNameForVal(pi->op_begin()->get());

  addEdge(instName, opName, pi, 33);  
}


void FunctionBFC::geBlank(Instruction *pi)
{
#ifdef DEBUG_GRAPH_BUILD
  blame_info<<"Not generating any edges for opcode "<<pi->getOpcodeName()<<endl;
#endif
}


void FunctionBFC::geInvoke()
{
#ifdef DEBUG_ERROR      
  blame_info<<"Ignoring invoke (geInvoke)"<<endl;
#endif 
}


void FunctionBFC::genEdges(Instruction *pi, set<const char*, ltstr> &iSet,
              property_map<MyGraphType, vertex_props_t>::type props,
              property_map<MyGraphType, edge_iore_t>::type edge_type,
              int &currentLineNum, set<NodeProps *> &seenCall)
{  
  if (pi == NULL) {
#ifdef DEBUG_ERROR
    cerr<<"GE -- pi is NULL"<<endl;
    blame_info<<"GE -- pi is NULL"<<endl;
#endif
    return;
  }
  
#ifdef DEBUG_GRAPH_BUILD
  if (pi->hasName())
    blame_info<<"GE Instruction "<<pi->getName().str()<<" "<<pi->getOpcodeName()<<endl;
  else
    blame_info<<"GE No name "<<pi->getOpcodeName()<<endl;
#endif 
  
  if (MDNode *N = pi->getMetadata("dbg")) {
      DILocation *Loc = dyn_cast<DILocation>(N);
      unsigned Line = Loc->getLine();
      currentLineNum = Line;
      string File = Loc->getFilename().str();
      string Dir = Loc->getDirectory().str();
      //Now this part is unique for cuda, it'll call cuda lib funcs with debug info in their source
      //we need to reset those line numbers since they can mixed up with user source lines  11/07/17
      //Usually these instructions won't appear in the begining of a func, so the moduleName should've been set correctly
      if (moduleSet && File != moduleName)
        currentLineNum = 0;
    }
/*  Not used right now, corresponding to firstGEPCheck in LLVMParser.cpp 10/16/17
  if (pi->getOpcode() == Instruction::GetElementPtr) { 
    Value *v = pi->getOperand(0);                   
      
      const llvm::Type *pointT = v->getType();
    unsigned typeVal = pointT->getTypeID();
    
    while (typeVal == Type::PointerTyID) {      
    pointT = cast<PointerType>(pointT)->getElementType();    
    typeVal = pointT->getTypeID();
    }
  
    if (typeVal == Type::StructTyID) {
      const llvm::StructType *type = cast<StructType>(pointT);
        if (type->hasName()) {
      string structNameFull = type->getStructName().str(); //parent module M
#ifdef DEBUG_GRAPH_BUILD
         blame_info<<"GenEdges -- structNameFull -- "<<structNameFull<<endl;
#endif       
        if (structNameFull.find("struct.descriptor_dimension") != string::npos) {
#ifdef DEBUG_P
            cout<<structNameFull<<" find struct.descriptor_dimension"<<endl;
#endif
            return;
          }
      
        if (structNameFull.find("struct.array1") != string::npos) {
#ifdef DEBUG_P
            cout<<structNameFull<<" find struct.array1"<<endl;
#endif
        if (pi->getNumOperands() >= 3) {
          Value *vOp = pi->getOperand(2);
          if (vOp->getValueID() == Value::ConstantIntVal) {
            ConstantInt *cv = (ConstantInt *)vOp;
            int number = cv->getSExtValue();
            if (number != 0)//TOCHECK: why is this cond
              return;
          }
        }
      }
          }
    }
  }
*/  
  // 1
  // TERMINATOR OPS
  if (pi->getOpcode() == Instruction::Ret)
    geBlank(pi);
  else if (pi->getOpcode() == Instruction::Br)
    geBlank(pi);
  else if (pi->getOpcode() == Instruction::Switch)
    geBlank(pi);
    else if (pi->getOpcode() == Instruction::IndirectBr)
      geBlank(pi);
  else if (pi->getOpcode() == Instruction::Invoke)
    geInvoke();
  else if (pi->getOpcode() == Instruction::Resume)
    geBlank(pi);
  else if (pi->getOpcode() == Instruction::Unreachable)
    geBlank(pi);
  /*else if (pi->getOpcode() == Instruction::CleanupReturn)
    geBlank(pi);
  else if (pi->getOpcode() == Instruction::CatchReturn)
    geBlank(pi);
  else if (pi->getOpcode() == Instruction::CatchSwitch)
    geBlank(pi);*/
  // END TERMINATOR OPS
  
  // BINARY OPS 
  else if (pi->getOpcode() == Instruction::Add)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
    else if (pi->getOpcode() == Instruction::FAdd)
      geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::Sub)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::FSub)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::Mul)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::FMul)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::UDiv)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::SDiv)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::FDiv)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::URem)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::SRem)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::FRem)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  // END BINARY OPS
  
  // LOGICAL OPERATORS 
  else if (pi->getOpcode() == Instruction::Shl)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::LShr)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::AShr)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::And)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::Or)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::Xor)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  // END LOGICAL OPERATORS
  
  // MEMORY OPERATORS
  else if (pi->getOpcode() == Instruction::Alloca)
    geBlank(pi);
  else if (pi->getOpcode() == Instruction::Load)
    geLoad(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::Store)
    geStore(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::GetElementPtr)  
    geGetElementPtr(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::Fence) //TC
    geBlank(pi);
  else if (pi->getOpcode() == Instruction::AtomicCmpXchg) //TC
    geMemAtomic(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::AtomicRMW) //TC
    geMemAtomic(pi, iSet, props, edge_type, currentLineNum);
  // END MEMORY OPERATORS
  
  // CAST OPERATORS
  else if (pi->getOpcode() == Instruction::Trunc)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::ZExt)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::SExt)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::FPToUI)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::FPToSI)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::UIToFP)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::SIToFP)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::FPTrunc)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::FPExt)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::PtrToInt)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::IntToPtr)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::BitCast)
    geBitCast(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::AddrSpaceCast) //cuda 04/24/18
    geBitCast(pi, iSet, props, edge_type, currentLineNum); //exactly same process as bitcast
  // END CAST OPERATORS
  
  // OTHER OPERATORS
  else if (pi->getOpcode() == Instruction::ICmp)
      geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::FCmp)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::PHI)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::Call)
    geCall(pi, iSet, props, edge_type, currentLineNum, seenCall);
  else if (pi->getOpcode() == Instruction::Select)
    geDefault(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::ExtractValue)
    geExtractValue(pi, iSet, props, edge_type, currentLineNum);
  else if (pi->getOpcode() == Instruction::InsertValue)
    geInsertValue(pi, iSet, props, edge_type, currentLineNum);
  else {
#ifdef DEBUG_ERROR
    blame_info<<"LLVM__(examineInstruction)(Not processed inst(ge)) -- "<<pi->getOpcodeName()<<endl;
#endif 
  }      
}


/*void FunctionBFC::transferImplicitEdges(int alloc_num, int bool_num)
{
  graph_traits < MyGraphType >::edge_descriptor ed;
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  bool inserted;
  
  // iterate through the out edges and delete edges to target, move other edges
  boost::graph_traits<MyGraphType>::out_edge_iterator oe_beg, oe_end;
  oe_beg = boost::out_edges(bool_num, G).first;    // edge iterator begin
  oe_end = boost::out_edges(bool_num, G).second;       // edge iterator end
  
  //vector<int> destNodes;
  
  // bool_num = delete Node
  // alloc_num = recipient Node
  NodeProps * rN =  get(get(vertex_props, G),alloc_num);
  NodeProps * dN =  get(get(vertex_props, G),bool_num);  
  
#ifdef DEBUG_GRAPH_IMPLICIT
  blame_info<<"Collapse_(transferImplicitEdges) - transfer lineNumber "<<dN->line_num<<" from "<<dN->name<<" to "<<rN->name<<endl;
#endif
  rN->lineNumbers.insert(dN->line_num);
  
            #ifdef DEBUG_LINE_NUMS
  blame_info<<"Inserting line number(3) "<<dN->line_num<<" to "<<rN->name<<endl;
  #endif
  
  if (dN->nStatus[CALL_PARAM])
  {
    dN->nStatus[CALL_PARAM] = false;
    rN->nStatus[CALL_PARAM] = true;
  }
  if (dN->nStatus[CALL_RETURN])
  {
    dN->nStatus[CALL_RETURN] = false;
    rN->nStatus[CALL_RETURN] = true;
  }
  if (dN->nStatus[CALL_NODE])
  {
    dN->nStatus[CALL_NODE] = false;
    rN->nStatus[CALL_NODE] = true;
  }
  
  set<FuncCall *>::iterator fc_i = dN->funcCalls.begin();
  
  for (; fc_i != dN->funcCalls.end(); fc_i++)
  {
    //FuncCall * fc = *fc_i;
    //cout<<"IMPLICIT Transferring call to node "<<alloc_num<<" from "<<fc->funcName<<" param num "<<fc->paramNumber<<endl;
    rN->addFuncCall(*fc_i);
  }
  
  
  
  for(; oe_beg != oe_end; ++oe_beg) 
  {
    
    int movedOpCode = get(get(edge_iore, G),*oe_beg);
    
    NodeProps * outTargetV = get(get(vertex_props,G), target(*oe_beg,G));
    
    //destNodes.push_back(outTargetV->number);
    
    // Don't want a self loop edge
    if (outTargetV->number != alloc_num )
    {
      tie(ed, inserted) = add_edge(alloc_num, outTargetV->number, G);
      if (inserted)
        edge_type[ed] = movedOpCode;  
    }
    
  }
  
  
  // iterate through the in edges and create new edge to destination, delete old edge
  boost::graph_traits<MyGraphType>::in_edge_iterator ie_beg, ie_end;
  ie_beg = boost::in_edges(bool_num, G).first;    // edge iterator begin
  ie_end = boost::in_edges(bool_num, G).second;       // edge iterator end
  
  for(; ie_beg != ie_end; ++ie_beg) 
  {
    int movedOpCode = get(get(edge_iore, G),*ie_beg);
    
    NodeProps * inTargetV = get(get(vertex_props,G), source(*ie_beg,G));
    
    // Don't want a self loop edge
    if (inTargetV->number != alloc_num )//&& movedOpCode > 0)
    {
      tie(ed, inserted) = add_edge(inTargetV->number, alloc_num, G);
      if (inserted)
        edge_type[ed] = movedOpCode;  
      
    }
    
  }
  
}*/


/*void FunctionBFC::deleteOldImplicit(int v_index)
{
  graph_traits < MyGraphType >::edge_descriptor ed;
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  //bool inserted;
  
  boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
  boost::graph_traits<MyGraphType>::in_edge_iterator i_beg, i_end;
  
  
  e_beg = boost::out_edges(v_index, G).first;    // edge iterator begin
  e_end = boost::out_edges(v_index, G).second;    // edge iterator end
  
  i_beg = boost::in_edges(v_index, G).first;    // edge iterator begin
  i_end = boost::in_edges(v_index, G).second;    // edge iterator end
  
  vector<int> destNodes;
  vector<int> sourceNodes;
  vector<int> allocNodes;
  
  // iterate through the edges trying to find a non-implicit instruction
  for(; e_beg != e_end; ++e_beg) 
  {
    int opCode = get(get(edge_iore, G),*e_beg);
    
    NodeProps * targetV = get(get(vertex_props,G), target(*e_beg,G));
    
    destNodes.push_back(targetV->number);
    
    if (opCode != 0)
    {
      if (targetV->llvm_inst != NULL)
      {
        if (isa<Instruction>(targetV->llvm_inst))
        {
          Instruction * pi = cast<Instruction>(targetV->llvm_inst);  
          
          
          //const llvm::Type * t = pi->getType();
          //unsigned typeVal = t->getTypeID();
          
          if (pi->getOpcode() != Instruction::Alloca)
            allocNodes.push_back(targetV->number);
        }
      }
    }
  }  
  
  for(; i_beg != i_end; ++i_beg) 
  {
    //int movedOpCode = get(get(edge_iore, G),*i_beg);
    NodeProps * inTargetV = get(get(vertex_props,G), source(*i_beg,G));
    sourceNodes.push_back(inTargetV->number);
  }
  
  
  vector<int>::iterator v_i;
  
  for (v_i = destNodes.begin(); v_i != destNodes.end(); ++v_i)
    remove_edge(v_index, *v_i, G);
  
  for (v_i = sourceNodes.begin(); v_i != sourceNodes.end(); ++v_i)
    remove_edge(*v_i, v_index, G);
  
  for (v_i = allocNodes.begin(); v_i != allocNodes.end(); ++v_i)
    deleteOldImplicit(*v_i);
}*/




void FunctionBFC::collapseIO()
{
  //bool inserted;
  //int collapsedEdges = 0;
  graph_traits < MyGraphType >::edge_descriptor ed;
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
  graph_traits<MyGraphType>::vertex_iterator i, v_end; 
  
  for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    NodeProps *v = get(get(vertex_props, G),*i);
    int v_index = get(get(vertex_index, G),*i);
    
    if (!v) {
#ifdef DEBUG_ERROR      
    cerr<<"Null V in collapseIO\n";
#endif 
    continue;
    }
    
      boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
    e_beg = boost::out_edges(v_index, G).first;  // edge iterator begin
    e_end = boost::out_edges(v_index, G).second;  // edge iterator end
    
    vector< pair<int,int> > deleteEdges;
    // iterate through the edges to find matching opcode
    for (; e_beg != e_end; ++e_beg) {
    int opCode = get(get(edge_iore, G),*e_beg);
      
    if (opCode == Instruction::Call || opCode == Instruction::Invoke) {                    
      NodeProps *sourceV = get(get(vertex_props,G), source(*e_beg,G));
      NodeProps *targetV = get(get(vertex_props,G), target(*e_beg,G));
      //c++ "<<" basic_ostream
          //TOCHECK: Hui 12/14/15: should add "writeln"? why check targetV
          //since edge is like: writeln -> a ??
      size_t loc = targetV != NULL ? targetV->name.find("basic_ostream") : string::npos;
      size_t loc2 = targetV != NULL ? targetV->name.find("SolsEi") : string::npos;
        
      if (loc != string::npos) {
       string tmpStr("COUT");
      ExitProgram *ep = findOrCreateExitProgram(tmpStr);
      if (ep != NULL) {
          //ep->addVertex(v);
      }
      else {
#ifdef DEBUG_ERROR      
        cerr<<"WTF!  Why is this shiznit NULL!(2) for "<<v->name<<endl;  
#endif 
        continue;
      }
          
      pair<int, int> tmpPair(sourceV->number, targetV->number);
#ifdef DEBUG_GRAPH_COLLAPSE
      blame_info<<"Remove edge(hui) from "<<sourceV->name<<" to "<<targetV->name<<" [6]"<<endl;
#endif
      deleteEdges.push_back(tmpPair);
      }

      else if (loc2 != string::npos) {  
      string tmpStr("COUT-VAR-");
      if (v->llvm_inst != NULL) {
          if (isa<Instruction>(v->llvm_inst)) {
        Instruction *l_i = cast<Instruction>(v->llvm_inst);  
        if (l_i->getOpcode() == Instruction::Alloca) {
            tmpStr += sourceV->name;
          ExitProgram *ep = findOrCreateExitProgram(tmpStr);
                
          if (ep != NULL){
          //ep->addVertex(v);
          }
          else {
#ifdef DEBUG_ERROR      
          cerr<<"WTF!  Why is this shiznit NULL!(3) for "<<v->name<<endl;  
#endif 
          continue;
          }
                }
        }
      }
          
      pair<int, int> tmpPair(sourceV->number, targetV->number);
#ifdef DEBUG_GRAPH_COLLAPSE
        blame_info<<"Remove edge(hui) from "<<sourceV->name<<" to "<<targetV->name<<" [7]"<<endl;
#endif
            deleteEdges.push_back(tmpPair);
      }
    }
    }
    
    vector<pair<int,int> >::iterator vecPair_i;
    
    for (vecPair_i = deleteEdges.begin();  vecPair_i != deleteEdges.end(); vecPair_i++) {
      pair<int,int> del = *vecPair_i;
      //cout<<"Remove edge for "<<del.first<<" to "<<del.second<<endl;
      remove_edge(del.first, del.second, G);
    }
  } //all vertices
}




// recursively descend edges until a local variable (or parameter) is found 
/*void FunctionBFC::findLocalVariable(int v_index, int top_index)
{
  graph_traits < MyGraphType >::edge_descriptor ed;
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
  boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
  
  e_beg = boost::out_edges(v_index, G).first;    // edge iterator begin
  e_end = boost::out_edges(v_index, G).second;    // edge iterator end
  
  // iterate through the edges trying to find a non-implicit instruction
  for(; e_beg != e_end; ++e_beg) 
  {
    int opCode = get(get(edge_iore, G),*e_beg);
    
    if (opCode != 0)
    {
      //cout<<"Op type "<<opCode<<" from "<<v->name<<endl;
      NodeProps * targetV = get(get(vertex_props,G), target(*e_beg,G));
      
      if (targetV->llvm_inst != NULL)
      {
        if (isa<Instruction>(targetV->llvm_inst))
        {
          Instruction * pi = cast<Instruction>(targetV->llvm_inst);  
          
          //const llvm::Type * t = pi->getType();
          //unsigned typeVal = t->getTypeID();
          
          if (pi->getOpcode() == Instruction::Alloca)
          {
            //cout<<"Found one - "<<targetV->name<<endl;
            transferImplicitEdges(targetV->number, top_index);
          }
          else
          {
            findLocalVariable(targetV->number, top_index);
          }
        }
        else
        {
          transferImplicitEdges(targetV->number, top_index);
        }
      }
      else
      {
        transferImplicitEdges(targetV->number, top_index);
      }
    }
  }  
}*/



// TODO:  Right now we are looking at parents
// x --> y --> z
// |     |     |
// v     v     v
// i  <-- <----
//   
// This works if you have a chain, but if 'y'
// doesn't have the link to 'i' than it doesn't work
// This function is more lightweight analysis, but
// should work for most cases
void FunctionBFC::collapseRedundantImplicit()
{
  // Iterator that goes through all the vertices, we
  // need to find vertices that have incoming implicit edges
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
  for(tie(i,v_end) = vertices(G); i != v_end; ++i) 
  {
    NodeProps * v = get(get(vertex_props, G),*i);
    int v_index = get(get(vertex_index, G),*i);
    
    if (!v)
    {
#ifdef DEBUG_ERROR      
      cerr<<"Null V in collapseRedundantImplicit\n";
#endif 
      continue;
    }
    
    set<int> edgesToDelete;
    
    //graph_traits < MyGraphType >::edge_descriptor ed;
    property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
    //bool inserted;
    boost::graph_traits<MyGraphType>::in_edge_iterator i_beg, i_end;
    
    i_beg = boost::in_edges(v_index, G).first;    // edge iterator begin
    i_end = boost::in_edges(v_index, G).second;    // edge iterator end
    
    
    // This checks all incoming edges and looks for any incoming implicit edges
    for(; i_beg != i_end; ++i_beg) 
    {
      int opCode = get(get(edge_iore, G),*i_beg);
      
      //  opcode 0 means an IMPLICIT edge
      // TODO: Make 0 a constant for IMPLICIT
      if (opCode == 0)
      {
        NodeProps * inTargetV = get(get(vertex_props,G), source(*i_beg,G));
        
        // We want to check if there is a parent that also connects to an implicit
        // if so, we want to delete the edge from the parent to implicit target
        
        //graph_traits < MyGraphType >::edge_descriptor ed;
        //property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
        //bool inserted;
        boost::graph_traits<MyGraphType>::in_edge_iterator i_beg2, i_end2;
        
        i_beg2 = boost::in_edges(inTargetV->number, G).first;    // edge iterator begin
        i_end2 = boost::in_edges(inTargetV->number, G).second;    // edge iterator end
        
        // We look over all the parent edges to the one that has the implicit look
        for(; i_beg2 != i_end2; ++i_beg2) 
        {
          int opCode3 = get(get(edge_iore, G),*i_beg2);
          
          if (opCode3 == 0)
          {
            
            NodeProps * parentV = get(get(vertex_props,G), source(*i_beg2,G));
            
            // Check for loops, if there is a loop we're aren't touching it
            boost::graph_traits<MyGraphType>::in_edge_iterator i_beg3, i_end3;
            
            i_beg3 = boost::in_edges(parentV->number, G).first;    // edge iterator begin
            i_end3 = boost::in_edges(parentV->number, G).second;    // edge iterator end
            
            for (; i_beg3 != i_end3; ++i_beg3)
            {
              NodeProps * maybeLoopV = get(get(vertex_props,G), source(*i_beg3,G));
              if (maybeLoopV->number == inTargetV->number)
              {
#ifdef DEBUG_GRAPH_IMPLICIT
                blame_info<<"Implicit__(collapseRedundantImplicit) - Loop detected between ";
                blame_info<<parentV->name<<" and "<<inTargetV->name<<endl;
#endif
                continue;
              }
            }
            
            // Now we check to see if any of the parents outgoing edges goes to the original
            //  implicit source, if so, we can target that edge for deletion
            boost::graph_traits<MyGraphType>::out_edge_iterator o_beg, o_end;
            
            o_beg = boost::out_edges(parentV->number, G).first;    // edge iterator begin
            o_end = boost::out_edges(parentV->number, G).second;    // edge iterator end
            
            for (; o_beg != o_end; ++o_beg)
            {
              int opCode2 = get(get(edge_iore, G), *o_beg);
              if (opCode2 == 0)
              {
                NodeProps * implicitV = get(get(vertex_props,G), target(*o_beg,G));
                if (v->number == implicitV->number)
                {
#ifdef DEBUG_GRAPH_IMPLICIT
                  blame_info<<"Implicit__(collapseRedundantImplicit) - NODE(TFD) "<<parentV->name;
                  blame_info<<" has implicit edge to "<<inTargetV->name<<" and "<<v->name;
                  blame_info<<" Delete imp edge from "<<inTargetV->name<<" to "<<v->name<<endl;
#endif
                  edgesToDelete.insert(inTargetV->number);
                }
              }
            }
            
          }
        }
      }
    }
    
    set<int>::iterator deleteIterator;
    
    for (deleteIterator = edgesToDelete.begin();  deleteIterator != edgesToDelete.end(); deleteIterator++)
    {
      remove_edge(*deleteIterator, v_index , G);
      //cout<<"CRI - Removeing edge from "<<*deleteIterator<<" to "<<v_index<<endl;
    }
    
    //edgesToDelete.erase();
  }
}



// This function takes a graph of the form
// toBool --> tmp --> ... --> <local variable>
// and deletes everything up to the local variable, moving
// all of the edges accordingly
/*
void FunctionBFC::collapseImplicit()
{
  
  //graph_traits < MyGraphType >::edge_descriptor ed;
  //property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
  for(tie(i,v_end) = vertices(G); i != v_end; ++i) 
  {
    
    NodeProps * v = get(get(vertex_props, G),*i);
    int v_index = get(get(vertex_index, G),*i);
    
    if (!v)
    {
#ifdef DEBUG_ERROR      
      cerr<<"Null V in collapseImplicit\n";
#endif 
      continue;
    }
    
    if ( strstr(v->name.c_str(), "toBool") != NULL )
    {
      //cout<<"For "<<v->name<<endl;
      findLocalVariable(v_index, v_index);
      //cout<<endl;
      deleteOldImplicit(v_index);
    }
  }
}*/

// For now we don't want to transfer over line nums for loads we are collapsing if they
// feed into calls.  The reason for this is that the transfer function will take care
// of this eventually.  We don't want line numbers feeding in when the parameter (or 
//  variable) is passed in as read only and can't contribute to the blame, so we
//  don't want the line number appended on
bool FunctionBFC::shouldTransferLineNums(NodeProps * v)
{
  bool shouldTransfer = true;

  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
  // iterate through the out edges and delete edges to target, move other edges
  boost::graph_traits<MyGraphType>::out_edge_iterator oe_beg, oe_end;
  oe_beg = boost::out_edges(v->number, G).first;    // edge iterator begin
  oe_end = boost::out_edges(v->number, G).second;       // edge iterator end
  
  for(; oe_beg != oe_end; ++oe_beg) {
    int opCode = get(get(edge_iore, G),*oe_beg);
    
    if (opCode == Instruction::Call) {
      shouldTransfer = false;
    }
    //NodeProps * outTargetV = get(get(vertex_props,G), target(*oe_beg,G));
  }
  return shouldTransfer;
}

/*
    Added by Hui 04/06/16
    Basically, for redundant field, we shall collapse it when it's a write 
    but keep it when it's a read:
    %6<--STORE--A-<--LOAD--%7==STORE==>newVal :     write (currently, we only consider one STORE-LOAD redundancy)
    %6<--STORE--A-<--LOAD--%7<--GEP--%8 :           read
    %6<--LOAD--%8:                                  read
    %6<--GEP--%8:                                   read

    //The conditions in this func can be extended in the future 
*/
bool FunctionBFC::shouldKeepEdge(int movedOpCode, NodeProps *inTargetV)
{
    if (movedOpCode == Instruction::Load || movedOpCode == GEP_BASE_OP) 
      return true;
    else if (movedOpCode==Instruction::Store) {
      // iterate through the in_edges 
      boost::graph_traits<MyGraphType>::in_edge_iterator ie_beg, ie_end;
      ie_beg = boost::in_edges(inTargetV->number, G).first;    // edge iterator begin
      ie_end = boost::in_edges(inTargetV->number, G).second;       // edge iterator end
        
      for (; ie_beg != ie_end; ++ie_beg) {
        int opCode = get(get(edge_iore, G),*ie_beg);
            
        if (opCode == Instruction::Load) {
          NodeProps *sV = get(get(vertex_props,G),source(*ie_beg,G));
          //we check the out_edge of sV 
          boost::graph_traits<MyGraphType>::out_edge_iterator oe_beg, oe_end;
          oe_beg = boost::out_edges(sV->number, G).first;    // edge iterator begin
          oe_end = boost::out_edges(sV->number, G).second;       // edge iterator end
          for (; oe_beg != oe_end; ++oe_beg) {
            int opCode2 = get(get(edge_iore, G), *oe_beg);
            //conservative way: as long as there's a write to the redundant field, it should collapse
            if (opCode2 == Instruction::Store)
              return false; 
          }
        }
      }
    }
    return true;
}

int FunctionBFC::transferEdgesAndDeleteNode(NodeProps *dN, NodeProps *rN, bool transferLineNumbers, bool fromGEP)
{
  //cerr<<"Made it TO here\n";
  //safe check, dN and rN can NOT be the same node, otherwise it'll go into an infinite loop
  if (dN == rN) {
    blame_info<<"Error, dN == rN  in transferEdgesAndDeleteNode!"<<endl;
    return -1;
  }

  int deleteNode = dN->number;
  int recipientNode = rN->number;
  
  if (shouldTransferLineNums(dN) && transferLineNumbers) {
#ifdef DEBUG_GRAPH_COLLAPSE
    blame_info<<"Collapse_(transferEdgesAndDeleteNode) - tranfer lineNumber "
              <<dN->line_num<<" from "<<dN->name<<" to "<<rN->name<<endl;
    blame_info<<"Inserting line number(4) "<<dN->line_num<<" to "<<rN->name<<endl;
#endif    
    rN->lineNumbers.insert(dN->line_num);
  }
#ifdef DEBUG_GRAPH_COLLAPSE
  blame_info<<"Node Props for delete node "<<dN->name<<": ";
  for (int a = 0; a < NODE_PROPS_SIZE; a++)
    blame_info<<dN->nStatus[a]<<" ";
  blame_info<<endl;  
  
  blame_info<<"Node Props for rec node "<<rN->name<<": ";
  for (int a = 0; a < NODE_PROPS_SIZE; a++)
    blame_info<<rN->nStatus[a]<<" ";
  blame_info<<endl;  
#endif 
  
  for (int a = 0; a < NODE_PROPS_SIZE; a++) {
    if (dN->nStatus[a])
      rN->nStatus[a] = dN->nStatus[a];
    
    dN->nStatus[a] = false;
  }
  
  // The following is VERY VERY important!
  set<FuncCall *>::iterator fc_i = dN->funcCalls.begin();
  
  for (; fc_i != dN->funcCalls.end(); fc_i++) {
    rN->addFuncCall(*fc_i);
  }
  
  if (dN->pointsTo != NULL) {
    if (rN->pointsTo == NULL)
      rN->pointsTo = dN->pointsTo;
    else { //TODO: about GEP pointers when they collapsed? 10/20/17
#ifdef DEBUG_ERROR
      blame_info<<"More than one pointsTo involved, fromGEP="<<fromGEP<<endl;
#endif  
    }
  }
  
#ifdef ADD_MULTI_LOCALE
  if (dN->isPid || dN->isObj) {
    if (dN->isPid && !rN->isPid && !rN->isObj) {
      rN->isPid = true;
      rN->myObj = dN->myObj;
    }
    else if (dN->isObj && !rN->isPid && !rN->isObj) {
      rN->isObj = true;
      rN->myPid = dN->myPid;
    }
    else
      blame_info<<"Very weird: dN->isPid="<<dN->isPid<<" dN->isObj="<<dN->isObj
          <<"; rN->isPid="<<rN->isPid<<" rN->isObj="<<rN->isObj<<endl;
  }
#endif
  
  dN->deleted = true;
  
  bool inserted;
  graph_traits < MyGraphType >::edge_descriptor ed;
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
  // iterate through the out edges and delete edges to target, move other edges
  boost::graph_traits<MyGraphType>::out_edge_iterator oe_beg, oe_end;
  oe_beg = boost::out_edges(deleteNode, G).first;    // edge iterator begin
  oe_end = boost::out_edges(deleteNode, G).second;       // edge iterator end

  /* Figure out all the destination nodes the would-be deleted node connects to
   
   |----> D1
   R-->S --|----> T
   |----> D2
   */
  
  vector<int> destNodes;  // D1, T, D2
  
  for (; oe_beg != oe_end; ++oe_beg) {
    int movedOpCode = get(get(edge_iore, G),*oe_beg);
    NodeProps *outTargetV = get(get(vertex_props,G), target(*oe_beg,G));
    if (!outTargetV) continue;
    /* Added by Hui 04/06/16: tricky ways to avoid include more lines than needed:
       if the out_edge is GEP_BASE_OP, and the dN/rN are redundant fields, then we
       choose NOT to transfer this edge since rN already has its base, and if it's another
       READ of the field,instead of WRITE, then new base will likely introduce a wrong line:
       %6 = GEP s, 0, 1 ----line 1
       %7 = GEP s, 0, 1 ----line 2
       %6 and %7 should be regarded as two different nodes(2 line#), 
       while if %6 and %7 are both WRITE, they should be seen as the same node. So we'll keep
       the edge %7->s, so %6 won't have wrong line# 2
    ********this change was for Chapel, not appropriate for cuda/c****************

    if (fromGEP && movedOpCode == GEP_BASE_OP) {
#ifdef DEBUG_GRAPH_COLLAPSE
      blame_info<<"We met a GEP_BASE_OP out_edge: "<<dN->name<<"-->"<<outTargetV->name<<" so we'll keep this edge and dN"<<endl;
#endif 
      continue;
    }*/
    
    if (outTargetV) {
      destNodes.push_back(outTargetV->number);  
      // Don't want a self loop edge
      if (outTargetV->number != recipientNode ) {
        // May be some implicit edges already there and we want explicit to trump implicit ... for now
        graph_traits < MyGraphType >::edge_descriptor oldEg;
        bool existed;
        tie(oldEg, existed) = edge(recipientNode, outTargetV->number, G);
        if (existed) {
          if (edge_type[oldEg] != IMPLICIT_OP)
            continue; //we keep the original edge between recipientNode and outTarget
          else
            remove_edge(recipientNode, outTargetV->number, G);
        }  
        
        tie(ed, inserted) = add_edge(recipientNode, outTargetV->number, G);
      
        if (inserted)
          edge_type[ed] = movedOpCode;  
        else {
#ifdef DEBUG_ERROR      
          blame_info<<"Insertion fail for "<<rN->name<<" to "<<outTargetV->name<<endl;
#endif 
        } 
      }
#ifdef DEBUG_GRAPH_COLLAPSE
      blame_info<<"Remove edge(hui) from "<<dN->name<<" to "<<outTargetV->name<<" [1]"<<endl;
#endif 
    }
  }
  
  vector<int>::iterator v_i, v_e = destNodes.end();
  
  for (v_i = destNodes.begin(); v_i != v_e; ++v_i) {
    remove_edge(deleteNode, *v_i, G);
  }  
  
  /* Figure out all the root nodes the would-be deleted node has incoming edges from
   |----> D1
   R-->S   T--| 
   |----> D2
   */
  vector<int> sourceNodes;  // R
  
  // iterate through the in edges and create new edge to destination, delete old edge
  boost::graph_traits<MyGraphType>::in_edge_iterator ie_beg, ie_end;
  ie_beg = boost::in_edges(deleteNode, G).first;    // edge iterator begin
  ie_end = boost::in_edges(deleteNode, G).second;     // edge iterator end
  
  for (; ie_beg != ie_end; ++ie_beg) {
    int movedOpCode = get(get(edge_iore, G),*ie_beg);
    NodeProps *inTargetV = get(get(vertex_props,G), source(*ie_beg,G));
    if (!inTargetV) continue;  
    /* Added by Hui 04/06/16: tricky ways to avoid include more lines than needed:
       if the in_edge is LOAD, and the dN/rN are redundant fields, then LOAD means 
       this field is to be READ, we don't want the blaming nodes to include line# of
       rn(usually it's earlier). So we simply NOT remove and transfer the edge: sN->dN;
       if the in_edge is STORE(usally it'll fowllowed by in_edge load, and out_edge store)
       it means the field is to be WRITTEN, then we do want the edges transffered so that all
       blame can go to the same field(rN)
    *******this change was for Chapel, not appropriate for cuda/c****************
    if (fromGEP && shouldKeepEdge(movedOpCode, inTargetV)) {
#ifdef DEBUG_GRAPH_COLLAPSE
      blame_info<<"We met a READ in_edge: "<<inTargetV->name<<"-->"<<dN->name<<" so we'll keep this edge and dN"<<endl;
#endif 
      continue;
    }*/

    sourceNodes.push_back(inTargetV->number);
    // Don't want a self loop edge
    if (inTargetV->number != recipientNode) {//&& movedOpCode > 0)
      // May be some implicit edges already there and we want explicit to trump implicit ... for now
      // If there already 
      //edge(u,v,g) returns pair<e_d, bool>, bool=>whether edge exists
      graph_traits < MyGraphType >::edge_descriptor oldEg;
      bool existed;
      tie(oldEg, existed) = edge(inTargetV->number, recipientNode, G);
      if (existed) {
        if (edge_type[oldEg] != IMPLICIT_OP)
          continue; //we keep the original edge between recipientNode and outTarget
        else
          remove_edge(inTargetV->number, recipientNode, G);
      }

      tie(ed, inserted) = add_edge(inTargetV->number, recipientNode, G);
      if (inserted)
        edge_type[ed] = movedOpCode;  
      else {
#ifdef DEBUG_ERROR      
        blame_info<<"Insertion fail for "<<inTargetV->name<<" to "<<rN->name<<endl;
#endif
      }
    }
#ifdef DEBUG_GRAPH_COLLAPSE
    blame_info<<"Remove edge(hui) from "<<inTargetV->name<<" to "<<dN->name<<" [2]"<<endl;
#endif 
  }
  
  v_e = sourceNodes.end();
  
  for (v_i = sourceNodes.begin(); v_i != v_e; ++v_i) {
    remove_edge(*v_i, deleteNode, G);
  }  
  
  /* Finish with something like this
   
   |----> D1
   S   R--> T--| 
   |----> D2
   
   Due to some weird graph side effects, we don't delete the 'S' node, but
   rather just get rid of all the edges so it essentially ceases to exist
   in terms of the graph
   */
  
  return 0;
}



int FunctionBFC::collapseAll(set<int>& collapseInstructions)
{
  //bool inserted;
  int collapsedEdges = 0;
  
  graph_traits <MyGraphType>::edge_descriptor ed;
  property_map<MyGraphType, edge_iore_t>::type edge_type = get(edge_iore, G);
  
  graph_traits<MyGraphType>::vertex_iterator i, v_end; 
  
  for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
    
    NodeProps *v = get(get(vertex_props, G),*i);
    int v_index = get(get(vertex_index, G),*i);
    
    if (!v) {
#ifdef DEBUG_ERROR      
      cerr<<"Null V in collapseAll\n";
#endif 
      continue;
    }
    
    boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
    e_beg = boost::out_edges(v_index, G).first;    // edge iterator begin
    e_end = boost::out_edges(v_index, G).second;    // edge iterator end
    
    // iterate through the edges to find matching opcode
    for(; e_beg != e_end; ++e_beg) {
      int opCode = get(get(edge_iore, G),*e_beg);
      // It's one of the opcodes we are trying to collapse
      if (collapseInstructions.count(opCode)) {//currently we only process BitCast/AddrSpaceCast operation                
        // would be delete node
        NodeProps *sourceV = get(get(vertex_props,G), source(*e_beg,G));
        // would be recipient node
        NodeProps *targetV = get(get(vertex_props,G), target(*e_beg,G));
      
        //if a = bitcast b to sometype, then if a is lv/gv/fa and b is a register, then we keep a and delete b
        if (sourceV->isLocalVar || sourceV->isFormalArg || sourceV->isGlobal) {
          if (!targetV->isLocalVar && !targetV->isFormalArg && !targetV->isGlobal) {
            // Swap the order for these cases 
            transferEdgesAndDeleteNode(targetV, sourceV);
            collapsedEdges++;
#ifdef DEBUG_GRAPH_COLLAPSE
            blame_info<<"Graph__(collapseAll)--Swapping order "<<opCode<<" "<<sourceV->name
                <<" recipient "<<" with "<<targetV->name<<" being deleted."<<endl;
#endif
            break;
          }
          else {
#ifdef DEBUG_GRAPH_COLLAPSE
            blame_info<<"Graph__(collapseAll)--both source("<<sourceV->name
                <<") and target("<<targetV->name<<") are important, we don't merge them "<<endl;
#endif
            break;
          }
        }
      
        else {
          targetV->collapsed_inst = sourceV->llvm_inst;
          transferEdgesAndDeleteNode(sourceV, targetV);
#ifdef DEBUG_GRAPH_COLLAPSE
          blame_info<<"Graph__(collapseAll)--standard delete "<<opCode<<" "<<targetV->name<<" recipient ";
          blame_info<<" with "<<sourceV->name<<" being deleted."<<endl;
#endif
          collapsedEdges++;
          break;      
        }
      }
    }
  }
  
  // Not going to be calling this function again, lets
  // update our GEP pointers
  if (collapsedEdges == 0) {
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
      NodeProps *v = get(get(vertex_props, G),*i);
      int v_index = get(get(vertex_index, G),*i);
    
      if (!v) {
#ifdef DEBUG_ERROR      
        cerr<<"Null V in collapseAll\n";
#endif 
        continue;
      }
      if (v->pointsTo == NULL)
        continue;
      
      boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
      e_beg = boost::out_edges(v_index, G).first;    // edge iterator begin
      e_end = boost::out_edges(v_index, G).second;    // edge iterator end
      
      // iterate through the edges to find matching opcode
      for (; e_beg != e_end; ++e_beg) {
        int opCode = get(get(edge_iore, G),*e_beg);
    
        if (opCode == GEP_BASE_OP) {
          NodeProps * targetV = get(get(vertex_props,G), target(*e_beg,G));
          v->pointsTo = targetV;
          targetV->pointedTo.insert(v);
        }
      }
    }
  }
  
  return collapsedEdges;
}


int FunctionBFC::collapseInvoke()
{
  //bool inserted;
  int collapsedEdges = 0;
  graph_traits < MyGraphType >::edge_descriptor ed;
  property_map<MyGraphType, edge_iore_t>::type edge_type
  = get(edge_iore, G);
  
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
  
  for(tie(i,v_end) = vertices(G); i != v_end; ++i) 
  {
    
    NodeProps *v = get(get(vertex_props, G),*i);
    int v_index = get(get(vertex_index, G),*i);
    
    if (!v)
    {
#ifdef DEBUG_ERROR      
      cerr<<"Null V in collapseInvoke\n";
#endif 
      continue;
    }
    
    
    boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
    
    e_beg = boost::out_edges(v_index, G).first;    // edge iterator begin
    e_end = boost::out_edges(v_index, G).second;    // edge iterator end
    
    vector<pair<int,int>  > deleteEdges;
    
    // iterate through the edges to find matching opcode
    for(; e_beg != e_end; ++e_beg) 
    {
      int opCode = get(get(edge_iore, G),*e_beg);
      
      if (opCode == Instruction::Invoke)
      {                    
        NodeProps * sourceV = get(get(vertex_props,G), source(*e_beg,G));
        NodeProps * targetV = get(get(vertex_props,G), target(*e_beg,G));
        
        string::size_type sourceTmp = sourceV->name.find( "tmp", 0 );
        string::size_type targetTmp = targetV->name.find( "tmp", 0 );
        
        if (sourceV->llvm_inst != NULL && targetV->llvm_inst != NULL)
        {
          if (isa<Instruction>(sourceV->llvm_inst) && isa<Instruction>(targetV->llvm_inst))
          {
            Instruction * sourceI = cast<Instruction>(sourceV->llvm_inst);
            Instruction * targetI = cast<Instruction>(targetV->llvm_inst);
            
            // Corner case where both Tmp and Var could be true is case
            //   where you have a ifdefd local variable in the program
            //   with the substring "tmp" in it
            
            bool isSourceTmp = false, isSourceVar = false;
            bool isTargetTmp = false, isTargetVar = false;
            
            // This sees if the source node is a tmp register (as opposed to 
            //      local variable with "tmp" in its name)
            if (sourceTmp != string::npos)
              isSourceTmp = true;
            
            if ( sourceI->getOpcode() == Instruction::Alloca || sourceV->pointsTo != NULL )
              isSourceVar = true;
            
            if (isSourceTmp && isSourceVar)
            {
              isSourceTmp = false;
            }
            
            if (targetTmp != string::npos)
              isTargetTmp = true;
            
            if (targetI->getOpcode() == Instruction::Alloca || targetV->pointsTo != NULL )
              isTargetVar = true;
            
            if (isTargetTmp && isTargetVar)
            {
              isTargetTmp = false;
            }
            
            
            if ( isSourceTmp && (isTargetTmp || isTargetVar) )
            {
              collapsedEdges++;
              //cout<<"Collapsing (1) op "<<opCode<<" for "<<targetV->name<<" and "<<sourceV->name<<endl;
#ifdef DEBUG_GRAPH_COLLAPSE
              blame_info<<"Collapse(4)"<<endl;
#endif 
              transferEdgesAndDeleteNode(sourceV, targetV);
              
              break;
            }
            
          }
        }
      }
    }
  }
  
  return collapsedEdges;
}


// This collapses exception handling nodes/edges.  We don't handle them yet.
// Technically, we don't handle anything c++ but accidentally we do, but not EH
void FunctionBFC::collapseEH()
{
  
  graph_traits < MyGraphType >::edge_descriptor ed;
  property_map<MyGraphType, edge_iore_t>::type edge_type;
  edge_type = get(edge_iore, G);
  
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
                                                     
  // For data structure and continuity reasons, we don't delete nodes,
  // Rather we delete edges orphaning nodes(single nodes) essentially removing them
  // implicitly from the graph
  set<graph_traits < MyGraphType >::edge_descriptor> edgesToCollapse;
  
  for(tie(i,v_end) = vertices(G); i != v_end; ++i) {
    //TC: below could be v = vertex_props[*i];
    NodeProps *v = get(get(vertex_props, G),*i);
    
    // TODO: Elegant way to make it so we aren't completely screwed if
    //    someone throws a wrench(make difficulties) in this by 
      //    actually naming one of their
    //    local variables one of these names
      if (v) {
    if (v->name.find("save_eptr") != string::npos ||
        v->name.find("eh_ptr") != string::npos ||
      v->name.find("eh_value") != string::npos ||
      v->name.find("eh_select") != string::npos ||
      v->name.find("save_filt") != string::npos ||
      v->name.find("eh_exception") != string::npos) 
        {  
        boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
    
      e_beg = boost::out_edges(v->number, G).first;  // edge iterator begin
      e_end = boost::out_edges(v->number, G).second;  // edge iterator end
      for (; e_beg != e_end; e_beg++) {
        edgesToCollapse.insert(*e_beg);
      }
    }
      }
  }
  
  set<graph_traits < MyGraphType >::edge_descriptor>::iterator set_ed_i;
  
  for (set_ed_i = edgesToCollapse.begin(); set_ed_i != edgesToCollapse.end(); set_ed_i++) {
    remove_edge(*set_ed_i, G);
  }  
}

// Do some edge manipulation for mallocs
// Return ----->
//                Malloc
// Size Calc -->
//
//     TO
//
// Return --> Size Calc
void FunctionBFC::handleMallocs()
{  
  //cout<<"Entering handleMallocs"<<endl;
  property_map<MyGraphType, edge_iore_t>::type edge_type
  = get(edge_iore, G);
  
  bool inserted;
  graph_traits < MyGraphType >::edge_descriptor ed;
  
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
  
  for(tie(i,v_end) = vertices(G); i != v_end; ++i) 
  {
    NodeProps * v = get(get(vertex_props, G),*i);
    int v_index = get(get(vertex_index, G),*i);
    
    if (!v)
    {
#ifdef DEBUG_ERROR      
      cerr<<"Null V in handleMallocs\n";
#endif
      continue;
    }
    
    if ( strstr(v->name.c_str(),"malloc--") != NULL )
    {
      boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
      
      e_beg = boost::in_edges(v_index, G).first;    // edge iterator begin
      e_end = boost::in_edges(v_index, G).second;    // edge iterator end
      
      
      // First param (oneParam) is associated with calculation of allocation size
      // Return val  (zeroParam) is associated with location for the allocation
      int oneParam = -1;
      int zeroParam = -1;
      
      // iterate through the edges to find matching opcode
      for(; e_beg != e_end; ++e_beg) 
      {
        int opCode = get(get(edge_iore, G),*e_beg);
        
        if (opCode == Instruction::Call)
        {
          //int sourceV = get(get(vertex_index, G), source(*e_beg, G));
          //int targetV = get(get(vertex_index, G), target(*e_beg, G));
          
          NodeProps * sourceVP = get(get(vertex_props, G), source(*e_beg, G));
          NodeProps * targetVP = get(get(vertex_props, G), target(*e_beg, G));
          
          int paramNum = MAX_PARAMS + 1;
          //cerr<<"Call from "<<sourceVP->name<<" to "<<targetVP->name<<endl;
          
          set<FuncCall *>::iterator fc_i = sourceVP->funcCalls.begin();
          
          for (; fc_i != sourceVP->funcCalls.end(); fc_i++)
          {
            FuncCall * fc = *fc_i;
                      if (!fc) continue;          
            //cerr<<"FC -- "<<fc->funcName<<"  "<<targetVP->name<<endl;
            
            if (fc->funcName == targetVP->name)
            {
              paramNum = fc->paramNumber;
              if (paramNum == -1)//changed by Hui 12/31/15, 0=>-1
                zeroParam = sourceVP->number;
              else if (paramNum == 0)//changed by Hui 12/31/15 1=>0
                oneParam = sourceVP->number;
              
              break;
            }            
          }
          
          //cerr<<"Param Num "<<paramNum<<" for "<<sourceVP->name<<endl;
          
        }
      }      
      
      if (oneParam > -1 && zeroParam > -1 && oneParam != zeroParam)
      {
        
        remove_edge(oneParam, v->number, G);
        remove_edge(zeroParam, v->number, G);
        tie(ed, inserted) = add_edge(zeroParam, oneParam, G);
        
        //cerr<<"Adding edge for "<<v->name<<" "<<zeroParam<<" to "<<oneParam<<endl;
        
        if (inserted)
          edge_type[ed] = RESOLVED_MALLOC_OP;  
      }
    }
  }
  //cout<<"Leaving handleMallocs"<<endl;
}


void FunctionBFC::collapseGraph()
{
  int collapsedAny = 1;
  
  // Collapse Implicit  --- need better description here
  //collapseImplicit();
  //printDotFiles("_afterFirstCompressImp.dot", true);
  
  // Collapse Exception Handling
  collapseEH();
  
  set<int> collapseInstructions;
  
  //collapseInstructions.insert(Instruction::Load);
  collapseInstructions.insert(Instruction::BitCast);
  collapseInstructions.insert(Instruction::AddrSpaceCast);
  //collapseInstructions.insert(Instruction::FPExt);
  //collapseInstructions.insert(Instruction::Trunc);
  //collapseInstructions.insert(Instruction::ZExt);
  //collapseInstructions.insert(Instruction::SExt);
  //collapseInstructions.insert(Instruction::FPToUI);
  //collapseInstructions.insert(Instruction::FPToSI);
  //collapseInstructions.insert(Instruction::UIToFP);
  //collapseInstructions.insert(Instruction::SIToFP);
  //collapseInstructions.insert(Instruction::FPTrunc);
  //collapseInstructions.insert(Instruction::PtrToInt);
  //collapseInstructions.insert(Instruction::IntToPtr);
  
  while (collapsedAny) {
    collapsedAny = collapseAll(collapseInstructions);  
  }
  
  //collapsedAny = 1;
  
  //while (collapsedAny)
  //{
  //collapsedAny = collapseInvoke();
  //}
  collapseIO();
  //handleMallocs();
  //printDotFiles("_beforeCompressImp.dot", true);
  //collapseRedundantImplicit();
}
