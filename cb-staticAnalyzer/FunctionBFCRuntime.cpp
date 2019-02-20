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

#include "FunctionBFC.h"

using namespace std;

int FunctionBFC::needExProc(string callName)
{
  if (callName.find("chpl_getPrivatizedCopy") == 0)
    return GET_PRIVATIZEDCOPY;
  else if (callName.find("chpl_getPrivatizedClass") == 0)
    return GET_PRIVATIZEDCLASS;
  else if (callName.find("chpl_gen_comm_get") == 0)
    return GEN_COMM_GET;
  else if (callName.find("chpl_gen_comm_put") == 0)
    return GEN_COMM_PUT;
  else if (callName.find("accessHelper") == 0)
    return ACCESSHELPER;
  else if (callName.find("chpl__convertRuntimeTypeToValue") == 0)
    return CONVERTRTTYPETOVALUE;

  // if none of above matches
  else return NO_SPECIAL;
}


void FunctionBFC::specialProcess(Instruction *pi, int specialCall, string callName)
{
#ifdef DEBUG_SPECIAL_PROC
  blame_info<<"Entering specialProcess for "<<callName<<endl;
#endif
  if (specialCall == GET_PRIVATIZEDCOPY) { 
    spGetPrivatizedCopy(pi);
  }

  else if (specialCall == GET_PRIVATIZEDCLASS) {
    spGetPrivatizedClass(pi);
  }

  else if (specialCall == GEN_COMM_GET) {
    spGenCommGet(pi);
  }

  else if (specialCall == GEN_COMM_PUT) {
    spGenCommPut(pi);
  }

  else if (specialCall == CONVERTRTTYPETOVALUE) {
    spConvertRTTypeToValue(pi);
  }
// TODO: Check whether the following 3 functions are necessary
/*  
  else if (specialCall == ACCESSHELPER) {
    spAccessHelper(pi);
  }

  else if (specialCall == WIDE_PTR_GET_ADDR) {
    spWidePtrGetAddr(pi);
  }

  else if (specialCall == WIDE_PTR_GET_NODE) {
    spWidePtrGetNode(pi);
  }
*/
}

      
void FunctionBFC::spGetPrivatizedCopy(Instruction *pi)
{
/*
    string retName;
    if (pi->hasName()) 
      retName.insert(0,pi->getName().data());
    else {
      char tempBuf[20];
      sprintf(tempBuf, "0x%x", pi);
      string name(tempBuf);
      retName.insert(0, name.c_str());
    }
*/
    NodeProps *objectPid = NULL, *pidObject = NULL;
    Value::use_iterator ui, ue;
    for (ui=pi->use_begin(),ue=pi->use_end(); ui!=ue; ui++) {
      if (Instruction *stInst = dyn_cast<Instruction>((*ui).getUser())) {
        if (stInst->getOpcode() == Instruction::Store) {
          User::op_iterator op_i = stInst->op_begin();
          Value *first = *op_i, *second = *(++op_i);
          
          string secondName;
          if (second->hasName())
            secondName = second->getName().str();
          else {
            char tempBuf[20];
            sprintf(tempBuf, "0x%x", second);
            string name(tempBuf);
            secondName.insert(0, name.c_str());
#ifdef DEBUG_SPECIAL_PROC
            blame_info<<"Weird: unamed object from pid: "<<secondName<<endl;
#endif
          }
          
          if (variables.count(secondName) >0)
            pidObject = variables[secondName];
          else {
#ifdef DEBUG_SPECIAL_PROC
            blame_info<<"Error:(pidObject)"<<secondName<<" Unfound in vars"<<endl;
#endif
            return;
          }
          break; //Have found pidObject, break for loop
        }
      }
    }
    // get objectPid
    Value *argPid = *(pi->op_begin()); //the param value passed to getPrivatized* calls
    if (Instruction *argPidInst = dyn_cast<Instruction>(argPid)) {
      if (argPidInst->getOpcode() == Instruction::Load) {
        //Chapel 1.15 may have multiple S-L chain before getting to the getPrivatized* calls
        Value *pid = getValueFromOrig(argPidInst); //get the original pid (RLS op)
        if (Instruction *pidInst = dyn_cast<Instruction>(pid)) {
          if (pidInst->getOpcode() == Instruction::Load) {
            Value *first = *(pidInst->op_begin());
            string firstName; //keep the name of realPid, not first
            
            //Chapel 1.15 use a wrapper struct of pid to reference distributed array
            //So the Node(with source var name) should follow:
            //realPid-GEP->Load->pid to get the realPid
            if (Instruction *firstInst = dyn_cast<Instruction>(first)) {
              if (firstInst->getOpcode() == Instruction::GetElementPtr) {
                Value *realPid = *(firstInst->op_begin());
                if (realPid->hasName()) 
                  firstName = realPid->getName().str();
                else {
                  char tempBuf[20];
                  sprintf(tempBuf, "0x%x", realPid);
                  string name(tempBuf);
                  firstName.insert(0, name.c_str());
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Weird: unamed pid: "<<firstName<<endl;
#endif
                }

                if (variables.count(firstName) >0) {
                  objectPid = variables[firstName];
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"We found pid (getPrivatizedCopy): "<<firstName<<endl;
#endif
                }
                else {
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Error:(objectPid)"<<firstName<<" Unfound in vars"<<endl;
#endif
                  return;
                }
              }
              else blame_info<<"CHECK ERROR: firstInst isn't GEP"<<endl;
            }
            else if (isa<ConstantExpr>(first)) { //it can be a constant expression 
              ConstantExpr *firstCe = cast<ConstantExpr>(first);
              if (firstCe->getOpcode() == Instruction::GetElementPtr) {
                Value *realPid = *(firstCe->op_begin());
                if (realPid->hasName()) 
                  firstName = realPid->getName().str();
                else {
                  char tempBuf[20];
                  sprintf(tempBuf, "0x%x", realPid);
                  string name(tempBuf);
                  firstName.insert(0, name.c_str());
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Weird: unamed pid: "<<firstName<<endl;
#endif
                }

                if (variables.count(firstName) >0) {
                  objectPid = variables[firstName];
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"We found pid (getPrivatizedCopy): "<<firstName<<endl;
#endif
                }
                else {
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Error:(objectPid)"<<firstName<<" Unfound in vars"<<endl;
#endif
                  return;
                }
              }
              else blame_info<<"CHECK ERROR: firstCe isn't GEP"<<endl;
            } 
            else blame_info<<"CHECK ERROR: first isn't inst nor constantexpr"<<endl;
          }
          else blame_info<<"CHECK ERROR: pidInst isn't Load"<<endl;
        }
        else blame_info<<"CHECK ERROR: pidInst isn't inst"<<endl;
      }
      else blame_info<<"CHECK ERROR: argPidInst isn't Load"<<endl;
    }
    else blame_info<<"CHECK ERROR: argPidInst isn't inst"<<endl;

    //when parsing internal module funcs, such as getPrivatizeCopy, you won't be
    //able to find the objectPid since the "pid" is directly passed-in as integer
    if (objectPid && pidObject) { 
      pair<NodeProps*, NodeProps*> pidToObj(objectPid, pidObject);
      distObjs.push_back(pidToObj);
      objectPid->isPid = true;
      objectPid->myObj = pidObject;
      pidObject->isObj = true;
      pidObject->myPid = objectPid;
    }
}

//getPrivatizedClass + bitcast = getPrivatizedCopy
void FunctionBFC::spGetPrivatizedClass(Instruction *pi)
{
/*
    string retName;
    if (pi->hasName()) 
      retName.insert(0,pi->getName().data());
    else {
      char tempBuf[20];
      sprintf(tempBuf, "0x%x", pi);
      string name(tempBuf);
      retName.insert(0, name.c_str());
    }
*/
    NodeProps *objectPid = NULL, *pidObject = NULL;
    Value::use_iterator ui, ue, ui2, ue2;
    for (ui=pi->use_begin(),ue=pi->use_end(); ui!=ue; ui++) {
      if (Instruction *i = dyn_cast<Instruction>((*ui).getUser())) {
        if (i->getOpcode() == Instruction::BitCast) {
          for (ui2=i->use_begin(),ue2=i->use_end(); ui2!=ue2; ui2++) {
            if (Instruction *i2 = dyn_cast<Instruction>((*ui2).getUser())) {
              if (i2->getOpcode() == Instruction::Store) {

                User::op_iterator op_i = i2->op_begin();
                Value *first = *op_i, *second = *(++op_i);
                string secondName;
                if (second->hasName())
                  secondName = second->getName().str();
                else {
                  char tempBuf[20];
                  sprintf(tempBuf, "0x%x", second);
                  string name(tempBuf);
                  secondName.insert(0, name.c_str());
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Weird2: unamed object from pid: "<<secondName<<endl;
#endif
                }
          
                if (variables.count(secondName) >0)
                  pidObject = variables[secondName];
                else {
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Error2:(pidObject)"<<secondName<<" Unfound"<<endl;
#endif
                  return;
                }
              }
            }
          }
        }
      }
    }
    // get objectPid
    Value *argPid = *(pi->op_begin()); //the param value passed to getPrivatized* calls
    if (Instruction *argPidInst = dyn_cast<Instruction>(argPid)) {
      if (argPidInst->getOpcode() == Instruction::Load) {
        //Chapel 1.15 may have multiple S-L chain before getting to the getPrivatized* calls
        Value *pid = getValueFromOrig(argPidInst); //get the original pid (RLS op)
        if (Instruction *pidInst = dyn_cast<Instruction>(pid)) {
          if (pidInst->getOpcode() == Instruction::Load) {
            Value *first = *(pidInst->op_begin());
            string firstName; //keep the name of realPid, not first
            
            //Chapel 1.15 use a wrapper struct of pid to reference distributed array
            //So the Node(with source var name) should follow:
            //realPid-GEP->Load->pid to get the realPid
            if (Instruction *firstInst = dyn_cast<Instruction>(first)) {
              if (firstInst->getOpcode() == Instruction::GetElementPtr) {
                Value *realPid = *(firstInst->op_begin());
                if (realPid->hasName()) 
                  firstName = realPid->getName().str();
                else {
                  char tempBuf[20];
                  sprintf(tempBuf, "0x%x", realPid);
                  string name(tempBuf);
                  firstName.insert(0, name.c_str());
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Weird2: unamed pid from object: "<<firstName<<endl;
#endif
                }

                if (variables.count(firstName) >0) {
                  objectPid = variables[firstName];
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"We found pid (getPrivatizedClass): "<<firstName<<endl;
#endif
                }
                else {
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Error2:(objectPid)"<<firstName<<" Unfound in vars"<<endl;
#endif
                  return;
                }
              }
              else blame_info<<"CHECK ERROR2: firstInst isn't GEP"<<endl;
            }
            else if (isa<ConstantExpr>(first)) { //it can be a constant expression 
              ConstantExpr *firstCe = cast<ConstantExpr>(first);
              if (firstCe->getOpcode() == Instruction::GetElementPtr) {
                Value *realPid = *(firstCe->op_begin());
                if (realPid->hasName()) 
                  firstName = realPid->getName().str();
                else {
                  char tempBuf[20];
                  sprintf(tempBuf, "0x%x", realPid);
                  string name(tempBuf);
                  firstName.insert(0, name.c_str());
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Weird2: unamed pid: "<<firstName<<endl;
#endif
                }

                if (variables.count(firstName) >0) {
                  objectPid = variables[firstName];
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"We found pid (getPrivatizedClass): "<<firstName<<endl;
#endif
                }
                else {
#ifdef DEBUG_SPECIAL_PROC
                  blame_info<<"Error2:(objectPid)"<<firstName<<" Unfound in vars"<<endl;
#endif
                  return;
                }
              }
              else blame_info<<"CHECK ERROR2: firstCe isn't GEP"<<endl;
            } 
            else blame_info<<"CHECK ERROR2: first isn't inst nor constantexpr"<<endl;
          }
          else blame_info<<"CHECK ERROR2: pidInst isn't Load"<<endl;
        }
        else blame_info<<"CHECK ERROR2: pidInst isn't inst"<<endl;
      }
      else blame_info<<"CHECK ERROR2: argPidInst isn't Load"<<endl;
    }
    else blame_info<<"CHECK ERROR2: argPidInst isn't inst"<<endl;

    //when parsing internal module funcs, such as getPrivatizeCopy, you won't be
    //able to find the objectPid since the "pid" is directly passed-in as integer
    if (objectPid && pidObject) { 
      pair<NodeProps*, NodeProps*> pidToObj(objectPid, pidObject);
      distObjs.push_back(pidToObj);
      objectPid->isPid = true;
      objectPid->myObj = pidObject;
      pidObject->isObj = true;
      pidObject->myPid = objectPid;
    }
}


// This function is currently NOT fully evaluated
// It will add ProblemSpace (domain) to Pid as well due to convertRuntimeTypeToValue2
void FunctionBFC::spConvertRTTypeToValue(Instruction *pi)
{
    //Not sure if the operand#1 of this func is pid always 3/29/17
    // for case like: pidTempNode=%99, pidNode=%type_tmp_chpl3
    /* store i64* %type_tmp_chpl3, i64** %ret_to_arg_ref_tmp__chpl9
     * %99 = load i64** %ret_to_arg_ref_tmp__chpl9
     * call void @chpl__convertRuntimeTypeToValue6(i64 %98, i64* %99,..)
     * %101 = load i64* %type_tmp_chpl3
     * store i64 %101, i64* %B_chpl
     */

  //basically pidTemp is the RLS from pid, but we dont hv that info at this time
  Value *pidTemp = pi->getOperand(1);
  string typeName = returnTypeName(pidTemp->getType(), string("")); 
  //In Chapel 1.15, pid is NOT i64*, it's generated complex type
  //if (typeName == "*Int") { //pid should always be i64*
    NodeProps *pidTempNode = NULL, *pidNode = NULL;
    Value *pid = NULL;
    string pidTempName, pidName;
    //get pidTemp's name first
    if (pidTemp->hasName())
      pidTempName = pidTemp->getName().str();
    else {
      char tempBuf[20];
      sprintf(tempBuf, "0x%x", pidTemp);
      string name(tempBuf);
      pidTempName.insert(0, name.c_str());
    }

    if (Instruction *i = dyn_cast<Instruction>(pidTemp)) {
      if (i->getOpcode() == Instruction::Load) {
        Value *first = i->getOperand(0); //first=%ret_to_arg_ref_tmp__chpl9
        Value::use_iterator ui, ue;
        for (ui=first->use_begin(),ue=first->use_end(); ui!=ue; ui++) {
          if (Instruction *i2 = dyn_cast<Instruction>((*ui).getUser())) {
            if (i2->getOpcode() == Instruction::Store) {
              
              Value *first2 = i2->getOperand(0);
              if (first == i2->getOperand(1) && //make sure first is the recipient of store
                  first2->getValueID() != Value::ConstantFPVal &&
                  first2->getValueID() != Value::ConstantIntVal && 
                  first2->getValueID() != Value::ConstantPointerNullVal && 
                  first2->getValueID() != Value::ConstantExprVal) {
                pid = first2; //found what we are looking for
                
                //get pid's name first
                if (pid->hasName())
                  pidName = pid->getName().str();
                else {
                  char tempBuf[20];
                  sprintf(tempBuf, "0x%x", pid);
                  string name(tempBuf);
                  pidName.insert(0, name.c_str());
                }
#ifdef DEBUG_SPECIAL_PROC
                blame_info<<"We found pid (convertRT): "<<pidName<<endl;
#endif
                break;
              }
            } //store
          }
        }
      } //load
    } //pidTemp

    if (!pidName.empty() && !pidTempName.empty()) {
      if (variables.count(pidName) && variables.count(pidTempName)) {
        pidNode = variables[pidName];
        pidTempNode = variables[pidTempName];
        if (pidNode && pidTempNode) {
          pidTempNode->isPid = true;
          pidNode->isPid = true;
          //This pid is discovered by checking convertRuntimeTypeToValue
          //so we need to do extra forward alias resolving for this later
          pidNode->isPidFromT2V = true; 
#ifdef DEBUG_SPECIAL_PROC
          blame_info<<"New pids added: "<<pidName<<" "<<pidTempName<<endl;
#endif
        }
      }
    }
  //}

  /*else {
#ifdef DEBUG_SPECIAL_PROC
  //  blame_info<<"This is not the convertRuntimeTypeToValue we are looking for"<<endl;
#endif
  }*/
}


void FunctionBFC::spGenCommGet(Instruction *pi)
{
    Value *dstAddr = pi->getOperand(0);
    NodeProps *dstAddrNode = NULL;
    dstAddrNode = getNodeBitCasted(dstAddr);
    // sometimes the dstAddrNode wasn't bitcasted, just get itself
    if (!dstAddrNode) {
#ifdef DEBUG_SPECIAL_PROC
      blame_info<<"Weird: what's opCode for comm_get?"<<endl;
#endif
      string argName;
      if (dstAddr->hasName()) 
        argName = dstAddr->getName().str();
      else {
        char tempBuf[20];
        sprintf(tempBuf, "0x%x", dstAddr);
        string name(tempBuf);
        argName.insert(0, name.c_str());
      }
        
      if (variables.count(argName) >0)
        dstAddrNode = variables[argName];
      else {
#ifdef DEBUG_SPECIAL_PROC
        blame_info<<"Error: (dstAddrNode)"<<argName<<" unFound in vars"<<endl;
#endif
        return;
      }
    }
    
    dstAddrNode->isRemoteWritten = true;
    dstAddrNode->isWritten = true; //Not sure whether we need to distinguish these 2 written
}

// Almost the same as spGenCommGet, except we getOperand(2) now
void FunctionBFC::spGenCommPut(Instruction *pi)
{
    Value *dstAddr = pi->getOperand(2);
    NodeProps *dstAddrNode = NULL;
    dstAddrNode = getNodeBitCasted(dstAddr);
    // sometimes dstAddrNode wasn't bitcasted, just get itself
    if (!dstAddrNode) {
#ifdef DEBUG_SPECIAL_PROC
      blame_info<<"Weird: what's opCode for comm_put?"<<endl;
#endif
      string argName;
      if (dstAddr->hasName()) 
        argName = dstAddr->getName().str();
      else {
        char tempBuf[20];
        sprintf(tempBuf, "0x%x", dstAddr);
        string name(tempBuf);
        argName.insert(0, name.c_str());
      }
        
      if (variables.count(argName) >0)
        dstAddrNode = variables[argName];
      else {
#ifdef DEBUG_SPECIAL_PROC
        blame_info<<"Error: (dstAddrNode)"<<argName<<" unFound in vars"<<endl;
#endif
        return;
      }
    }
    
    dstAddrNode->isRemoteWritten = true;
    dstAddrNode->isWritten = true; //Not sure whether we need to distinguish these 2 written
}


void FunctionBFC::spAccessHelper(Instruction *pi)
{
    string instName, argName;
    if (pi->hasName())
      instName = pi->getName().str();
    else {
      char tempBuf[20];
      sprintf(tempBuf, "0x%x", pi);
      string name(tempBuf);
      instName.insert(0, name.c_str());
    }

    Value *arg = pi->getOperand(1);
    if (arg->hasName())
      instName = arg->getName().str();
    else {
      char tempBuf[20];
      sprintf(tempBuf, "0x%x", arg);
      string name(tempBuf);
      argName.insert(0, name.c_str());
    }

    NodeProps *instNode, *argNode;
    if (variables.count(instName)>0 && variables.count(argName)>0) {
      instNode = variables[instName];
      argNode = variables[argName];

      //TO CHECK: what relationship should we build here?
      argNode->arrayAccess.insert(instNode);
    }
}


void FunctionBFC::resolvePidAliases()
{
#ifdef DEBUG_SPECIAL_PROC
    blame_info<<"\nWe are in resolvePidAliases now\n";
#endif
    graph_traits<MyGraphType>::vertex_iterator i, v_end; 

    // 1st round pidAlias search: backward straight
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
	  NodeProps *v = get(get(vertex_props, G),*i);
	  int v_index = get(get(vertex_index, G),*i);
      if (!v)
        continue;

      if (v->isTempPid || v->isPid) {
        v->isPid = true; //set all temp pid from 0th round to be real pid
        set<int> visited;
        resolvePidAliasForNode_bw_new(v, visited);
      }
    }

    // 2nd round pidAlias search: forward 
    // we need to forward check each pid to add in more aliases
    // for cases when more than one aliasOut exist in the middle
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
      NodeProps *v = get(get(vertex_props, G),*i);
      if (!v)
        continue;
      
      if (v->isTempPid || v->isPid) {
        v->isPid = true; //set all temp pid from 1st round to be real pid
        set<int> visited;
        resolvePidAliasForNode_fw_new(v, visited);
      }
    } 

    // 3rd round pidAlias search: merge in & out
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
	  NodeProps *v = get(get(vertex_props, G),*i);
      if (!v)
        continue;
      if (v->isTempPid)
        v->isPid = true; //set new temp pids gotten from the 2nd round search above
      //merge all pidAliasesIn&pidAliasesOut in to pidAliases
      if (v->isPid) {
        set<NodeProps *>::iterator si, se;
        for (si=v->pidAliasesIn.begin(), se=v->pidAliasesIn.end(); si!=se; si++)
          v->pidAliases.insert(*si);
        for (si=v->pidAliasesOut.begin(), se=v->pidAliasesOut.end(); si!=se; si++)
          v->pidAliases.insert(*si);
      }
    }

    // 4th round pidAlias search: make everyone equal
    // Transitive pidAliases, just like processing aliases
    resolveTransitivePidAliases();
}


void FunctionBFC::resolveObjAliases()
{
#ifdef DEBUG_SPECIAL_PROC
    blame_info<<"\nWe are in resolveObjAliases now\n";
#endif
    graph_traits<MyGraphType>::vertex_iterator i, v_end; 
	
    for (tie(i,v_end) = vertices(G); i != v_end; ++i) {
	  NodeProps *v = get(get(vertex_props, G),*i);
      if (!v)
        continue;

      if (v->isObj) {
        //Get the pointer level of obj
#ifdef DEBUG_SPECIAL_PROC
        blame_info<<"Met obj "<<v->name<<" here1"<<endl;
#endif
        int origPtrLevel=0;
	    if (v->llvm_inst != NULL) {
	      if (isa<Instruction>(v->llvm_inst)) {
		    Instruction *pi = cast<Instruction>(v->llvm_inst);	
            const llvm::Type *origT = pi->getType();	
		    origPtrLevel = pointerLevel(origT,0);
	      }
	      else if (isa<ConstantExpr>(v->llvm_inst)) {
		    ConstantExpr *ce = cast<ConstantExpr>(v->llvm_inst);
		    const llvm::Type *origT = ce->getType();
	        origPtrLevel = pointerLevel(origT, 0);
	      }
	    }
        if (origPtrLevel <= 1) {//should be >=2
#ifdef DEBUG_SPECIAL_PROC
          blame_info<<"Woops! obj isn't 2-level ptr"<<endl;
#endif
          continue;
        }
#ifdef DEBUG_SPECIAL_PROC
        blame_info<<"Met obj "<<v->name<<" here2"<<endl;
#endif
        v->objAliasesIn = v->aliasesIn;
        v->objAliasesOut = v->aliasesOut;
        v->objAliases = v->aliases;
      }
    }
#ifdef DEBUG_SPECIAL_PROC
    blame_info<<"Arrive here for "<<getSourceFuncName()<<endl;
#endif

    // Transitive objAliases, just like processing aliases
    resolveTransitiveObjAliases();
}


// Resolve potential pid aliases for formalArgs(not detected as pid in "this" func
// maybe used in postmortem2 later once pid params reflected by a callee
void FunctionBFC::resolvePPA()
{
    RegHashProps::iterator begin, end;
    for (begin = variables.begin(), end = variables.end(); begin != end; begin++) {
      NodeProps *v = begin->second;
      //we only care about the EVs that aren't Pids (not detected by this func)
      if (v->isFormalArg && !v->isPid) {
        int ptrlvl = pointerLevel(v->llvm_inst->getType(), 0);
        if (ptrlvl >= 1) { 
          //first, we add all preRLS to PPAs
          set<int> visited1;
          resolvePreRLS(v, v, visited1);
          v->PPAs.insert(v->preRLS.begin(), v->preRLS.end());
          //second, we add preRLS or 'aliases' of each preRLS to PPAs
          set<NodeProps *>::iterator ivp;
          for (ivp = v->preRLS.begin(); ivp != v->preRLS.end(); ivp++) {
            NodeProps *rls = *ivp;
            set<int> visited2;
            resolvePPAFromRLSNode(v, rls, visited2); //very similar to "v->isPidFromT2V"
          }
        }
      }
    }
}


// Helper function 1 for resolvePPA
void FunctionBFC::resolvePreRLS(NodeProps *origV, NodeProps *currNode, set<int> &visited)
{
#ifdef DEBUG_RESOLVEPPA
    blame_info<<"Entering resolvePreRLS for "<<origV->name<<" from "<<currNode->name<<endl;
#endif
    if (visited.count(currNode->number))
      return;
    visited.insert(currNode->number);

    boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
	e_beg = boost::in_edges(currNode->number, G).first;	//edge iterator begin
	e_end = boost::in_edges(currNode->number, G).second; // edge iterator end
    // farward tracing for pidAliases
    for (; e_beg!=e_end; e_beg++) {
      int opCode = get(get(edge_iore, G), *e_beg);
      if (opCode == Instruction::Store) {  
        NodeProps *sourceV = get(get(vertex_props,G), source(*e_beg,G));
  
        boost::graph_traits<MyGraphType>::in_edge_iterator e_beg2, e_end2;
	    e_beg2 = boost::in_edges(sourceV->number, G).first;	//edge iterator begin
	    e_end2 = boost::in_edges(sourceV->number, G).second; // edge iterator end

        for (; e_beg2!=e_end2; e_beg2++) {
          int opCode2 = get(get(edge_iore, G), *e_beg2);
          if (opCode2 == Instruction::Load) {
            NodeProps *targetV = get(get(vertex_props,G), source(*e_beg2,G));
            // Insert targetV to the origV !!! Not currNode!!!
            origV->preRLS.insert(targetV); //we found one forward 'alias' of origV
            //Start recursion on targetV
            resolvePreRLS(origV, targetV, visited);
          } //if opCode2=Load
        }//for edges2
      }//if opCode=Store
    }//for edges
}


// Helper function 2 for resolvePPA
void FunctionBFC::resolvePPAFromRLSNode(NodeProps *origV, NodeProps *currNode, set<int> &visited)
{
#ifdef DEBUG_RESOLVEPPA
    blame_info<<"Entering resolvePPAFromRLSNode for "<<origV->name<<" from "<<currNode->name<<endl;
#endif
    if (visited.count(currNode->number))
      return;
    visited.insert(currNode->number);

    boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
	e_beg = boost::in_edges(currNode->number, G).first;	//edge iterator begin
	e_end = boost::in_edges(currNode->number, G).second; // edge iterator end
    // farward tracing for pidAliases
    for (; e_beg!=e_end; e_beg++) {
      int opCode = get(get(edge_iore, G), *e_beg);
      if (opCode == Instruction::Load) {  
        NodeProps *sourceV = get(get(vertex_props,G), source(*e_beg,G));
  
        boost::graph_traits<MyGraphType>::in_edge_iterator e_beg2, e_end2;
	    e_beg2 = boost::in_edges(sourceV->number, G).first;	//edge iterator begin
	    e_end2 = boost::in_edges(sourceV->number, G).second; // edge iterator end

        for (; e_beg2!=e_end2; e_beg2++) {
          int opCode2 = get(get(edge_iore, G), *e_beg2);
          if (opCode2 == Instruction::Store) {
            NodeProps *targetV = get(get(vertex_props,G), source(*e_beg2,G));
            // Insert targetV to the origV !!! Not currNode!!!
            origV->PPAs.insert(targetV); //we found one forward 'alias' of origV
            //Start recursion on targetV
            resolvePPAFromRLSNode(origV, targetV, visited);
          } //if opCode2=Store
        }//for edges2
      }//if opCode=Load
    }//for edges
}     

            
// Backward search new: Chapel 1.15 establish pidAliases NOT based on aliases mechanism
// but based on RLS mechanism: A-store->C-load->B, so backward search from B should
// first check out_edge(LD) to C, then out_edge(ST) to A from C
void FunctionBFC::resolvePidAliasForNode_bw_new(NodeProps *currNode, set<int> &visited)
{
#ifdef DEBUG_SPECIAL_PROC
    blame_info<<"Entering recursively resolvePidAliasForNode_bw_new for "<<currNode->name<<endl;
#endif
    if (visited.count(currNode->number)) 
      return;
    visited.insert(currNode->number);

    boost::graph_traits<MyGraphType>::out_edge_iterator e_beg, e_end;
	e_beg = boost::out_edges(currNode->number, G).first;	//edge iterator begin
	e_end = boost::out_edges(currNode->number, G).second; // edge iterator end
    // Backward tracing for pidAliases
    for (; e_beg!=e_end; e_beg++) {
      int opCode = get(get(edge_iore, G), *e_beg);
      // if the bw propagation follows LD->ST chain
      if (opCode == Instruction::Load) {  
        NodeProps *sourceV = get(get(vertex_props,G), target(*e_beg,G));
  
        boost::graph_traits<MyGraphType>::out_edge_iterator e_beg2, e_end2;
	    e_beg2 = boost::out_edges(sourceV->number, G).first;	//edge iterator begin
	    e_end2 = boost::out_edges(sourceV->number, G).second; // edge iterator end

        for (; e_beg2!=e_end2; e_beg2++) {
          int opCode2 = get(get(edge_iore, G), *e_beg2);
          if (opCode2 == Instruction::Store) {
            NodeProps *targetV = get(get(vertex_props,G), target(*e_beg2,G));
            // For pid, their type should always be i64* as far as I know
#ifdef DEBUG_SPECIAL_PROC
            blame_info<<"We find a pidAliasIn(1): "<<targetV->name<<" for "<<currNode->name<<endl;
#endif
            //Copy some atrributes from currNode
            if (!targetV->isPid && (currNode->isPid||currNode->isTempPid))
              targetV->isTempPid = true; //Later, we'll change all tempPid to pid
            if (!targetV->isWritten && currNode->isWritten)
              targetV->isWritten = true;

            currNode->pidAliasesIn.insert(targetV);
            targetV->pidAliasesOut.insert(currNode);
            //Start recursion on targetV
            resolvePidAliasForNode_bw_new(targetV, visited);
          } //if opCode2=Load
        }//for edges2
      }//if opCode=Store
      // if the bw propagation follows ST->LD chain
      else if (opCode == Instruction::Store) {  
        NodeProps *sourceV = get(get(vertex_props,G), target(*e_beg,G));
  
        boost::graph_traits<MyGraphType>::out_edge_iterator e_beg2, e_end2;
	    e_beg2 = boost::out_edges(sourceV->number, G).first;	//edge iterator begin
	    e_end2 = boost::out_edges(sourceV->number, G).second; // edge iterator end

        for (; e_beg2!=e_end2; e_beg2++) {
          int opCode2 = get(get(edge_iore, G), *e_beg2);
          if (opCode2 == Instruction::Load) {
            NodeProps *targetV = get(get(vertex_props,G), target(*e_beg2,G));
            // For pid, their type should always be i64* as far as I know
#ifdef DEBUG_SPECIAL_PROC
            blame_info<<"We find a pidAliasIn(2): "<<targetV->name<<" for "<<currNode->name<<endl;
#endif
            //Copy some atrributes from currNode
            if (!targetV->isPid && (currNode->isPid||currNode->isTempPid))
              targetV->isTempPid = true; //Later, we'll change all tempPid to pid
            if (!targetV->isWritten && currNode->isWritten)
              targetV->isWritten = true;

            currNode->pidAliasesIn.insert(targetV);
            targetV->pidAliasesOut.insert(currNode);
            //Start recursion on targetV
            resolvePidAliasForNode_bw_new(targetV, visited);
          } //if opCode2=Load
        }//for edges2
      }//if opCode=Store
    }//for edges
}


/* Foward Search Old
void FunctionBFC::resolvePidAliasForNode_fw(NodeProps *currNode, set<int> &visited)
{
#ifdef DEBUG_SPECIAL_PROC
    blame_info<<"Entering recursively resolvePidAliasForNode_fw for "<<currNode->name<<endl;
#endif
    if (visited.count(currNode->number)) 
      return;
    visited.insert(currNode->number);

    boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
	e_beg = boost::in_edges(currNode->number, G).first;	//edge iterator begin
	e_end = boost::in_edges(currNode->number, G).second; // edge iterator end
    // farward tracing for pidAliases
    for (; e_beg!=e_end; e_beg++) {
      int opCode = get(get(edge_iore, G), *e_beg);
      if (opCode == Instruction::Load) {  
        NodeProps *sourceV = get(get(vertex_props,G), source(*e_beg,G));
  
        boost::graph_traits<MyGraphType>::in_edge_iterator e_beg2, e_end2;
	    e_beg2 = boost::in_edges(sourceV->number, G).first;	//edge iterator begin
	    e_end2 = boost::in_edges(sourceV->number, G).second; // edge iterator end

        for (; e_beg2!=e_end2; e_beg2++) {
          int opCode2 = get(get(edge_iore, G), *e_beg2);
          if (opCode2 == Instruction::Store) {
            NodeProps *targetV = get(get(vertex_props,G), source(*e_beg2,G));
            // For pid, their type should always be i64* as far as I know
#ifdef DEBUG_SPECIAL_PROC
            blame_info<<"We find a pidAliasOut: "<<targetV->name<<" for "<<currNode->name<<endl;
#endif
            //Copy some atrributes from currNode
            if (!targetV->isPid && (currNode->isPid||currNode->isTempPid))
              targetV->isTempPid = true; //Later, we'll change all tempPid to pid
            if (!targetV->isWritten && currNode->isWritten)
              targetV->isWritten = true;

            currNode->pidAliasesOut.insert(targetV);
            targetV->pidAliasesIn.insert(currNode);
            
            //Start recursion on targetV
            resolvePidAliasForNode_fw(targetV, visited);
          } //if opCode2=Store
        }//for edges2
      }//if opCode=Load
    }//for edges

    //03/30/17: we need to take care of some special case: a/b are fields
    //When a(early) and b(late) are collapsePair, and b is not finally deleted
    //due to some reasons we added, then we need to mark b as Pid if a is Pid
    string fieldName;
    if (!(currNode->uniqueNameAsField.empty()))
      fieldName = string(currNode->uniqueNameAsField);
    if (!fieldName.empty()) {
      if (cpHash.count(fieldName) >0) {
        if (currNode == cpHash[fieldName]) { //currNode is destVertex
          set<NodeProps *>::iterator si = currNode->collapseNodes.begin();
          set<NodeProps *>::iterator se = currNode->collapseNodes.end();
          for (; si != se; si++) {
            NodeProps *cpNode = *si;
            int v_index = cpNode->number;
            int in_d = in_degree(v_index, G);
            int out_d = out_degree(v_index, G);
            // if collapseVertex still has edges attached, then it wasn't deleted
            if ((in_d+out_d) > 0) {
#ifdef DEBUG_SPECIAL_PROC
              blame_info<<"We find a pidAliasOut(cp): "<<cpNode->name<<" for "<<currNode->name<<endl;
#endif
              //Copy some atrributes from currNode
              if (!cpNode->isPid && (currNode->isPid||currNode->isTempPid))
                cpNode->isTempPid = true; //Later, we'll change all tempPid to pid
              //We set cpNode as currNode's pidAliasOut because destVertex always
              //appears early than the collapseVertex
              currNode->pidAliasesOut.insert(cpNode);
              cpNode->pidAliasesIn.insert(currNode);
            
              //Start recursion on targetV
              resolvePidAliasForNode_fw(cpNode, visited);
            } // cpNode not actually deleted
          } // iterate all collapseNodes
        } // currNode exists as destVertex (the one that's retained)
      } // if the collapsable pair exists
    } // if currNode is a field
}
*/

// Forward search new: Chapel 1.15 establish pidAliases NOT based on aliases mechanism
// but based on RLS mechanism: A-store->C-load->B, so forward search from A should
// first check in_edge(ST) from C, then in_edge(LD) from A
void FunctionBFC::resolvePidAliasForNode_fw_new(NodeProps *currNode, set<int> &visited)
{
#ifdef DEBUG_SPECIAL_PROC
    blame_info<<"Entering recursively resolvePidAliasForNode_fw_new for "<<currNode->name<<endl;
#endif
    if (visited.count(currNode->number)) 
      return;
    visited.insert(currNode->number);

    boost::graph_traits<MyGraphType>::in_edge_iterator e_beg, e_end;
	e_beg = boost::in_edges(currNode->number, G).first;	//edge iterator begin
	e_end = boost::in_edges(currNode->number, G).second; // edge iterator end
    // farward tracing for pidAliases
    for (; e_beg!=e_end; e_beg++) {
      int opCode = get(get(edge_iore, G), *e_beg);
      // if the propagation follows ST->LD chain
      if (opCode == Instruction::Store) {  
        NodeProps *sourceV = get(get(vertex_props,G), source(*e_beg,G));
  
        boost::graph_traits<MyGraphType>::in_edge_iterator e_beg2, e_end2;
	    e_beg2 = boost::in_edges(sourceV->number, G).first;	//edge iterator begin
	    e_end2 = boost::in_edges(sourceV->number, G).second; // edge iterator end

        for (; e_beg2!=e_end2; e_beg2++) {
          int opCode2 = get(get(edge_iore, G), *e_beg2);
          if (opCode2 == Instruction::Load) {
            NodeProps *targetV = get(get(vertex_props,G), source(*e_beg2,G));
            // For pid, their type should always be i64* as far as I know
#ifdef DEBUG_SPECIAL_PROC
            blame_info<<"We find a pidAliasOut(1): "<<targetV->name<<" for "<<currNode->name<<endl;
#endif
            //Copy some atrributes from currNode
            if (!targetV->isPid && (currNode->isPid||currNode->isTempPid))
              targetV->isTempPid = true; //Later, we'll change all tempPid to pid
            if (!targetV->isWritten && currNode->isWritten)
              targetV->isWritten = true;

            currNode->pidAliasesOut.insert(targetV);
            targetV->pidAliasesIn.insert(currNode);
            
            //Start recursion on targetV
            resolvePidAliasForNode_fw_new(targetV, visited);
          } //if opCode2=Store
        }//for edges2
      }//if opCode=Load
      // if the propagattion follows LD->ST chain 
      else if (opCode == Instruction::Load) {  
        NodeProps *sourceV = get(get(vertex_props,G), source(*e_beg,G));
  
        boost::graph_traits<MyGraphType>::in_edge_iterator e_beg2, e_end2;
	    e_beg2 = boost::in_edges(sourceV->number, G).first;	//edge iterator begin
	    e_end2 = boost::in_edges(sourceV->number, G).second; // edge iterator end

        for (; e_beg2!=e_end2; e_beg2++) {
          int opCode2 = get(get(edge_iore, G), *e_beg2);
          if (opCode2 == Instruction::Store) {
            NodeProps *targetV = get(get(vertex_props,G), source(*e_beg2,G));
            // For pid, their type should always be i64* as far as I know
#ifdef DEBUG_SPECIAL_PROC
            blame_info<<"We find a pidAliasOut(2): "<<targetV->name<<" for "<<currNode->name<<endl;
#endif
            //Copy some atrributes from currNode
            if (!targetV->isPid && (currNode->isPid||currNode->isTempPid))
              targetV->isTempPid = true; //Later, we'll change all tempPid to pid
            if (!targetV->isWritten && currNode->isWritten)
              targetV->isWritten = true;

            currNode->pidAliasesOut.insert(targetV);
            targetV->pidAliasesIn.insert(currNode);
            
            //Start recursion on targetV
            resolvePidAliasForNode_fw_new(targetV, visited);
          } //if opCode2=Store
        }//for edges2
      }//if opCode=Load
    }//for edges

    //03/30/17: we need to take care of some special case: a/b are fields
    //When a(early) and b(late) are collapsePair, and b is not finally deleted
    //due to some reasons we added, then we need to mark b as Pid if a is Pid
    string fieldName;
    if (!(currNode->uniqueNameAsField.empty()))
      fieldName = string(currNode->uniqueNameAsField);
    if (!fieldName.empty()) {
      if (cpHash.count(fieldName) >0) {
        if (currNode == cpHash[fieldName]) { //currNode is destVertex
          set<NodeProps *>::iterator si = currNode->collapseNodes.begin();
          set<NodeProps *>::iterator se = currNode->collapseNodes.end();
          for (; si != se; si++) {
            NodeProps *cpNode = *si;
            int v_index = cpNode->number;
            int in_d = in_degree(v_index, G);
            int out_d = out_degree(v_index, G);
            // if collapseVertex still has edges attached, then it wasn't deleted
            if ((in_d+out_d) > 0) {
#ifdef DEBUG_SPECIAL_PROC
              blame_info<<"We find a pidAliasOut(cp): "<<cpNode->name<<" for "<<currNode->name<<endl;
#endif
              //Copy some atrributes from currNode
              if (!cpNode->isPid && (currNode->isPid||currNode->isTempPid))
                cpNode->isTempPid = true; //Later, we'll change all tempPid to pid
              //We set cpNode as currNode's pidAliasOut because destVertex always
              //appears early than the collapseVertex
              currNode->pidAliasesOut.insert(cpNode);
              cpNode->pidAliasesIn.insert(currNode);
            
              //Start recursion on targetV
              resolvePidAliasForNode_fw_new(cpNode, visited);
            } // cpNode not actually deleted
          } // iterate all collapseNodes
        } // currNode exists as destVertex (the one that's retained)
      } // if the collapsable pair exists
    } // if currNode is a field
}


void FunctionBFC::resolveTransitivePidAliases()
{
    graph_traits<MyGraphType>::vertex_iterator i, v_end;
    for (tie(i, v_end)=vertices(G); i!=v_end; i++) {
      NodeProps *v = get(get(vertex_props, G), *i);
      if (!v)
        continue;
      
      set<NodeProps*>::iterator si, si2;
      for (si=v->pidAliases.begin(); si!=v->pidAliases.end(); si++) {
        NodeProps *al1 = *si;
        for (si2=si; si2!=v->pidAliases.end(); si2++) {
          NodeProps *al2 = *si2;
          if (al1 == al2)
            continue;
          if (al1 != v) {
#ifdef DEBUG_SPECIAL_PROC
            blame_info<<"Transi pA: "<<al1->name<<" and "<<al2->name<<endl;
#endif
            al1->pidAliases.insert(al2);
          }
          if (al2 != v) {
#ifdef DEBUG_SPECIAL_PROC
            blame_info<<"Transi pA: "<<al2->name<<" and "<<al1->name<<endl;
#endif
            al2->pidAliases.insert(al1);
          }
        } //3rd for loop
      } //2nd for loop
    } //1sr for loop
}

// Since objAliases was computed before, we just do the copy
void FunctionBFC::resolveTransitiveObjAliases()
{
    set<int> visited;
    graph_traits<MyGraphType>::vertex_iterator i, v_end;
    for (tie(i, v_end)=vertices(G); i!=v_end; i++) {
      NodeProps *v = get(get(vertex_props, G), *i);
      if (!v)
        continue;
      if (visited.count(v->number)) //set visited to avoid infinite loop
        continue;                   //skip this loop if any of its aliases is processed

      if (v->isObj) {
        visited.insert(v->number); //add v to visited
        set<NodeProps*>::iterator si;
        for (si=v->objAliases.begin(); si!=v->objAliases.end(); si++) {
          NodeProps *vA = *si;
          vA->isObj = true;
          vA->objAliasesIn = vA->aliasesIn;
          vA->objAliasesOut = vA->aliasesOut;
          vA->objAliases = vA->aliases;
          visited.insert(vA->number); //add vA to visited
        }
      }
    }
}
