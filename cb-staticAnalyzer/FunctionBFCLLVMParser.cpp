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
#include "ModuleBFC.h"
#include "FunctionBFC.h"
#include "FunctionBFCCFG.h"

#include <iostream>
#include <sstream> 
//#include "llvm/Support/raw_os_ostream.h"
//struct fltSemantics;
using namespace std;

void FunctionBFC::parseLLVM(vector<NodeProps *> &globalVars)
{
    // push back all the exit variables in this function(ret, pointer params)
    determineFunctionExitStatus();
    // set isBFCPoint to be true if no exit variables
    RegHashProps::iterator begin, end;
  
    populateGlobals(globalVars);
  
    int varCount = globalVars.size(); 
    int currentLineNum = 0; 
  
    // We iterate through the function first basic block by basic block, 
    // then instruction by instruction (within basic block) to determine the 
    // symbols that will be used to create dependencies (put in "variables")
    for (Function::iterator b = func->begin(), be = func->end(); b != be; ++b) {
    //llvm 4.0 needs to deference+reference the iterator to get the pointer
    FunctionBFCBB *fbb = new FunctionBFCBB(&*b); //FunctionBFCBB is defined in FunctionBFCCFG.h
    
    cfg->LLVM_BBs[b->getName().str()] = fbb;
    
      for (BasicBlock::iterator i = (*b).begin(), ie = (*b).end(); i != ie; ++i) {
        examineInstruction(&*i, varCount, currentLineNum, variables, fbb);      
      }
    }
  
  cfg->genEdges(); //gen pred/succ edges between FunctionBFCBBs
  cfg->genAD();   //gen ancestors & descendants of a FunctionBFCBB

    //for test Hui
    //printCurrentVariables();

  adjustLVnEVs(); //We tag true localVar and formal arg as exitVar
    generateImplicits();  
}

void FunctionBFC::adjustLVnEVs()
{
  RegHashProps::iterator begin, end;
  /* Iterate through variables and label them with their node IDs (integers)
   -> first is string, associated with their name
   -> second is NodeProps * for node
  */
  blame_info<<"#variables="<<variables.size()<<", #localVars="<<localVars.size()<<", #exiVariables="<<exitVariables.size()<<endl;

  //Check if v is LV first
  for (begin = variables.begin(), end = variables.end(); begin != end; begin++) {
    NodeProps *v = begin->second;
    vector<LocalVar *>::iterator lv_i;
  for (lv_i = localVars.begin();  lv_i != localVars.end(); lv_i++) {
      string chplName = string(begin->first);
      if (chplName.compare((*lv_i)->varName) == 0) {
#ifdef DEBUG_LOCALS      
        blame_info<<"Local Var found begin="<<begin->first<<
          ", it's line_num will be "<<(*lv_i)->definedLine<<endl;
#endif        
        v->isLocalVar = true;
        v->line_num = (*lv_i)->definedLine;
        allLineNums.insert(v->line_num);

        if ((*lv_i)->varName.find(".") != string::npos  
            && (*lv_i)->varName.find("equiv.") == string::npos   
            && (*lv_i)->varName.find("result.") == string::npos) 
        {
          v->isFakeLocal = true; //no use later
        }
        break; //No need to continue search for this v anymore
      }
    }//end of localVars
  } //end of variables

  //Now we check if v is EV, We don't tag arg holders/global vars/retVal as isFormalAr
  for (begin = variables.begin(), end = variables.end(); begin != end; begin++) {
    NodeProps *v = begin->second;
    //string vName = v->name;
    string vName = begin->first;
    if (!vName.empty() && vName.find("retval")==string::npos && v->isGlobal==false) {
      vector<ExitVariable *>::iterator ev_i; 
      for (ev_i=exitVariables.begin(); ev_i != exitVariables.end(); ev_i++) {
      if (vName.compare((*ev_i)->realName)==0) {
#ifdef DEBUG_EXIT
          //for formal arg that don't have arg holders and used directly in the func
        blame_info<<"We found a formalArgHolder(EV): "<<vName<<endl;
#endif
          v->isFormalArg = true; //really it's the formal arg holder 10/23/17
          //v->llvm_inst = (*ev_i)->llvmNode;//we dont hv alloc for v, use formal arg
          if (v->isLocalVar) {
#ifdef DEBUG_EXIT
          blame_info<<"Weird arg (lv&ev): "<<vName<<endl;
#endif
          }
          break; //No need to search the exitvariables for this node anymore
        }
      }
    }
  } //end of variables
}


void FunctionBFC::populateGlobals(vector<NodeProps *> &gvs)
{
  vector<NodeProps *>::iterator gv_i, gv_e;
  for (gv_i = gvs.begin(), gv_e = gvs.end(); gv_i != gv_e; gv_i++) {
    NodeProps *global = *gv_i;
    
#ifdef DEBUG_VP_CREATE
    blame_info<<"Adding global NodeProps(9) for "<<global->name<<endl;
#endif
    
    NodeProps *np = new NodeProps(global->number, global->name, global->line_num, global->llvm_inst);
    np->realName = global->realName; //deep copy 06/20/18
    np->isGlobal = true;
    np->impNumber = 0;
    variables[np->name] = np; //RegHashProps variables: string --> NodeProps*
    //addGlobalExitVar(new ExitVariable(vp->name.c_str(), GLOBAL, -1));
    addExitVar(new ExitVariable(np->name, GLOBAL, -2, false)); //changed by Hui 03/15/16: whichParam from -1 to -2
                                                            //because we set retVal to -1 now
  }
}


void FunctionBFC::propagateConditional(DominatorTreeBase<BasicBlock> *DT, const DomTreeNodeBase<BasicBlock> *N, 
                                        const char *condName, BasicBlock *termNode)
{
  BasicBlock *rootBlock = N->getBlock();
  if (rootBlock == termNode)
    return;
  
    if (rootBlock->hasName()) {
    set<const char*, ltstr> t;
    ImpRegSet::iterator irs_i;
    
    irs_i = iReg.find(rootBlock->getName().str());
    if (irs_i != iReg.end()) {
      t = iReg[rootBlock->getName().str()];
    }
    
    if (condName != NULL) {
      t.insert(condName);
#ifdef DEBUG_LLVM_IMPLICIT
      blame_info<<"Inserting (4)"<<condName<<" into "<<rootBlock->getName().str()<<endl;
#endif
    }
    
    iReg[rootBlock->getName().str()] = t; //after update t, update t in iReg
    //blame_info<<"New iReg (2)"<<(unsigned) &(iReg[rootBlock->getNameStart()])<<endl;
  }
  
    for (DomTreeNodeBase<BasicBlock>::const_iterator I = N->begin(), E = N->end(); I != E; ++I)
    propagateConditional(DT, *I, condName, termNode);   

}


void FunctionBFC::gatherAllDescendants(DominatorTreeBase<BasicBlock> *DT, BasicBlock *original, 
                BasicBlock *&b, set<BasicBlock *> &cfgDesc, set<BasicBlock *> &visited)
{
  if (visited.count(b))
    return;
  visited.insert(b);
    
  if (DT->properlyDominates(b, original)) //if b dominates original
    return;
  
  for (succ_iterator Iter = succ_begin(b), En = succ_end(b); Iter != En; ++Iter) {
    BasicBlock *bb = *Iter;
    cfgDesc.insert(bb);
    gatherAllDescendants(DT, original, bb, cfgDesc, visited);
  }
}


void FunctionBFC::handleOneConditional(DominatorTreeBase<BasicBlock> *DT, \
        const DomTreeNodeBase<BasicBlock> *N, BranchInst *br)
{
#ifdef DEBUG_LLVM_IMPLICIT
    blame_info<<"Looking at handling conditional for "<<N->getBlock()->getName().str() \
        <<" for cond name "<<br->getCondition()->getName().str()<<endl;
#endif
    BasicBlock *b = N->getBlock();
  
  // We essentially need to find the terminal node for each dominator 
    // and make sure that any if/else blame does not go past the terminal node for this.  
    // Based on whether there are 2 or 3 immediately dominated nodes we find the terminal node
  // in different ways
  
  size_t numChildren = N->getNumChildren();
#ifdef DEBUG_LLVM_IMPLICIT
  blame_info<<"Number of children is "<<numChildren<<endl;
#endif
  BasicBlock *termBB = NULL;
  set<DomTreeNodeBase<BasicBlock> *> blocks;
  
  // Case where we have 
  // if ( bool)
  //    { body }
  // one child is body, one child is terminal
  
  //     OR
  
  //  else if (bool)
  //  {body}
  //  else
  //   {body}
  //  both children are bodies, terminal is not accessible by dominator tree, need to use CFG
  //  term in that case is first shared node in both CFG successors
  if (numChildren == 2) {
    BasicBlock *succ1 = br->getSuccessor(0);
    BasicBlock *succ2 = br->getSuccessor(1);
    
#ifdef DEBUG_LLVM_IMPLICIT
    blame_info<<"First successor is "<<succ1->getName().str()<<endl;
    blame_info<<"Second successor is "<<succ2->getName().str()<<endl;
#endif 
    // Second successor is either
    //  a) the terminal node (in the case of a straight if() with no else if or else
    //  b) the else body in the case of a else if/else situation (if/else with no more else if,which  has>=3 children)
    
    // First, lets handle the straight up if case
    // If an immediate dominated node is equal to the second successor then it must be the terminal 
    set<BasicBlock *> cfgDesc;
    set<BasicBlock *> visited;
    
    // Insert parent so we don't traverse the other conditional in the case of a loop
    visited.insert(b);
    gatherAllDescendants(DT, succ1, succ1, cfgDesc, visited);
    
    // If one of the descendants to succ1 is succ2 then we know that succ2 is the terminal
    if (cfgDesc.count(succ2)) {
#ifdef DEBUG_LLVM_IMPLICIT
      blame_info<<"Standard if, no else -- "<<succ2->getName().str()<<" is terminal."<<endl;
#endif
      termBB = succ2;
    }
    
    // If we don't find a terminal that's fine, the if/else and elses only dominate the stuff they should
    // The problem with the straight up if case is their basic block can dominate all the linear code
    // that comes after it
  
    for (DomTreeNodeBase<BasicBlock>::const_iterator I = N->begin(), E = N->end(); I != E; ++I) {
      BasicBlock *dest = (*I)->getBlock(); //get the bb of its children
      if (dest == succ1) {
        DomTreeNodeBase<BasicBlock> *dbb = *I;
        blocks.insert(dbb);
      }
      else if (dest == succ2) {
          // if the terminal equal null it means the second dominated block was an if/else or else block
        if (termBB == NULL) {
          DomTreeNodeBase<BasicBlock> *dbb = *I;
          blocks.insert(dbb);
        }
      }
    }
  } //end of if(numChildren==2)
  
  // Case where we have
  //   if ()
  //   {body}
  //   else if ()
  //   { body}
  //  else()
  //  one child is if body, one child is first else if test, one child is block that merges all conditions at end
  else if (numChildren == 3) {
    // This one is much easier.  If the successor in the CFG is in the dominator tree then it is one of the bodies,
    //  otherwise it is the terminal node
    
    for (DomTreeNodeBase<BasicBlock>::const_iterator I = N->begin(), E = N->end(); I != E; ++I) {
      BasicBlock *dest = (*I)->getBlock(); 
      int match = 0;
      
      // for all children in CFG
      for (succ_iterator Iter = succ_begin(b), En = succ_end(b); Iter != En; ++Iter) {
#ifdef DEBUG_LLVM_IMPLICIT
        blame_info<<"CFG - "<<Iter->getName().str()<<" D - "<<dest->getName().str()<<endl;
#endif 
        if (strcmp(Iter->getName().data(), dest->getName().data()) == 0)
          match++;
      }
      
      if (match == 0) {
#ifdef DEBUG_LLVM_IMPLICIT
        blame_info<<"Terminal for if/else if/else case is -- "<<dest->getName().str()<<endl;
#endif
        termBB = dest;
      }
      else if (match) {
#ifdef DEBUG_LLVM_IMPLICIT
        blame_info<<"Block "<<dest->getName().str()<<" is a block to be inserted."<<endl;
#endif
        DomTreeNodeBase<BasicBlock> *dbb = *I;
        blocks.insert(dbb);
      }
    }  
  }
  
  set< DomTreeNodeBase<BasicBlock> *>::iterator set_dbb_i;
  
  for (set_dbb_i = blocks.begin(); set_dbb_i != blocks.end(); set_dbb_i++) {
    DomTreeNodeBase<BasicBlock> *dbb = *set_dbb_i;
    if (br->getCondition()->hasName())
      propagateConditional(DT, dbb, br->getCondition()->getName().data(), termBB);    
    else {
      char tempBuf2[18];
      sprintf(tempBuf2, "0x%x", /*(unsigned)*/br->getCondition());
      string name(tempBuf2);
      
      if (variables.count(name)) {
        NodeProps *vp = variables[name];
        propagateConditional(DT, dbb, vp->name.c_str(), termBB);  
      }              
    }
  }
    
  /*
   for (DomTreeNodeBase<BasicBlock>::const_iterator I = N->begin(), E = N->end(); I != E; ++I)
   {
   BasicBlock * dest = (*I)->getBlock(); 
   int match = 0;
   
   // for all children in CFG
   for (succ_iterator Iter = succ_begin(b), En = succ_end(b); Iter != En; ++Iter) {
   blame_info<<"CFG - "<<Iter->getNameStart()<<" D - "<<dest->getNameStart()<<endl;
   if (strcmp(Iter->getNameStart(), dest->getNameStart()) == 0)
   match++;
   }
   
   //if (v → vchild ∈ D and CFG.v → CFG.vchild ∈ E) 
   if ( match )
   propagateConditional(*I, condName);   
   }
   */
}


bool FunctionBFC::errorRetCheck(User *v)
{
#ifdef DEBUG_ERR_RET
  blame_info<<"in errorRetCheck for "<<v->getName().str()<<endl;
#endif
  Instruction *icmpInst = NULL;
  
  if (isa<Instruction>(v)) {
    icmpInst = dyn_cast<Instruction>(v);
#ifdef DEBUG_ERR_RET
    blame_info<<"ToBool instruction is "<<icmpInst->getName().str()<<" op "
            <<icmpInst->getOpcodeName()<<endl;
#endif 
  }
  else  
    return true;
  
  while (icmpInst->getOpcode() == Instruction::ICmp) {
    Value *zextVal = icmpInst->getOperand(0);   //TC: zextVal should be the keyword: 
    if (isa<Instruction>(zextVal)) {            //condition code, like eq, ugt...why would be ZExt/Load?
      Instruction *zextInst = dyn_cast<Instruction>(zextVal);    
      if (zextInst->getOpcode() == Instruction::ZExt) {
      #ifdef DEBUG_ERR_RET
        blame_info<<"ZEXT inst is "<<zextInst->getName().str()<<endl;
      #endif 
        Value *icmpVal = zextInst->getOperand(0);
        
        if (isa<Instruction>(icmpVal)) 
          icmpInst = dyn_cast<Instruction>(icmpVal);  
        else
          return true;
      }

            else if (zextInst->getOpcode() == Instruction::Load) {
      #ifdef DEBUG_ERR_RET
        blame_info<<"Load inst is "<<zextInst->getName().str()<<endl;
      #endif 
        Value *condLocal = zextInst->getOperand(0); //First operand of Load is the ptr(mem addr)
      #ifdef DEBUG_ERR_RET
          blame_info<<"CondLocal is "<<condLocal->getName().str()<<endl;
      #endif 
        if (condLocal->getName().str().find("ierr") != string::npos) {
                    cout<<"errorRetCheck returns false on "<<zextInst->getName().str()<<endl;
                    return false;
                }
        else
          return true;
      }

      else
        return true;
    }

    else
      return true;
  }
  
  return true;
}


void FunctionBFC::handleAllConditionals(DominatorTreeBase<BasicBlock> *DT, const DomTreeNodeBase<BasicBlock> *N, 
                      LoopInfoBase<BasicBlock,Loop> &LI, set<BasicBlock *> &termBlocks)
{
  BasicBlock *b = N->getBlock();
  
  // Only concerned with branch instructions
  if (isa<BranchInst>(b->getTerminator())) {
#ifdef DEBUG_LLVM_IMPLICIT
    blame_info<<"Branch Instruction found for basic block "<<b->getName().str()<<endl;
#endif
    BranchInst *bBr = cast<BranchInst>(b->getTerminator());  
    
#ifdef DEBUG_LLVM_IMPLICIT
    if (bBr->isConditional())
      blame_info<<"... is conditional"<<endl;
    
    if (!LI.isLoopHeader(b))
      blame_info<<"... not loop header"<<endl;
#endif     
    // Only concerend with conditional non loop header instructions    
    if (bBr->isConditional() && !LI.isLoopHeader(b)) {
      bool eRC = true;  
      if (isa<Instruction>(bBr->getCondition())){  
        Instruction *icmpInst = dyn_cast<Instruction>(bBr->getCondition());
        eRC = errorRetCheck(icmpInst);  
      }      
      #ifdef DEBUG_ERR_RET
      blame_info<<"... is an error checking call."<<endl;
      #endif 
      if (eRC)  
        handleOneConditional(DT, N, bBr);  
    }
  }
    // for all children in dominator tree
    for (DomTreeNodeBase<BasicBlock>::const_iterator I = N->begin(), E = N->end(); I != E; ++I)
        handleAllConditionals(DT, *I, LI, termBlocks); //TC: what's termBlocks for ?    
}


void FunctionBFC::handleLoops(LoopBase<BasicBlock,Loop> *lb)
{
    if (lb == NULL) //added to end the recursion
      return; 
    else
#ifdef DEBUG_LLVM_IMPLICIT
    blame_info<<"Entering handleLoops "<<endl;
#endif 
  
    //short blockCount = 0;
    BranchInst *bBr = NULL;
  // These will be the vertices for each loop that will be blamed
  set<NodeProps *> blamedConditions;
  
  // We get all the exit blocks in the loop using LLVM calls
  SmallVector<BasicBlock *, 8> exitBlocks;
  lb->getExitBlocks(exitBlocks);
  SmallVectorImpl<BasicBlock *>::iterator svimpl_i;
  
#ifdef DEBUG_LLVM_IMPLICIT
  for (svimpl_i = exitBlocks.begin(); svimpl_i != exitBlocks.end(); svimpl_i++) {
    BasicBlock *bb = *svimpl_i;
    blame_info<<"Exit Block is "<<bb->getName().str()<<endl;
  }
#endif 
  
  // We go through every basic block in the loop and find the edgs that
  // go to exit basic blocks, we identify those edges as edges that have
  // conditions that dictate the loop, and the values in the loop are 
  // dependent on them
    for (LoopBase<BasicBlock,Loop>::block_iterator li_bi = lb->block_begin(); 
      li_bi != lb->block_end(); li_bi++) {
      BasicBlock *bb = *li_bi;
    
    if (isa<BranchInst>(bb->getTerminator())) {
    bBr = cast<BranchInst>(bb->getTerminator());
      if (bBr->isConditional()) {
#ifdef DEBUG_LLVM_IMPLICIT
      blame_info<<"Terminator is "<<bBr->getCondition()->getName().str()
                  <<" "<<bBr->getNumOperands()<<endl;
#endif         
      Value *bCondVal = bBr->getCondition();
      
      bool eRC = true;  //QUESTION: NOT SURE ABOUT THE REASON FOR ERC
      
      if (isa<Instruction>(bCondVal)) {  
        Instruction *icmpInst = cast<Instruction>(bCondVal);
        eRC = errorRetCheck(icmpInst);  
      }
        
      if (eRC == false) {
#ifdef DEBUG_ERR_RET
        blame_info<<"Not adding break from loop due to error checking."<<endl;
#endif 
        continue;
      }
        
      for (unsigned i = 0; i < bBr->getNumSuccessors(); i++) {
          BasicBlock *bbSucc = bBr->getSuccessor(i);      
#ifdef DEBUG_LLVM_IMPLICIT
        blame_info<<"Examining successor "<<bbSucc->getName().str()<<endl;
#endif
        for (svimpl_i = exitBlocks.begin(); svimpl_i != exitBlocks.end(); svimpl_i++) {
          BasicBlock *bbMatch = *svimpl_i;      
        if (bbMatch == bbSucc) {
#ifdef DEBUG_LLVM_IMPLICIT
          blame_info<<"Matching Exit Block is "<<bbMatch->getName().str()<<" for "; 
          if (bCondVal->hasName())
          blame_info<<bCondVal->getName().str()<<endl;
          else
          blame_info<<std::hex<</*(unsigned)*/ bCondVal<<std::dec<<endl;
#endif
          if (bCondVal->hasName()) {
          if (variables.count(bCondVal->getName().str()))
              blamedConditions.insert(variables[bCondVal->getName().str()]);
          }
          else {
          char tempBuf2[18];
          sprintf(tempBuf2, "0x%x", /*(unsigned)*/bBr->getCondition());
          string name(tempBuf2);
          if (variables.count(name)) {
            blamedConditions.insert(variables[name]);
          }              
          }// end else for having a name
        }// end if (bbMatch == bbSucc)
        }// end for (exitBlocks)
      }//end successors
      } // end isConditional
    }// end getTerminator()
    }// end basic block iterator 
  

    // We couldn't find any traditional ones, so now we do deeper analysis,
  //  sometimes we have to deal with switch statements
  if (blamedConditions.size() == 0) {
#ifdef DEBUG_LLVM_IMPLICIT
    blame_info<<"...Now looking through switch statements "<<endl;
#endif
    for (LoopBase<BasicBlock,Loop>::block_iterator li_bi = lb->block_begin(); 
         li_bi != lb->block_end(); li_bi++) {
    BasicBlock *bb = *li_bi;
      
    if (blamedConditions.size())
      break;
      
    for (BasicBlock::iterator i = bb->begin(), ie = bb->end(); i != ie; ++i) {
      Instruction *pi = &*i;
      if (pi->getOpcode() == Instruction::Switch) {
      SwitchInst *bBr = dyn_cast<SwitchInst>(pi);
      Value *v = bBr->getCondition();
      for (unsigned i = 0; i < bBr->getNumSuccessors(); i++) {
          BasicBlock *bbSucc = bBr->getSuccessor(i);
#ifdef DEBUG_LLVM_IMPLICIT
        blame_info<<"Examining successor "<<bbSucc->getName().str()<<endl;
#endif 
        for (svimpl_i = exitBlocks.begin(); svimpl_i != exitBlocks.end(); svimpl_i++) {
        BasicBlock * bbMatch = *svimpl_i;  
        if (bbMatch == bbSucc) {
#ifdef DEBUG_LLVM_IMPLICIT
          blame_info<<"Matching Exit Block is "<<bbMatch->getName().str()<<" for ";
#endif
#ifdef DEBUG_LLVM_IMPLICIT
          if (v->hasName())
          blame_info<<v->getName().str()<<endl;
          else
          blame_info<<std::hex<</*(unsigned)*/ v<<std::dec<<endl;
#endif                 
          if (v->hasName()) {
          if (variables.count(v->getName().str()))
              blamedConditions.insert(variables[v->getName().str()]);
          }
          else {
          char tempBuf2[18];
          sprintf(tempBuf2, "0x%x", /*(unsigned)*/bBr->getCondition());
            string name(tempBuf2);
                  
          if (variables.count(name)) {
            blamedConditions.insert(variables[name]);
          }              
          }// end else for having a name
        }// end if (bbMatch == bbSucc)
        }// end for (exitBlocks)
      }//end successors
          }// end isSwitch
    }  // end Instruction iterator
    } // end Basic Block iterator
  } // end if (blamedConditions.size())
  
  // Now we go through all the basic blocks and add the blame nodes discovered from above to the set
    // QUESTION: same blame nodes set for each bb in the loop ??
    for (LoopBase<BasicBlock,Loop>::block_iterator li_bi = lb->block_begin(); 
      li_bi != lb->block_end(); li_bi++) {
    
    set<const char*, ltstr> t; // every bb has a 't'
    BasicBlock *bb = *li_bi;
    ImpRegSet::iterator irs_i; //ptr to each pair: (bb_name, set 't')
    
#ifdef DEBUG_LLVM_IMPLICIT
    blame_info<<"Loop now in basic block "<<bb->getName().str()<<endl;
#endif
  
    if (bb->hasName()) {
    irs_i = iReg.find(bb->getName().str());
    if (irs_i != iReg.end()) {
      t = iReg[bb->getName().str()];
      set<NodeProps *>::iterator set_vp_i, set_vp_e;
      for (set_vp_i = blamedConditions.begin(), set_vp_e = blamedConditions.end(); set_vp_i != set_vp_e; set_vp_i++) {
      NodeProps *vp = *set_vp_i;
#ifdef DEBUG_LLVM_IMPLICIT
      blame_info<<"Implicit -- Inserting "<<vp->name<<" into loop implicit set for bb:"<<bb->getName().str()<<endl;
#endif 
      t.insert(vp->name.c_str());
      }
      iReg[bb->getName().str()] = t;  
        }
    }
  }
  
    for (LoopBase<BasicBlock,Loop>::iterator li_si = lb->begin(); li_si != lb->end(); li_si++) 
    handleLoops(*li_si); //start handling all subloops in this loop
                             //since it's the last call in recursion, it'll terminate anyway
#ifdef DEBUG_LLVM_IMPLICIT
  blame_info<<"Exiting handleLoops "<<endl;
#endif
  
}

void FunctionBFC::generateImplicits()
{
  // Create Dominator Tree for Function
  DominatorTreeBase<BasicBlock> *DT = new DominatorTreeBase<BasicBlock>(false);  
  DT->recalculate(*func);
  
  string dom_path("DOMINATOR/");
  string func_name = func->getName();
  string dot_extension(".dot");
  dom_path += func_name;
  dom_path += dot_extension;
  ofstream dot_file(dom_path.c_str()); //Not used except 'print2' below
  //DT->print2(dot_file, 0);
    
  LoopInfoBase<BasicBlock,Loop> LI;
  //LI.Calculate(*DT);
  LI.analyze(*DT); //llvm 3.3(Analyze) llvm2.5(Calculate)
  for (LoopInfoBase<BasicBlock,Loop>::iterator li_i = LI.begin(); li_i != LI.end(); li_i++)
    handleLoops(*li_i); //typedef typename vector<LoopT *>::const_iterator iterator
  
  set<BasicBlock *> terminalNodes;
  
  handleAllConditionals(DT, DT->getRootNode(), LI, terminalNodes);
    
  delete(DT);
}

// TODO: Make constants a little more intuitive
void FunctionBFC::determineFunctionExitStatus()
{
  bool varLengthPar = varLengthParams();
  // The array will always be at least size 1 to account for return value
  // which always is "parameter 0", if var length, we set it to MAX_PARAMS
  if (varLengthPar) {
    numParams = MAX_PARAMS + 1; //MAX_PARAMS = 128  
    isVarLen = true;
  }
  else {
    numParams = func->arg_size() + 1; //return size_t,which is unsigned in x86
    isVarLen = false;
  }
  
  // Check the return type
  if (func->getReturnType()->isVoidTy()) {
    voidReturn = true;
  }
  else {
    //{realName, ExitType, whichParam, isStructPtr}, changed by Hui from 0 to -1
    ExitVariable *ev = new ExitVariable(string("DEFAULT_RET"), RET, -1, false); 
    addExitVar(ev); //exitVariables.push_back(ev)
  }
  
  //bool isParam = false;
#ifdef DEBUG_LLVM  
  blame_info<<"LLVM__(checkFunctionProto) - Number of args is "<<func->arg_size()<<endl;
#endif
  
  int whichParam = 0;
  
  // Iterates through all parameters for a function 
  for (Function::arg_iterator af_i = func->arg_begin(); af_i != func->arg_end(); af_i++) {
    //whichParam++; //commented out by Hui 03/15/16, now real args of func starts from #0
    Argument *v = dyn_cast<Argument>(af_i); 

    const llvm::Type *argT = v->getType();    
    string argTStr = returnTypeName(argT, string(""));
    int argPtrLevel = pointerLevel(argT, 0); //ptrLevel = 1 means it's a 1-level pointer, like int *ip
    string argName = v->getName().str(); //formal arg has a name for sure
    bool isStructPtr = false;    

#ifdef DEBUG_LLVM  
    blame_info<<"Param# "<<whichParam<<" is "<<argName<<", ptr="<<argPtrLevel<<endl;
#endif
    // EXIT VAR HERE
    // We are only concerned with args that are pointers 
    if (argPtrLevel >= 1 || argTStr.find("Array") != string::npos 
                         || argTStr.find("Struct") != string::npos) {
      numPointerParams++;//we treat Array/Struct args as pointer args as well even if they aren't
      // Dealing with a struct pointer
      if (argPtrLevel >= 1 && argTStr.find("Struct") != string::npos) {
        isStructPtr = true;
      }
      if ((argTStr.find("Array") != string::npos || argTStr.find("Struct") != string::npos)
          && argPtrLevel < 1) {
#ifdef DEBUG_LLVM
        blame_info<<"WEIRD: Non-ptr Struct/Array formal arg: "<<argName<<endl;
#endif
      }
      //This is different against Chapel, for Chapel, since it uses Pid as some parameters, and
      //many times there are no .addr value in the callee as a arg holder, so we treat the formal
      //arg as EV directly, but C/Cuda always has arg holder and to be consistent with the alias
      //analysis with LV, we treat arg holders(.addr) as the EVs, all source-level pointers
      //have to have >=2 ptr level since ptr=1 only represents the value of a pointer, not the
      //pointer variable(source-lvel) itself. At this stage, EV->vertex is unknown
      for (Value::use_iterator ui = v->use_begin(), ue = v->use_end(); ui != ue; ui++) {
        if (Instruction *i = dyn_cast<Instruction>((*ui).getUser())) {
          if (i->getOpcode() == Instruction::Store) { // store arg1 arg1.addr
            Value *second = i->getOperand(1);
            if (second->hasName()) {
              string argHolderName = second->getName().str();
              if (argHolderName == (argName + ".addr")) { //make sure it's the real arg holder
                addExitVar(new ExitVariable(argHolderName, PARAM, whichParam, isStructPtr));
#ifdef DEBUG_LLVM  
                blame_info<<"LLVM_(checkFunctionProto) - (1)Adding exit var "<<argHolderName;
                blame_info<<" Param "<<whichParam<<" in "<<getSourceFuncName()<<endl;
#endif
              }//argHolderName match
            }//argHolder has name
          }//if first use is Store
          //If first use is Bitcast (rarely happen so far)
          else if (i->getOpcode() == Instruction::BitCast || i->getOpcode() == Instruction::AddrSpaceCast) {
            for (Value::use_iterator ui2=i->use_begin(), ue2=i->use_end(); ui2!=ue2; ui2++) {
              if (Instruction *i2 = dyn_cast<Instruction>((*ui2).getUser())) {
                if (i2->getOpcode() == Instruction::Store) { // store arg1 arg1.addr
                  Value *second = i2->getOperand(1);
                  if (second->hasName()) {
                    string argHolderName = second->getName().str();
                    if (argHolderName == (argName + ".addr")) { 
                      addExitVar(new ExitVariable(argHolderName,PARAM,whichParam,isStructPtr));
#ifdef DEBUG_LLVM  
                      blame_info<<"LLVM_(checkFunctionProto) - (2)Adding exit var "
                          <<argHolderName<<" Param "<<whichParam
                          <<" in "<<getSourceFuncName()<<endl;
#endif
#ifdef DEBUG_LLVM  
                    }//argHolderName match
                  }//argHolder has name
                }//if first use is Store
              }
            }
          }//if firt use is bitcast: a = bitcast arg1, t1; store a, arg1.addr;
        }
      }//for all use of this formal arg
#endif
    } //if arg(v) is a pointer or non-ptr array/struct

    whichParam++; //added by Hui 03/15/16, moved from the beginning
  } // end of all args for loop  
  
  if ((numPointerParams == 0 && voidReturn == true) || ft == KERNEL) { //for cuda code
    isBFCPoint = true;  //if the func has 0 pointer param and returns nothing, then it's a blame point ???
#ifdef DEBUG_LLVM    //Yes, since you don't need to go further up
    blame_info<<"IS BP - "<<numPointerParams<<" "<<voidReturn<<" "<<ft<<endl;  
#endif    
  }
}



//TODO:: GlobalVariable
/*
 VoidTyID   0: type with no size
 HalfTyID   1: 16-bit floating point type
 FloatTyID   2: 32 bit floating point type
 DoubleTyID   3: 64 bit floating point type
 X86_FP80TyID   4: 80 bit floating point type (X87)
 FP128TyID   5: 128 bit floating point type (112-bit mantissa)
 PPC_FP128TyID   6: 128 bit floating point type (two 64-bits, PowerPC)
 LabelTyID   7: Labels
 MetadataTyID  8: Metadata
 X86_MMXTyID    9: MMX vectors (64 bts, X86 specific)

 IntegerTyID   10: Arbitrary bit width integers
 FunctionTyID   11: Functions
 StructTyID   12: Structures
 ArrayTyID   13: Arrays
 PointerTyID   14: Pointers
 VectorTyID   15: SIMD 'packed' format, or other vector type
 
 NumTypeIDs     // Must remain as last defined ID
 LastPrimitiveTyID = X86_MMXTyID
 FirstDerivedTyID = IntegerTyID
*/
/* Given a type return the string that describes the type */
string FunctionBFC::returnTypeName(const llvm::Type *t, string prefix)
{
  if (t == NULL)
    return prefix += string("NULL");
  
  unsigned typeVal = t->getTypeID();
  
    if (typeVal == Type::VoidTyID)
        return prefix += string("Void");
    else if (typeVal == Type::HalfTyID)
        return prefix += string("Half");
    else if (typeVal == Type::FloatTyID)
        return prefix += string("Float");
    else if (typeVal == Type::DoubleTyID)
        return prefix += string("Double");
    else if (typeVal == Type::X86_FP80TyID)
        return prefix += string("80 bit FP");
    else if (typeVal == Type::FP128TyID)
        return prefix += string("128 bit FP");
    else if (typeVal == Type::PPC_FP128TyID)
        return prefix += string("2-64 bit FP");
    else if (typeVal == Type::LabelTyID)
        return prefix += string("Label");
    else if (typeVal == Type::MetadataTyID)
        return prefix += string("Metadata");
    else if (typeVal == Type::IntegerTyID)
        return prefix += string("Int");
    else if (typeVal == Type::FunctionTyID)
        return prefix += string("Function");
    else if (typeVal == Type::StructTyID)
        return prefix += string("Struct");
    else if (typeVal == Type::ArrayTyID)
        return prefix += string("Array");
    else if (typeVal == Type::PointerTyID)
        return prefix += returnTypeName(cast<PointerType>(t)->getElementType(),  string("*"));
    else if (typeVal == Type::VectorTyID)
        return prefix += string("Vector");
    else
        return prefix += string("UNKNOWN");
}



void printConstantType(Value * compUnit)
{
  
  if (isa<GlobalValue>(compUnit))
    cout<<"is Global Val";
  else if (isa<ConstantStruct>(compUnit))
    cout<<"is ConstantStruct";
  else if (isa<ConstantPointerNull>(compUnit))
    cout<<"is CPNull";
  else if (isa<ConstantAggregateZero>(compUnit))
    cout<<"is ConstantAggZero";
  else if (isa<ConstantArray>(compUnit))
    cout<<"is C Array";
  else if (isa<ConstantExpr>(compUnit))
    cout<<"is C Expr";
  else if (isa<ConstantFP>(compUnit))
    cout<<"is C FP";
  else if (isa<UndefValue>(compUnit))
    cout<<"is UndefValue";
  else if (isa<ConstantInt>(compUnit))
    cout<<"is C Int";
  else
    cout<<"is ?";
  
  cout<<endl;
  
}


// special function to resolve pidArrays: build new StructBFC, StructFild
void FunctionBFC::pidArrayResolve(Value *v, int fieldNum, NodeProps *fieldVP, int numElems)
{
  const llvm::Type *pointT = v->getType();
  unsigned typeVal = pointT->getTypeID();
  
#ifdef DEBUG_STRUCTS
  blame_info<<"pidArrayResolve Here"<<endl;
#endif
  
    while (typeVal == Type::PointerTyID) {    
    pointT = cast<PointerType>(pointT)->getElementType();
    typeVal = pointT->getTypeID();
  }

    if (typeVal == Type::ArrayTyID) {
      //create the pidArray name for StructBFC with num of elements
      char tempBuf[20];
      sprintf(tempBuf, "PidArray_X%d", numElems);
    string pidArrayName = string(tempBuf);
#ifdef DEBUG_STRUCTS
    blame_info<<"pidArrayName -- "<<pidArrayName<<endl;
#endif
      StructBFC *sb = mb->findOrCreatePidArray(pidArrayName, numElems, pointT);
    if (sb == NULL)
    return;
    
#ifdef DEBUG_STRUCTS
      //Here we only assign sFiled to fieldVP, but not sBFC to v (struct node)
      //we will assign sBFC after this call if sField is found successfully
    blame_info<<"Found sb for "<<pidArrayName<<endl;
#endif
    
    // TODO: Hash
    vector<StructField *>::iterator vec_sf_i;
    for (vec_sf_i = sb->fields.begin(); vec_sf_i != sb->fields.end(); vec_sf_i++){
    StructField *sf = (*vec_sf_i);
    if (sf->fieldNum == fieldNum) {        
#ifdef DEBUG_STRUCTS
      blame_info<<"Assigning fieldVP->sfield: "<<sf->fieldName
              <<" to "<<fieldVP->name<<endl;
#endif
      fieldVP->sField = sf;
    }
    }
  }
}


void FunctionBFC::structResolve(Value *v, int fieldNum, NodeProps *fieldVP)
{
  const llvm::Type *pointT = v->getType();
  unsigned typeVal = pointT->getTypeID();
    while (typeVal == Type::PointerTyID) {    
      pointT = cast<PointerType>(pointT)->getElementType();
    //string origTStr = returnTypeName(pointT, string(" "));
    typeVal = pointT->getTypeID();
  }

    if (typeVal == Type::StructTyID) {
    const llvm::StructType * type = cast<StructType>(pointT);
      if (type->hasName()) {
      string structNameFull = type->getStructName().str();
#ifdef DEBUG_STRUCTS
      blame_info<<"structNameFull -- "<<structNameFull<<endl;
#endif
      if (structNameFull.find("struct.") == 0) { 
        // need to get rid of preceding "struct." and trailing NULL character
        structNameFull = structNameFull.substr(7, structNameFull.length()-7);
      }

        StructBFC *sb = mb->structLookUp(structNameFull);
      if (sb == NULL)
      return;
    
#ifdef DEBUG_STRUCTS
        //Here we only assign sFiled to fieldVP, but not sBFC to v (struct node)
        //we will assign sBFC after this call if sField is found successfully
      blame_info<<"Found sb for "<<structNameFull<<endl;
#endif
    
      // TODO: Hash
      vector<StructField *>::iterator vec_sf_i;
      for (vec_sf_i = sb->fields.begin(); vec_sf_i != sb->fields.end(); vec_sf_i++){
      StructField *sf = (*vec_sf_i);
      if (sf->fieldNum == fieldNum) {        
#ifdef DEBUG_STRUCTS
        blame_info<<"Assigning fieldVP->sfield "<<fieldVP->name<<" to "<<sf->fieldName<<endl;
#endif
        fieldVP->sField = sf;
      }
      }
    }
    }
}


void FunctionBFC::structDump(Value * compUnit)
{
#ifdef DEBUG_STRUCTS
  unsigned numStructElements;
  const llvm::Type * pointT = compUnit->getType();
  unsigned typeVal = pointT->getTypeID();
    //llvm::raw_os_ostream OS(blame_info);
  
    while (typeVal == Type::PointerTyID) {    
    pointT = cast<PointerType>(pointT)->getElementType();
    //string origTStr = returnTypeName(pointT, string(" "));
    typeVal = pointT->getTypeID();
  }
    
    if (typeVal == Type::StructTyID) {
    const llvm::StructType *type = cast<StructType>(pointT);
    numStructElements = cast<StructType>(pointT)->getNumElements();
    blame_info<<"Num of elements "<<numStructElements<<endl;
    //cout<<"TYPE - "<<type->getDescription()<<endl;
        if (type->hasName())
      blame_info<<"TYPE - "<<returnTypeName(type, string(""))<<" "<<type->getName().data()<<endl;
    
    for (int eleBegin = 0, eleEnd = type->getNumElements(); eleBegin != eleEnd; eleBegin++) {
      llvm::Type *elem = type->getElementType(eleBegin);
            //elem->print(OS);
            blame_info<<"typeid: "<<elem->getTypeID()<<endl;
      blame_info<<endl;
      //fprintf(stdout,"dump element: type=%p: ", elem);
      //elem->dump();
    }
  }
#endif
}

/*
string getStringFromMetadata(Value * v)
{
  if (isa<ConstantExpr>(v))
  {
    ConstantExpr * ce = cast<ConstantExpr>(v);
    
    User::op_iterator op_i = ce->op_begin();
    for (; op_i != ce->op_end(); op_i++)
    {
      Value * stringArr = *op_i;      
      
      if (isa<GlobalValue>(stringArr))
      {
        GlobalValue * gv = cast<GlobalValue>(stringArr);    
        User::op_iterator op_i2 = gv->op_begin();
        for (; op_i2 != gv->op_end(); op_i2++)
        {
          Value * stringArr2 = *op_i2;
          
          if (isa<ConstantArray>(stringArr2))
          {
            ConstantArray * ca = cast<ConstantArray>(stringArr2);
            if (ca->isString())
            {
              //cout<<"String name "<<ca->getAsString();
              return ca->getAsString();
            }
          }    
        }
      }
    }
  }
  
  string fail("fail");
  return fail;
}


void FunctionBFC::getPathOrName(Value * v, bool isName)
{
  if (isa<ConstantExpr>(v))
  {
    ConstantExpr * op4ce = cast<ConstantExpr>(v);
    
    User::op_iterator op_i3 = op4ce->op_begin();
    for (; op_i3 != op4ce->op_end(); op_i3++)
    {
      Value * pathArr = *op_i3;      
      //cout<<"Op 4 of type "<<returnTypeName(pathArr->getType(), string(" "));//;<<endl;   
      //cout<<" and kind ";
      //printConstantType(pathArr);
      
      if (isa<GlobalValue>(pathArr))
      {
        GlobalValue * gvOp2 = cast<GlobalValue>(pathArr);    
        User::op_iterator op_i4 = gvOp2->op_begin();
        for (; op_i4 != gvOp2->op_end(); op_i4++)
        {
          //Value * pathArr2 = op4ce->getOperand(1);
          Value * pathArr2 = *op_i4;
          
          if (isa<ConstantArray>(pathArr2))
          {
            ConstantArray * ca = cast<ConstantArray>(pathArr2);
            if (ca->isString())
            {
              if (isName)
              {
                //moduleName = ca->getAsString();
                setModuleName(ca->getAsString());
#ifdef DEBUG_LLVM  
                blame_info<<"Module Name is "<<moduleName<<endl;
#endif
                //cerr<<"Module Name is "<<moduleName<<endl;
              }
              else
              {
                setModulePathName(ca->getAsString());
                //modulePathName = ca->getAsString();
                //cout<<"Path to module is "<<modulePathName<<endl;
              }
              
            }
          }    
        }
      }
    }
  }
}
*/

/*
Descriptor for local variables:
!7 = metadata !{
    i32,      ;; Tag (see below)
    metadata, ;; Context
    metadata, ;; Name
    metadata, ;; Reference to file where defined
    i32,      ;; 24 bit - Line number where defined
              ;; 8 bit - Argument number. 1 indicates 1st argument.
    metadata, ;; Type descriptor
    i32,      ;; flags
    metadata  ;; (optional) Reference to inline location
}
*/

/*
void FunctionBFC::grabModuleNameAndPath(llvm::Value * compUnit)
{
  //blame_info<<"Entering grabModuleNameAndPath"<<endl;
  ConstantExpr * ce = cast<ConstantExpr>(compUnit);
  User::op_iterator op_i = ce->op_begin();
  
  for (; op_i != ce->op_end(); op_i++)
  {
    //blame_info<<"Op ";
    Value * bitCastOp = *op_i;
    
    if (isa<GlobalValue>(bitCastOp))
    {
      //blame_info<<"Global op "<<endl;
      GlobalValue * gvOp = cast<GlobalValue>(bitCastOp);    
      User::op_iterator op_i2 = gvOp->op_begin();
      for (; op_i2 != gvOp->op_end(); op_i2++)
      {
        Value * structOp = *op_i2;      
        if (isa<ConstantStruct>(structOp))
        {
          //  blame_info<<"Struct Op "<<endl;
          ConstantStruct * csOp = cast<ConstantStruct>(structOp);
          Value * op4 = csOp->getOperand(3);
          Value * op5 = csOp->getOperand(4);
          getPathOrName(op4, true);
          getPathOrName(op5, false);
        }
      }
    }
  }
}
*/


void FunctionBFC::printValueIDName(Value *v)
{
  if (isa<Argument>(v))
    blame_info<<" ArgumentVal ";
  else if (isa<BasicBlock>(v))
    blame_info<<" BasicBlockVal ";
  else if (isa<Function>(v))
    blame_info<<" FunctionVal ";
  else if (isa<GlobalAlias>(v))
    blame_info<<" GlobalAliasVal ";
  else if (isa<GlobalVariable>(v))
    blame_info<<" GlobalVariableVal ";
  else if (isa<UndefValue>(v))
    blame_info<<" UndefValueVal ";
  else if (isa<ConstantExpr>(v))
    blame_info<<" ConstantExprVal ";
  else if (isa<ConstantAggregateZero>(v))
    blame_info<<" ConstantAggregateZeroVal ";
  else if (isa<ConstantInt>(v))
    blame_info<<" ConstantIntVal ";
  else if (isa<ConstantFP>(v))
    blame_info<<" ConstantFPVal ";
  else if (isa<ConstantArray>(v))
    blame_info<<" ConstantArrayVal ";
  else if (isa<ConstantStruct>(v))
    blame_info<<" ConstantStructVal ";
  else if (isa<ConstantVector>(v))
    blame_info<<" ConstantVectorVal ";
  else if (isa<ConstantPointerNull>(v))
    blame_info<<" ConstantPointerNullVal ";
  else if (isa<InlineAsm>(v))
    blame_info<<" InlineAsmVal ";
  //else if (isa<PseudoSourceValue>(v)) //treated as a special value in llvm 4.0
  //  blame_info<<" PseudoSourceValueVal ";
    else if (isa<Instruction>(v)) {
      blame_info<<" InstructionVal for Instruction ";
    Instruction * pi = cast<Instruction>(v);
    blame_info<<pi->getOpcodeName()<<" ";
  }
  else
    blame_info<<"UNKNOWNVal";
  
}

string FunctionBFC::calcMetaFuncName(RegHashProps &variables, Value *v, bool isTradName, string nonTradName, int currentLineNum)
{
  
  string newName;
  char tempBuf[1024];
  
  if (isTradName) //isTradName = is traditional name
    sprintf (tempBuf, "%s--%i", v->getName().data(), currentLineNum);  
  else
    sprintf (tempBuf, "%s--%i", nonTradName.c_str(), currentLineNum);  
  
  newName.insert(0, tempBuf);
  
  while (variables.count(newName))
    newName.push_back('a'); //same as append, but with only one char
  
  return newName;
}


void FunctionBFC::printCurrentVariables()
{
#ifdef DEBUG_LLVM_L2
  RegHashProps::iterator begin, end;
  /* Iterate through variables and label them with their node IDs (integers)
   -> first is const char * associated with their name
   -> second is NodeProps * for node
   */
    for (begin = variables.begin(), end = variables.end(); begin != end; begin++)
    {
    string name(begin->first);
    NodeProps * v = begin->second;
    blame_info<<"Node "<<v->number<<" ("<<v->name<<")"<<endl;
  }
#endif 
}


//This function may be needed but we don't use it for now 10/10/17
bool FunctionBFC::firstGEPCheck(User *pi)
{  
    if (isa<ConstantExpr>(pi)) {
    ConstantExpr *ce = cast<ConstantExpr>(pi);
    if (ce->getOpcode() != Instruction::GetElementPtr)
    return true;
  }
  else if (isa<Instruction>(pi)) {
    Instruction * i = cast<Instruction>(pi);
    if (i->getOpcode() != Instruction::GetElementPtr)
    return true;
  }
  else {
#ifdef DEBUG_ERROR
    cerr<<"What are we?!?"<<endl;
#endif 
    exit(0);
  }
  
  Value *v = pi->getOperand(0);
  const llvm::Type *pointT = v->getType();
  unsigned typeVal = pointT->getTypeID();
  
  while (typeVal == Type::PointerTyID) {      
    pointT = cast<PointerType>(pointT)->getElementType();    
    typeVal = pointT->getTypeID();
  }
  
  if (typeVal == Type::StructTyID) {
    const llvm::StructType * type = cast<StructType>(pointT);
      if (type->hasName()) {  
    string structNameFull = type->getName().str();
    
#ifdef DEBUG_LLVM
    blame_info<<"first GEP check -- structNameFull -- "<<structNameFull<<endl;
#endif
    
    if (structNameFull.find("struct.descriptor_dimension") != string::npos)
      return false;
    
    if (structNameFull.find("struct.array1") != string::npos) {
#ifdef DEBUG_LLVM
      blame_info<<"I'm in Hui 1 for "<<structNameFull<<endl;
#endif
      if (pi->getNumOperands() >= 3){
#ifdef DEBUG_LLVM
        blame_info<<"I'm in Hui 2 for "<<structNameFull<<endl;
#endif
        Value *vOp = pi->getOperand(2);
      if (isa<ConstantInt>(vOp)) {
        ConstantInt *cv = cast<ConstantInt>(vOp);
        int number = cv->getSExtValue();
#ifdef DEBUG_LLVM
          blame_info<<"I'm in Hui 3, cv= "<<number<<endl;
#endif
        if (number != 0)
          return false;
      }
      }
    }  
      }//type->hasName
  }
  return true;
}


void FunctionBFC::genDILocationInfo(Instruction *pi, int &currentLineNum, FunctionBFCBB *fbb)
{
    if (MDNode *N = pi->getMetadata("dbg")) {
      DILocation *Loc = dyn_cast<DILocation>(N);
      unsigned Line = Loc->getLine();
      string File = Loc->getFilename().str();
      string Dir = Loc->getDirectory().str();

      currentLineNum = Line;
      //Now this part is unique for cuda, it'll call cuda lib funcs with debug info in their source
      //we need to reset those line numbers since they can mixed up with user source lines  11/07/17
      //Usually these instructions won't appear in the begining of a func, so the moduleName should've been set correctly
      if (moduleSet && File != moduleName)
        currentLineNum = 0;
      //02/02/16: we really shouldn't zero lineNumOrder before every inst
      if (lnm.find(currentLineNum) == lnm.end())
        lnm[currentLineNum] = 0;
      //allLineNums in this function 
      allLineNums.insert(currentLineNum);
      //In FunctionBFCBB
      fbb->lineNumbers.insert(currentLineNum);

      if (currentLineNum < startLineNum && currentLineNum != 0) { 
        //startLineNum: start line of a function, shouldn't include 0 forever
        // The shortest lines should be in entry, in FORTRAN we have cases where
        // there are branches ot pseudo blocks where it includes the definition line
        // of the function which messes up the analysis
        if (fbb->getName().find("entry") != string::npos)
          startLineNum = currentLineNum;
      }

      if (endLineNum < currentLineNum)
        endLineNum = currentLineNum;

      if (!moduleSet) //unlikely happen since it's set before firstPass
        setModuleAndPathNames(File, Dir);
    }
}

bool FunctionBFC::parseDeclareIntrinsic(Instruction *pi, int &currentLineNum, FunctionBFCBB *fbb)
{
    DbgDeclareInst *ddi = dyn_cast<DbgDeclareInst>(pi);
  if (ddi == NULL) {
#ifdef DEBUG_LLVM
      blame_info<<"LLVM__(parseDeclareIntrinsic) "<<pi->getName().str()<<
          " is not llvm.dbg.declare"<<endl;
#endif
      return false;
    }
    
#ifdef DEBUG_P
    blame_info<<"parseDeclareIntrinsic called!"<<endl;
#endif
  DILocalVariable *varDeclare = ddi->getVariable();//Only this work, aboves NOT
    // We don't treat formal argument as local variables
    if (!varDeclare->isParameter()) {
      LocalVar *lv = new LocalVar(); // in FunctionBFC
#ifdef DEBUG_P
      blame_info<<"adding localVar from declare: "<<varDeclare->getName().str()<<endl;
#endif
      lv->definedLine = varDeclare->getLine();
      lv->varName = varDeclare->getName().str();
      localVars.push_back(lv);
    }
    else
      blame_info<<"We met a formal arg: "<<varDeclare->getName().str()<<endl;
  
    return true;
}


void FunctionBFC::ieInvoke(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb) 
{
#ifdef DEBUG_ERROR      
  blame_info<<"Ignoring invoke (ieInvoke)"<<endl;
#endif 
}


NodeProps* FunctionBFC::getNodeBitCasted(Value *val) 
{
  blame_info<<"Getting the original node that's bitcasted from"<<endl;

  NodeProps *node = NULL;
  if (isa<Instruction>(val)) {
    Instruction *bitCastInst = cast<Instruction>(val);
    if (bitCastInst->getOpcode() == Instruction::BitCast ||
        bitCastInst->getOpcode() == Instruction::AddrSpaceCast) {
      Value *bcFrom = bitCastInst->getOperand(0); //get original node from bitcast
      // If the casted value is still a bitcast, recursively call this
      if (Instruction *bi2 = dyn_cast<Instruction>(bcFrom)) {
        if (bi2->getOpcode() == Instruction::BitCast ||
            bi2->getOpcode() == Instruction::AddrSpaceCast) {
          blame_info<<"Start recursively call getNodeBitCasted"<<endl; 
          return getNodeBitCasted(bi2);
        }
      }
      // Otherwise, we return this node
      string bcFromName;
      if (bcFrom->hasName()) // not likely
        bcFromName = bcFrom->getName().str();
      else {
        if (!isa<ConstantInt>(bcFrom) && !isa<ConstantFP>(bcFrom) && 
            !isa<UndefValue>(bcFrom) && !isa<ConstantPointerNull>(bcFrom)) {//v isn't a constant value
          char tempBuf2[18];
          sprintf(tempBuf2, "0x%x", /*(unsigned)*/bcFrom);
          bcFromName = string(tempBuf2);
        }
      }
      //With the name, we can get vp from variables since it's been there
      node = variables[bcFromName];
      if (node != NULL) {
        blame_info<<"We found vp for "<<bcFromName<<endl;
      }//TOCHECK: can we use the node after bitcast when fail ??
      else blame_info<<"Error: we can't find vp for "<<bcFromName<<endl; 
    }
    else blame_info<<"Error: The llvm_inst isn't a bitcast "<<endl;
  }
  else blame_info<<"Error: The llvm_inst isn't an instruction at all"<<endl;

  return node; 
}


void FunctionBFC::ieCallWrapFunc(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  Value *lastOp = pi->getOperand(pi->getNumOperands()-1);
  blame_info<<"Entering ieCallWrapFunc for "<<lastOp->getName().str()<<endl;
  //First, we still need to parse executeON for fid and arg value
  int fidHolder = -1;
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
      blame_info<<"Error: params for "<<realName<<" are incomplete: param#"<<i<<endl;
      complete = false;
    }
  }

  //Now we can construct function call for on/coforall_fn_chpl*
  //name for each callnode: bar--51a, bar--51aa.. 
  //basically same as in calcMetaFuncName, just we don't need lastOp here
  string mangledCallName;
  char tempBuf[1024];
  sprintf(tempBuf, "%s--%i", realName.c_str(), currentLineNum);
  mangledCallName.insert(0, tempBuf);
  while (variables.count(mangledCallName))
    mangledCallName.push_back('a');
  blame_info<<"MangledCallName is: "<<mangledCallName<<endl;

  //Now we need to create the node for wrap*_fn_chpl* function call
  if (variables.count(mangledCallName) == 0) { 
#ifdef DEBUG_VP_CREATE
  blame_info<<"Adding NodeProps(ieCallWrapFunc) for "<<mangledCallName<<endl;
#endif
  NodeProps *vp = new NodeProps(varCount,mangledCallName,currentLineNum,pi);
  vp->fbb = fbb;
      
  if (currentLineNum != 0) {
    int lnm_cln = lnm[currentLineNum];
    vp->lineNumOrder = lnm_cln;
    lnm_cln++;
    lnm[currentLineNum] = lnm_cln;
    }
      
  variables[mangledCallName] = vp;
  varCount++;
    // we need a funcCall instantiation for the call node      
    FuncCall *fp = new FuncCall(-2, mangledCallName); //CHANGEd: -1 =>-2
  fp->lineNum = currentLineNum;
  vp->addFuncCall(fp);
  addFuncCalls(fp);
  vp->nStatus[CALL_NODE] = true; //CALL_NODE = 16, NODE_PROPS_SIZE = 20
      
    //Since wrap* func returns void,We don't need to worry about the retval
  }    
  else {
    blame_info<<"Error: how could "<<mangledCallName<<" exist before"<<endl;
    return;
  }

  //Now we add funcCalls for all params, they all should be in variables already
  // We have a funcCall instantiation for each arg
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
      //With the name, we can get vp from variables since it's been there
      NodeProps *paramNode = variables[paramName];
      if (paramNode) {
        FuncCall *fp = new FuncCall(i, mangledCallName); //only one param: arg
        fp->lineNum = currentLineNum;
        paramNode->addFuncCall(fp);
        addFuncCalls(fp);
        paramNode->nStatus[CALL_PARAM] = true;
        
        blame_info<<"Adding FuncCall for param#"<<i<<" name: "<<paramName<<endl;
      }
      else 
        blame_info<<"Error: can't find node for param#"<<i<<" name: "<<paramName<<endl;
    }
  }

  //delete the params' holder
  delete[] params;
}


int FunctionBFC::paramTypeMatch(const llvm::Type *t1, const llvm::Type *t2) 
{
  int ptrL1 = 0, ptrL2 = 0;
  string tName1, tName2;
  const llvm::Type *tReal1, *tReal2;

  // Get t1, t2 info
  tReal1 = t1;
  while (tReal1->getTypeID() == Type::PointerTyID) {
    ptrL1 ++;
    tReal1 = cast<PointerType>(tReal1)->getElementType();
  }
  tReal2 = t2;
  while (tReal2->getTypeID() == Type::PointerTyID) {
    ptrL2 ++;
    tReal2 = cast<PointerType>(tReal2)->getElementType();
  }

  // If both are struct, we need a further check on names
  if (tReal1->getTypeID()==Type::StructTyID && tReal2->getTypeID()==Type::StructTyID) {
    const llvm::StructType *t1 = cast<StructType>(tReal1);
    const llvm::StructType *t2 = cast<StructType>(tReal2);
    if (t1->hasName() && t2->hasName()) {
      tName1 = cast<StructType>(tReal1)->getName().str();
      tName2 = cast<StructType>(tReal2)->getName().str();
      if (tName1 == tName2)
        return (ptrL1 - ptrL2);
      else
        return 99; 
    }
    else
      return 99;
  }
  // If typeId are same but they are not struct, simply return the ptr diff
  else if (tReal1->getTypeID() == tReal2->getTypeID()) 
    return (ptrL1 - ptrL2);
  // If typeID are different, they are totally different
  else
    return 99;
}


void FunctionBFC::getParamsForCoforall(Instruction *pi, Value **params, int numArgs, vector<FuncFormalArg*> &args) 
{
  // pi->taskListAddCoStmt
  Value *p2 = get_args_for(pi);
  if (p2->hasName()) { // it should have _args_forcoforall_fn_chpl
    if (p2->getName().str().find("_args_forcoforall_fn_chpl") == string::npos) 
      blame_info<<"Check what's the name for coforall: "<<p2->getName().str()<<endl;
    // we check all use of _args*:  p3 = load _args*
    for (Value::use_iterator u_i=p2->use_begin(), u_e=p2->use_end(); u_i!=u_e; u_i++) {
      Value *p3 = (*u_i).getUser();
      if (Instruction *inst3 = dyn_cast<Instruction>(p3)) {
        if (inst3->getOpcode() == Instruction::Load) {
          // now we check use of p3: inst3 = GEP inst2, 0,..   
          for (Value::use_iterator u_i2=p3->use_begin(), u_e2=p3->use_end(); u_i2!=u_e2; u_i2++) {
            Value *p4 = (*u_i2).getUser();
            if (Instruction *inst4 = dyn_cast<Instruction>(p4)) {
              if (inst4->getOpcode() == Instruction::GetElementPtr) {
                // get the param index
                Value *paramIdx = inst4->getOperand(2); //GEP a, 0, 8..
                int whichParam; 
                if (isa<ConstantInt>(paramIdx)) {
                  ConstantInt *paramIdxVal = cast<ConstantInt>(paramIdx);
                  whichParam= (int)(paramIdxVal->getZExtValue()) - 1; //coforall_fn param starts from GEP x,0,1
                }
                //Check if it's within the range, coforall param starts from GEP x,0,0
                if (whichParam < numArgs && whichParam >= 0) {
                  if (params[whichParam] == NULL) {
                    int typeMatchResult = paramTypeMatch(p4->getType(), args[whichParam]->argType);
                    // Total match, simply put in p4
                    if (typeMatchResult == 0)
                      params[whichParam] = p4;
                    // p4 is *arg, should put in p4->storesTo instead
                    else if (typeMatchResult == 1) {
                      for (Value::use_iterator u_i3=p4->use_begin(), u_e3=p4->use_end(); u_i3!=u_e3; u_i3++) {
                        Value *p5 = (*u_i3).getUser();
                        if (Instruction *inst5 = dyn_cast<Instruction>(p5)) {
                          if (inst5->getOpcode() == Instruction::Store) {
                            Value *p6 = inst5->getOperand(0); // store p6, p4
                            params[whichParam] = p6;
                          }
                        }
                      }
                    }
                    // Total Unmatch 
                    else if (typeMatchResult == 99)
                      blame_info<<"Check: param type Unmatch to arg: "<<whichParam<<endl;
                    // Otherwise, weird !
                    else
                      blame_info<<"Weird ! tMR="<<typeMatchResult<<" at param="<<whichParam<<endl;
                  }
                  else
                    blame_info<<"Check: old param existed: "<<whichParam<<endl;
                }
                // whichParam isn't within the range
                else
                  blame_info<<"Check: whichParam is out of the range ! p="<<whichParam<<endl;
              } //if inst4 is GEP
            } //if p4 is inst
          } //all uses of GEP 
        } //isnt 3 is Load
      } //p3 is inst
    } //all uses of _args*
  } //p2 has name: _args*
}


//Helper func for getParamsFor*
Value* FunctionBFC::get_args_for(Instruction *pi)
{
  //pi->executeOn*
  //Chapel 1.15 has additional bitcasts for new "_args_vforon_fn_chpl", so we started from p_2 (means -2)
  Value *p0 = pi->getOperand(2); //Naming pattern: p#<==>inst# same thing,same#
  if (Instruction *inst0 = dyn_cast<Instruction>(p0)) { 
    if (inst0->getOpcode() == Instruction::BitCast ||
        inst0->getOpcode() == Instruction::AddrSpaceCast) {
      Value *p1 = inst0->getOperand(0);
      if (Instruction *inst1 = dyn_cast<Instruction>(p1)) {
        if (inst1->getOpcode() == Instruction::Load) {
          Value *p2 = inst1->getOperand(0);
          if (p2->hasName()) { //p2 should be _args_vforon_fn_chpl
            for (Value::use_iterator u_i=p2->use_begin(), u_e=p2->use_end(); u_i!=u_e; u_i++) {
              Value *p3 = (*u_i).getUser(); //p3 is the inst pointing to "store %1, _args_vforon_"
              if (Instruction *inst3 = dyn_cast<Instruction>(p3)) {
                if (inst3->getOpcode() == Instruction::Store) {
                  Value *p4 = inst3->getOperand(0); //p4 is %1
                  if (Instruction *inst4 = dyn_cast<Instruction>(p4)) {
                    if (inst4->getOpcode() == Instruction::BitCast ||
                        inst4->getOpcode() == Instruction::AddrSpaceCast) {
                      Value *p5 = inst4->getOperand(0);
                      if (Instruction *inst5 = dyn_cast<Instruction>(p5)) {
                        if (inst5->getOpcode() == Instruction::Load) {
                          Value *argVal = inst5->getOperand(0);
                          //argVal should be _args_foron(coforall)_fn_chpl, we don't need to
                          //check its name here since getParamsForOn(Coforall) will check it later
                          if (argVal != NULL) {
                            blame_info<<"we've found the _args_for"<<endl;
                            return argVal;
                          }
                        }
                        else blame_info<<"inst5 is not load"<<endl;
                      }
                      else blame_info<<"inst5 is not inst"<<endl;
                    }
                    else blame_info<<"inst4 is not bitcast"<<endl;
                  }
                  else blame_info<<"inst4 is not inst"<<endl;
                }
                else blame_info<<"inst3 is not store"<<endl;
              }
              else blame_info<<"inst3 is not inst"<<endl;
            }//end of for loop
          }
          else blame_info<<"p2 does not have name"<<endl;
        }
        else blame_info<<"inst1 is not load"<<endl;
      }
      else blame_info<<"inst1 is not inst"<<endl;
    }
    else blame_info<<"inst0 is not bitcast"<<endl;
  }
  else blame_info<<"inst0 is not inst"<<endl;

  blame_info<<"get_args_for failed"<<endl;
  return NULL;
}
 

//logic changed for Chapel 1.15
void FunctionBFC::getParamsForOn(Instruction *pi, Value **params, int numArgs, vector<FuncFormalArg*> &args) 
{
  //pi->executeOn*
  Value *p2 = get_args_for(pi);
  if (p2->hasName()) { // it should have _args_foron_fn_chpl
    if (p2->getName().str().find("_args_foron_fn_chpl") == string::npos) 
      blame_info<<"Check what's the name for on: "<<p2->getName().str()<<endl;
    // we check all use of _args*:  p3 = load _args*
    for (Value::use_iterator u_i=p2->use_begin(), u_e=p2->use_end(); u_i!=u_e; u_i++) {
      Value *p3 = (*u_i).getUser();
      if (Instruction *inst3 = dyn_cast<Instruction>(p3)) {
        if (inst3->getOpcode() == Instruction::Load) {
          // now we check use of p3: inst3 = GEP inst2, 0,..   
          for (Value::use_iterator u_i2=p3->use_begin(), u_e2=p3->use_end(); u_i2!=u_e2; u_i2++) {
            Value *p4 = (*u_i2).getUser();
            if (Instruction *inst4 = dyn_cast<Instruction>(p4)) {
              if (inst4->getOpcode() == Instruction::GetElementPtr) {
                // get the param index
                Value *paramIdx = inst4->getOperand(2); //GEP a, 0, 8..
                int whichParam; //starts from 1, should ignore 0
                if (isa<ConstantInt>(paramIdx)) {
                  ConstantInt *paramIdxVal = cast<ConstantInt>(paramIdx);
                  whichParam= (int)(paramIdxVal->getZExtValue()) - 2; //on_fn param starts from GEP x,0,2
                }
                else {
                  blame_info<<"Bad param for on, Check!"<<endl;
                  continue;
                }
                //Check if it's within the range
                if (whichParam < numArgs && whichParam >= 0) {
                  if (params[whichParam] == NULL) {
                    int typeMatchResult = paramTypeMatch(p4->getType(), args[whichParam]->argType);
                    // Total match, simply put in p4 : %1 = load val; store %1, p4
                    if (typeMatchResult == 0)
                      params[whichParam] = p4; //TOCHECK:Should we put in p4 or val??
                    // p4 is *arg, should put in p4->storesTo instead
                    else if (typeMatchResult == 1) {
                      for (Value::use_iterator u_i3=p4->use_begin(), u_e3=p4->use_end(); u_i3!=u_e3; u_i3++) {
                        Value *p5 = (*u_i3).getUser();
                        if (Instruction *inst5 = dyn_cast<Instruction>(p5)) {
                          if (inst5->getOpcode() == Instruction::Store) {
                            Value *p6 = inst5->getOperand(0); // store p6, p4
                            params[whichParam] = p6;
                          }
                        }
                      }
                    }
                    // Total Unmatch 
                    else if (typeMatchResult == 99)
                      blame_info<<"Check2: param type Unmatch to arg: "<<whichParam<<endl;
                    // Otherwise, weird !
                    else
                      blame_info<<"Weird2! tMR="<<typeMatchResult<<" at param="<<whichParam<<endl;
                  }
                  else
                    blame_info<<"Check2: old param existed: "<<whichParam<<endl;
                }
                // whichParam isn't within the range
                else
                  blame_info<<"Check2: whichParam is out of the range ! p="<<whichParam<<endl;
              } //if inst4 is GEP
            } //if p4 is inst
          } //all uses of GEP 
        } //isnt 3 is Load
      } //p3 is inst
    } //all uses of _args*
  } //p2 has name: _args*
}


void FunctionBFC::ieCall(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  // pi->getName = return value
  // op[#ops-1] = name of call (the last operand of pi)
  // op[0,1,...] = parameters
#ifdef DEBUG_LLVM
  blame_info<<"LLVM__(examineInstruction)(Call) -- pi "<<pi->getName().str()<<" "<<pi<<" "<<pi->getOpcodeName()<<endl;
#endif 
  // Add the Node Props for the return value
  if (pi->hasName() && variables.count(pi->getName().str()) == 0) {
    string name = pi->getName().str();
#ifdef DEBUG_VP_CREATE
    blame_info<<"Adding NodeProps(12) for "<<name<<endl;
#endif 
    NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
    vp->fbb = fbb;
    
    if (currentLineNum != 0) {
    int lnm_cln = lnm[currentLineNum];
    vp->lineNumOrder = lnm_cln;
    lnm_cln++;
    lnm[currentLineNum] = lnm_cln;
    }
  
    variables[pi->getName().str()] = vp;
    varCount++;          
  }
  
  int opNum = 0;
    int callNameIdx = pi->getNumOperands()-1; //called func is the last operand of this callInst
  bool isTradName = true; //is traditional name
  string nonTradName; //represents embedded func names 
  
    /////added by Hui,trying to get the called function///////////
    llvm::CallInst *cpi = cast<CallInst>(pi);
    llvm::Function *calledFunc = cpi->getCalledFunction();
    if (calledFunc != NULL && calledFunc->hasName())
      blame_info<<"In ieCall, calledFunc's name = "<<calledFunc->getName().data();
    blame_info<<"  pi->getNumOperands()="<<pi->getNumOperands()<<endl;
    //////////////////////////////////////////////////////////
    //KEEP it here for now, may need to be replaced with CUDA-specific calls
    Value *lastOp = pi->getOperand(callNameIdx);
    if (lastOp->hasName()) {
      string fn = lastOp->getName().str();
      //save the original function names
      funcCallNames.insert(lastOp->getName().data()); 
      blame_info<<"Called function has a name: "<<fn<<endl;
      //modified by Hui 01/23/18: we treat llvm.nvvm intrinsics from ptx math library 
      //not as normal func calls but some operations such as add, sub...
#ifdef SPECIAL_FUNC_PTR 
      if (fn.find("llvm.nvvm") == 0) {
        if (fn.find("llvm.nvvm.read") != string::npos) {
          blame_info<<"Not deal with special ptx registers"<<endl;
          return;
        }

        if (pi->hasNUsesOrMore(1)) {
          ieDefaultPTXIntrinsic(pi, varCount, currentLineNum, fbb);
        }
        else {
          blame_info<<fn<<" has no uses"<<endl;
        }
        return; //IMPORTANT ! We should give up parsing this instruction completely
      }
      //not process other llvm instrinsics
      else if (fn.find("llvm.dbg") != string::npos) {
        blame_info<<"Not deal with intrinsic dbg func calls"<<endl;
        return;
      }
#endif
    }
    else { //if no name, then it's an embedded func
      if (isa<ConstantExpr>(lastOp)) {
#ifdef DEBUG_LLVM
        blame_info<<"Called function is ConstantExpr"<<endl;
#endif
        ConstantExpr *ce = cast<ConstantExpr>(lastOp);
        User::op_iterator op_i = ce->op_begin();
        for (; op_i != ce->op_end(); op_i++) {
          Value *funcVal = op_i->get();      
          if (isa<Function>(funcVal)) {
            Function * embFunc = cast<Function>(funcVal);
#ifdef DEBUG_LLVM
            blame_info<<"EMB Func "<<embFunc->getName().str()<<endl;
#endif 
            isTradName = false;
            nonTradName = embFunc->getName().str();
            funcCallNames.insert(embFunc->getName().data());
          }
        }
      }
      else { 
        //we currently not deal with func calls without name 
        //and not a CE either, e.g., "asm"
#ifdef DEBUG_LLVM
        blame_info<<"Called func has NO name and NOT a CE"<<endl;
#endif
        return;
      }
    }
 
    //name for each callnode: bar--51a, bar--51aa..
    string callName = calcMetaFuncName(variables, lastOp, isTradName, nonTradName, currentLineNum);
    blame_info<<"After calcMetaFuncName, callName="<<callName<<endl;
  
    // Assigning function name VP and VPs for all the parameters
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
      if (!(v->hasName())) {
#ifdef DEBUG_LLVM
    blame_info<<"In ieCall -- Call Operand No Name "<<v<<" ";
    printValueIDName(v);
    blame_info<<endl;
#endif      
      }
      else { //v has a name
#ifdef DEBUG_LLVM
    blame_info<<"In ieCall -- Call Operand "<<opNum<<" "<<v->getName().str()<<endl;
#endif
    }
      // The parameter is a BitCast or GEP operation pointing to something else
    if (isa<ConstantExpr>(v)) {
      ConstantExpr *ce = cast<ConstantExpr>(v);
      if (ce->getOpcode() == Instruction::GetElementPtr) {
        // Overwriting 
        v = ce->getOperand(0);//get the real one 
#ifdef DEBUG_LLVM
        blame_info<<"Overwrite(1) Call Param GEP: "<<getNameForVal(v)<<endl;
#endif 
      }  
      else if (ce->getOpcode() == Instruction::BitCast ||
               ce->getOpcode() == Instruction::AddrSpaceCast) {
        v = ce->getOperand(0);//get the real one
#ifdef DEBUG_LLVM
        blame_info<<"Overwrite(2) Call Param Bitcast: "<<getNameForVal(v)<<endl;
#endif 
      }
    }
    
    // We add the VP for the actual call
    if (variables.count(callName) == 0 && opNum == callNameIdx) { 
#ifdef DEBUG_VP_CREATE
    blame_info<<"Adding NodeProps(13) for "<<callName<<endl;
#endif
    NodeProps *vp = new NodeProps(varCount,callName,currentLineNum,pi);
    vp->fbb = fbb;
      
    if (currentLineNum != 0) {
      int lnm_cln = lnm[currentLineNum];
      vp->lineNumOrder = lnm_cln;
      lnm_cln++;
      lnm[currentLineNum] = lnm_cln;
    }
      
    variables[callName] = vp;
    varCount++;
    //printCurrentVariables();
      
    // OpNum -2 signifies an entry for the name of the function call
    FuncCall *fp = new FuncCall(-2, callName); //CHANGEd: -1 =>-2
    fp->lineNum = currentLineNum;
    vp->addFuncCall(fp);
    addFuncCalls(fp);
    vp->nStatus[CALL_NODE] = true; //CALL_NODE = 16, NODE_PROPS_SIZE = 20
      
        //We need to assign a funcCall object to the return and treat it as "parameter"-1
    if (pi->hasName()) { //was added at the begining of ieCall
      if (variables.count(pi->getName().str())) {
      NodeProps *retVP = variables[pi->getName().str()];
          
      if (retVP) {
       // Adding a funcCall object for the return value ("parameter" 0)
          FuncCall *fp = new FuncCall(-1, callName);//CHANGEd 0=>-1
        fp->lineNum = currentLineNum;
        retVP->addFuncCall(fp);
        addFuncCalls(fp);
        retVP->nStatus[CALL_RETURN] = true; //CALL_RETURN = 18
      }
      }
    }
    // TODO: Put this check in everywhere
    else if (pi->hasNUsesOrMore(1)) { //the return is used >=1 places
#ifdef DEBUG_LLVM
      blame_info<<"Call "<<callName<<
              " has at least one use. Return value valid."<<endl;
#endif         
      char tempBuf2[18];
      sprintf(tempBuf2, "0x%x", /*(unsigned)*/pi);
      string name(tempBuf2);
      
      NodeProps *retVP;  
      if (variables.count(name) == 0) { //most likely
#ifdef DEBUG_VP_CREATE
      blame_info<<"Adding NodeProps(F5) for "<<name<<endl;
#endif
      retVP = new NodeProps(varCount,name,currentLineNum,pi);
      retVP->fbb = fbb;
          
      if (currentLineNum != 0) {
        int lnm_cln = lnm[currentLineNum];
        retVP->lineNumOrder = lnm_cln;
        lnm_cln++;
        lnm[currentLineNum] = lnm_cln;
      }
      
      variables[name] = retVP;
      varCount++;
          }//if not added into variables yet
          else //already added before, rarely happen
            retVP = variables[name];
          
      FuncCall *fp = new FuncCall(-1, callName); //CHANGEd 0=>-1
      fp->lineNum = currentLineNum;
      retVP->addFuncCall(fp);
      addFuncCalls(fp);
      retVP->nStatus[CALL_RETURN] = true;   
    }

    else {
#ifdef DEBUG_LLVM
      blame_info<<"Call "<<callName<<" has at NO users.  Return value not valid."<<endl;
#endif 
    }
    }
    
      else if (opNum != callNameIdx) {// We take care of the parameters
      // opNum is the parameter number to call, callName is as advertised
    FuncCall *fp = new FuncCall(opNum, callName);
    fp->lineNum = currentLineNum;
#ifdef DEBUG_LLVM
    blame_info<<"Adding func call in "<<getSourceFuncName()<<" to "<<callName<<" p "<<opNum<<" for node ";
    blame_info<<v->getName().str()<<"("<<std::hex<<v<<std::dec<<")"<<endl;
#endif
    if (v->hasName()) {
      NodeProps *vp;
      if (variables.count(v->getName().str())) {
        // Param might already exist ... in fact, it's likely since they are also local vars
      vp = variables[v->getName().str()];
      // If param doesn't exist, we add it to list of variables in the program
      if (!vp) {        
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(14) for "<<v->getName().str()<<endl;
#endif          
        vp = new NodeProps(varCount, v->getName().str(), currentLineNum, pi);
        vp->fbb = fbb;
            
        if (currentLineNum != 0) {
        int lnm_cln = lnm[currentLineNum];
        vp->lineNumOrder = lnm_cln;
        lnm_cln++;
        lnm[currentLineNum] = lnm_cln;
        }
        variables[v->getName().str()] = vp;
        varCount++;            
      }
      }
          else { //Param didn't exist
#ifdef DEBUG_VP_CREATE
      blame_info<<"Adding NodeProps(14a) for "<<v->getName().str()<<endl;
#endif
      vp = new NodeProps(varCount, v->getName().str(), currentLineNum, pi);
      vp->fbb = fbb;
          
      if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
        vp->lineNumOrder = lnm_cln;
        lnm_cln++;
        lnm[currentLineNum] = lnm_cln;
      }
      variables[v->getName().str()] = vp;
      varCount++;
      //printCurrentVariables();
      }
      // We have a funcCall instantiation for every parameter
      vp->addFuncCall(fp);
      addFuncCalls(fp);
      vp->nStatus[CALL_PARAM] = true;
    }
        //param has no name   
    else { 
          if (!isa<ConstantInt>(v) && !isa<ConstantFP>(v) && 
              !isa<UndefValue>(v) && !isa<ConstantPointerNull>(v)) {//v isn't a constant value
      char tempBuf2[18];
      sprintf(tempBuf2, "0x%x", /*(unsigned)*/v);
      string name(tempBuf2);
      
      NodeProps *vp;
      if (variables.count(name) == 0){
#ifdef DEBUG_VP_CREATE
          blame_info<<"Adding NodeProps(F2) for "<<name<<endl;
#endif
        vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;  
        if (currentLineNum != 0) {
        int lnm_cln = lnm[currentLineNum];
        vp->lineNumOrder = lnm_cln;
        lnm_cln++;
        lnm[currentLineNum] = lnm_cln;
        }
          
        variables[name] = vp;
        varCount++;
        //printCurrentVariables();
      }
      else {
        vp = variables[name];
      }
        
      vp->addFuncCall(fp);
      addFuncCalls(fp);
      vp->nStatus[CALL_PARAM] = true;
      }
        }
    }

    opNum++;
  }
}


void FunctionBFC::ieGen_LHS_Alloca(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  string name;
  // Add LHS variable to list of symbols
  if (pi->hasName())
    name = pi->getName().str();
  else {
    char tempBuf[18];
    sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
    name.insert(0,tempBuf);
  }
  
#ifdef DEBUG_VP_CREATE
  blame_info<<"Adding NodeProps(A1) for "<<name<<endl;
#endif
  NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
  vp->fbb = fbb;
  
  if (currentLineNum != 0) {
      int lnm_cln = lnm[currentLineNum];
    vp->lineNumOrder = lnm_cln;
    lnm_cln++;
    lnm[currentLineNum] = lnm_cln;
  }
  
  if (variables.count(name) == 0) {
    variables[name] = vp;
    varCount++;
  }
  
  //if (name.find(".") != string::npos || name.find("0x") != string::npos)  
    //"." refers to some struct field like %a.tmp.field
    //TC: DO we need to treat the arg holder as local vars for CUDA? 10/11/17
  if ((name.find(".")!=string::npos && name.find(".addr")==string::npos) 
                                        || name.find("0x")!=string::npos) {
      LocalVar *lv = new LocalVar();
    lv->definedLine = currentLineNum;
    lv->varName = name;
#ifdef DEBUG_P
      blame_info<<"Adding lV from alloca: "<<name<<endl;
#endif
    localVars.push_back(lv);
  }
}

void FunctionBFC::ieGen_LHS(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb) 
{        
  //added by Hui 05/10/16: to get the opCode
  Instruction *pii;
  int opCode;
  NodeProps *vp;

  if (isa<Instruction>(pi)) {
    pii = dyn_cast<Instruction>(pi);
    opCode = pii->getOpcode();
  }

  // Add LHS variable to list of symbols
  if (pi->hasName() && variables.count(pi->getName().str()) == 0) {
    string name = pi->getName().str();
    
#ifdef DEBUG_VP_CREATE
    blame_info<<"Adding NodeProps(1) for "<<name<<" currentLineNum="<<currentLineNum<<" lnm="<<lnm[currentLineNum]<<endl;
#endif 
    if (isa<ConstantExpr>(pi)) {
#ifdef DEBUG_VP_CREATE
      blame_info<<"Weird: pi "<<name<<" has name but is ConstantExpr"<<endl;
#endif 
    }  

    vp = new NodeProps(varCount, name, currentLineNum, pi);
    vp->fbb = fbb;
    
    bool lnmChanged = false; //added by Hui 05/10/16
    if (currentLineNum != 0) {
      int lnm_cln = lnm[currentLineNum];
      vp->lineNumOrder = lnm_cln;
      lnm_cln++;
      lnm[currentLineNum] = lnm_cln;
      lnmChanged = true;  //added by Hui 05/10/16
    }
    
    variables[name] = vp;
    varCount++;        
    //printCurrentVariables();
    //added by Hui 05/10/16: for loadLineNumOrder
    if (opCode == Instruction::Load) {
      if (lnmChanged)
        (vp->loadLineNumOrder)[currentLineNum]=lnm[currentLineNum]-1;
      else
        (vp->loadLineNumOrder)[currentLineNum]=lnm[currentLineNum];
    }       
  }
  
  else { //if the instruction has NO name, then it's a register, because LLVM use SSA
           //so it's NOT in variables for sure
    char tempBuf[18];
    sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
    string name(tempBuf); // Use the address of the instruction as its name ??
    
    if (isa<ConstantExpr>(pi)) {
      name += ".CE";
      
      char tempBuf2[10];
      sprintf(tempBuf2, ".%d", currentLineNum);
      name.append(tempBuf2);
#ifdef DEBUG_VP_CREATE
      blame_info<<"Adding NodeProps(F1CE) for "<<name<<" currentLineNum="<<currentLineNum<<" lnm="<<lnm[currentLineNum]<<endl;
#endif
      vp = new NodeProps(varCount,name,currentLineNum,pi);
    }
    else {
#ifdef DEBUG_VP_CREATE
      blame_info<<"Adding NodeProps(F1) for "<<name<<" currentLineNum="<<currentLineNum<<" lnm="<<lnm[currentLineNum]<<endl;
#endif
      vp = new NodeProps(varCount,name,currentLineNum,pi);
    }
    
    vp->fbb = fbb;
    bool lnmChanged = false; //added by hui 05/10/16
    if (currentLineNum != 0) {
      int lnm_cln = lnm[currentLineNum];
      vp->lineNumOrder = lnm_cln;
      lnm_cln++;
      lnm[currentLineNum] = lnm_cln;
      lnmChanged = true;
    }

    if (variables.count(name) == 0) {
      variables[name] = vp;
      varCount++;
#ifdef DEBUG_VP_CREATE
      blame_info<<"New added variable:  "<<name<<endl;
#endif 
    }
    else {
#ifdef DEBUG_VP_CREATE
      blame_info<<"Attention! there was an old VP: "<<name<<endl;
#endif
    }

    //added by Hui 05/10/16: for loadLineNumOrder
    if (opCode == Instruction::Load) {
      if (lnmChanged)
        (vp->loadLineNumOrder)[currentLineNum]=lnm[currentLineNum]-1;
      else
        (vp->loadLineNumOrder)[currentLineNum]=lnm[currentLineNum];
    }       
  }
}


void FunctionBFC::ieGen_OperandsPTXIntrinsic(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  int opNum = 0, callNameIdx = pi->getNumOperands()-1;
  // Add operands to list of symbols
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
    if (opNum  == callNameIdx)//we don't deal with the call node for ptx intrinsics
        continue;
#ifdef DEBUG_LLVM
    if (!(v->hasName())) {
    blame_info<<"Standard Operand No Name "<<" "<<v<<" ";
    printValueIDName(v);
    blame_info<<endl;
    }
#endif      
  
      if (v->hasName() && !isa<BasicBlock>(v)) {
    if (variables.count(v->getName().str()) == 0) {
        string name = v->getName().str();
#ifdef DEBUG_VP_CREATE
      blame_info<<"Adding NodeProps(2-Operands) for "<<name<<endl;
#endif 
      NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
      vp->fbb = fbb;
        
      if (currentLineNum != 0) {
      int lnm_cln = lnm[currentLineNum];
      vp->lineNumOrder = lnm_cln;
      lnm_cln++;
      lnm[currentLineNum] = lnm_cln;
      }

      variables[v->getName().str()] = vp;          
      varCount++;
      //printCurrentVariables();
    }
    }
    else if(isa<ConstantExpr>(v)) { //v has no name but is a constant expression
#ifdef DEBUG_LLVM
      blame_info<<"Value is ConstantExpr"<<endl;
#endif 
      ConstantExpr *ce = cast<ConstantExpr>(v);  
      createNPFromConstantExpr(ce, varCount, currentLineNum, fbb); 
    }

    opNum++;
  }
}


void FunctionBFC::ieGen_Operands(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  int opNum = 0;
  // Add operands to list of symbols
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
    
#ifdef DEBUG_LLVM
    if (!(v->hasName())) {
      blame_info<<"Standard Operand No Name "<<" "<<v<<" ";
      printValueIDName(v);
      blame_info<<endl;
    }
#endif      
  
    if (v->hasName() && !isa<BasicBlock>(v)) {
      if (variables.count(v->getName().str()) == 0) {
        string name = v->getName().str();
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(2-Operands) for "<<name<<endl;
#endif 
        NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;
        
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
        }

        variables[name] = vp;          
        varCount++;
        //printCurrentVariables();
      }
    }
    else if(isa<ConstantExpr>(v)) { //v has no name but is a constant expression
#ifdef DEBUG_LLVM
      blame_info<<"Value is ConstantExpr"<<endl;
#endif 
      ConstantExpr *ce = cast<ConstantExpr>(v);  
      createNPFromConstantExpr(ce, varCount, currentLineNum, fbb); 
    }
    opNum++;
  }
}


void FunctionBFC::ieGen_OperandsStore(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  int opNum = 0;
  // Add operands to list of symbols
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
#ifdef DEBUG_LLVM
    if (!(v->hasName())) {
      blame_info<<"Standard Operand No Name "<<" "<<v<<" ";
      printValueIDName(v);
      blame_info<<endl;
    }
#endif      
    if (v->hasName() && !isa<BasicBlock>(v)) {
      bool lnmChanged = false;
      NodeProps *vp = NULL;
      if (variables.count(v->getName().str()) == 0) { //Usually the operands in store are pre-declared
        string name = v->getName().str();              //so it should've been in variables already
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(2-Store) for "<<name<<endl;
#endif 
        vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;
        
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
          lnmChanged = true;
        }
        variables[v->getName().str()] = vp;          
        varCount++;
      }
      else 
        vp = variables[v->getName().str()];

      //we create a FuncStore only when we met the content  
      if (opNum==0) {
        User::op_iterator op_i2 = pi->op_begin();
        op_i2++;
        Value  /**firstStr = *op_i2, */  *secondStr = op_i2->get();
#ifdef DEBUG_LLVM
        blame_info<<"STORE to(1) "<<secondStr->getName().str()<<" from "<<vp->name<<" "<<lnm[currentLineNum]<<endl;
#endif 
        FuncStores *fs = new FuncStores();
                
        if (secondStr->hasName() && variables.count(secondStr->getName().str()))
          fs->receiver = variables[secondStr->getName().str()];
        else
          fs->receiver = NULL;
          
        fs->contents = vp;
        fs->line_num = currentLineNum;
        if (lnmChanged)
          fs->lineNumOrder = lnm[currentLineNum]-1; //since we've increment lnm_cln before, so we need to one step back
        else
          fs->lineNumOrder = lnm[currentLineNum];
#ifdef DEBUG_LLVM
        blame_info<<"STORE to(1) fs->lineNumOrder="<<fs->lineNumOrder<<" in line# "<<currentLineNum<<endl;
#endif 
        allStores.push_back(fs);
        //added by Hui 05/10/16: for storeLineNumOrder
        (vp->storeLineNumOrder)[currentLineNum] = fs->lineNumOrder;
      }  
    }
  
    // This is for dealing with Constants 
    else if (isa<ConstantInt>(v)) {
      ConstantInt *cv = cast<ConstantInt>(v);
      int number = cv->getSExtValue();
    
      char tempBuf[64];
      sprintf (tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, opNum, Instruction::Store);    
      char *vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
      
      strcpy(vN,tempBuf);
      vN[strlen(tempBuf)]='\0';
      const char * vName = vN;
      bool lnmChanged = false;
      NodeProps *vp = NULL;

      string vNameStr(vName);
      if (variables.count(vNameStr) == 0) {
        string name(vName);
        //cout<<"Creating VP for Constant "<<vName<<" in "<<getSourceFuncName()<<endl;
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(3) for "<<name<<endl;
#endif 
        vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;
              
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
          lnmChanged = true;
        }
        
        variables[vNameStr] = vp;
        varCount++;
      }
      else
        vp = variables[vNameStr];

      //we create a FuncStore only when we met the content  
      if (opNum==0) {
        User::op_iterator op_i2 = pi->op_begin();
        op_i2++;
        Value  /**firstStr = *op_i2, */  *secondStr = op_i2->get();
#ifdef DEBUG_LLVM
        blame_info<<"STORE to(2) "<<secondStr->getName().str()<<" from "<<vp->name<<" "<<lnm[currentLineNum]<<endl;
#endif 
        FuncStores *fs = new FuncStores();
                
        if (secondStr->hasName() && variables.count(secondStr->getName().str()))
          fs->receiver = variables[secondStr->getName().str()];
        else
          fs->receiver = NULL;
              
        fs->contents = vp;
        fs->line_num = currentLineNum;
        if (lnmChanged)
          fs->lineNumOrder = lnm[currentLineNum]-1;//same reason as before, we need to one step back
        else
          fs->lineNumOrder = lnm[currentLineNum];
#ifdef DEBUG_LLVM
        blame_info<<"STORE to(2) fs->lineNumOrder="<<fs->lineNumOrder<<" in line# "<<currentLineNum<<endl;
#endif 
        allStores.push_back(fs);
      }
    }

    else if (isa<ConstantFP>(v)) {
      ConstantFP *cfp = cast<ConstantFP>(v);
      const APFloat apf = cfp->getValueAPF(); //llvm::APFloat
    
      if (APFloat::semanticsPrecision(apf.getSemantics()) == 24) { //why 24 ? :single-precision float
        float floatNum = apf.convertToFloat();
#ifdef DEBUG_LLVM
        blame_info<<"Converted to float! "<<floatNum<<endl;
#endif 
        char tempBuf[64];
        sprintf (tempBuf, "Constant+%g+%i+%i+%i", floatNum, currentLineNum, opNum, Instruction::Store);    
        char *vN = (char *)malloc(sizeof(char)*(strlen(tempBuf)+1));
        
        strcpy(vN,tempBuf);
        vN[strlen(tempBuf)]='\0';
        const char * vName = vN;
        bool lnmChanged = false;
        NodeProps *vp = NULL;

        string vNameStr(vName);
        if (variables.count(vNameStr) == 0) {
          string name(vName);
#ifdef DEBUG_VP_CREATE
          blame_info<<"Adding NodeProps(5) for "<<name<<endl;
#endif 
          vp = new NodeProps(varCount,name,currentLineNum,pi);
          vp->fbb = fbb;
          
          if (currentLineNum != 0) {
            int lnm_cln = lnm[currentLineNum];
            vp->lineNumOrder = lnm_cln;
            lnm_cln++;
            lnm[currentLineNum] = lnm_cln;
            lnmChanged = true;
          }
          
          variables[vNameStr] = vp;
          varCount++;
        }
        else
          vp = variables[vNameStr];

        //we create a FuncStore only when we met the content  
        if (opNum==0) {
          User::op_iterator op_i2 = pi->op_begin();
          op_i2++;
          Value  /**firstStr = *op_i2, */  *secondStr = op_i2->get();
#ifdef DEBUG_LLVM
          blame_info<<"STORE to(3) "<<secondStr->getName().str()<<" from "<<vp->name<<" "<<lnm[currentLineNum]<<endl;
#endif 
          FuncStores *fs = new FuncStores();
              
          if (secondStr->hasName() && variables.count(secondStr->getName().str()))
            fs->receiver = variables[secondStr->getName().str()];
          else
            fs->receiver = NULL;
            
          fs->contents = vp;
          fs->line_num = currentLineNum;
          if (lnmChanged)
            fs->lineNumOrder = lnm[currentLineNum]-1;
          else
            fs->lineNumOrder = lnm[currentLineNum];
#ifdef DEBUG_LLVM
          blame_info<<"STORE to(3) fs->lineNumOrder="<<fs->lineNumOrder<<" in line# "<<currentLineNum<<endl;
#endif 
          allStores.push_back(fs);
        }
      }
      else if(APFloat::semanticsPrecision(apf.getSemantics()) == 53) {
        double floatNum = apf.convertToDouble();
        char tempBuf[70];
        sprintf (tempBuf, "Constant+%g2.2+%i+%i+%i", floatNum, currentLineNum, opNum, Instruction::Store);
#ifdef DEBUG_LLVM
        blame_info<<"Converted to double! "<<tempBuf<<endl;
#endif 
        char * vN = (char *) malloc(sizeof(char)*(strlen(tempBuf)+1));
        
        strcpy(vN,tempBuf);
        vN[strlen(tempBuf)]='\0';
        const char * vName = vN;
        bool lnmChanged = false;
        NodeProps *vp = NULL;
        
        string vNameStr(vName);
        if (variables.count(vNameStr) == 0) {
          string name(vName);
#ifdef DEBUG_VP_CREATE
          blame_info<<"Creating VP for Constant "<<vName<<" in "<<getSourceFuncName()<<endl;
          blame_info<<"Adding NodeProps(6) for "<<name<<endl;
#endif 
          
          NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
          vp->fbb = fbb;
          
          if (currentLineNum != 0) {
            int lnm_cln = lnm[currentLineNum];
            vp->lineNumOrder = lnm_cln;
            lnm_cln++;
            lnm[currentLineNum] = lnm_cln;
            lnmChanged = true;
          }
        
          variables[vNameStr] = vp;
          varCount++;
        }
        else 
          vp = variables[vNameStr];
    
        //we create a FuncStore only when we met the content  
        if (opNum==0) {
          User::op_iterator op_i2 = pi->op_begin();
          op_i2++;
          Value  /**firstStr = *op_i2, */  *secondStr = op_i2->get();
          FuncStores *fs = new FuncStores();
                
          if (secondStr->hasName() && variables.count(secondStr->getName().str()))
            fs->receiver = variables[secondStr->getName().str()];
          else
            fs->receiver = NULL;
          
          fs->contents = vp;
          fs->line_num = currentLineNum;
          if (lnmChanged)
            fs->lineNumOrder = lnm[currentLineNum]-1;
          else
            fs->lineNumOrder = lnm[currentLineNum];

          blame_info<<"STORE to(4) fs->lineNumOrder="<<fs->lineNumOrder<<" in line# "<<currentLineNum<<endl;
          allStores.push_back(fs);
        }
      }
    }
        
    else if (isa<ConstantExpr>(v)) {
#ifdef DEBUG_LLVM
      blame_info<<"Value is ConstantExpr"<<endl;
#endif 
      ConstantExpr *ce = cast<ConstantExpr>(v);  
      createNPFromConstantExpr(ce, varCount, currentLineNum, fbb);
                
      //---------------------------------------------------------------------------//
      char tempBuf[18];
      sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
      string name(tempBuf); // Use the address of the instruction as its name ??
      name += ".CE";
                
      char tempBuf2[10];
      sprintf(tempBuf2, ".%d", currentLineNum);
      name.append(tempBuf2);

      bool lnmChanged = false;
      NodeProps *vp = NULL;
                
      if (variables.count(name) == 0){
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(Store ConstantExpr Operand) for "<<name<<endl;
#endif
        vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;          
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
          lnmChanged = true;
        }
                      
        variables[name] = vp;
        varCount++;
      }
      else
        vp = variables[name];

      if (opNum==0) {
        User::op_iterator op_i2 = pi->op_begin();
        op_i2++;
        Value  /**firstStr = *op_i2, */  *secondStr = op_i2->get();
#ifdef DEBUG_LLVM
        blame_info<<"STORE to(6) "<<secondStr->getName().str()<<" from "<<vp->name<<" "<<lnm[currentLineNum]<<endl;
#endif 
        FuncStores *fs = new FuncStores();
                    
        if (secondStr->hasName() && variables.count(secondStr->getName().str()))
          fs->receiver = variables[secondStr->getName().str()];
        else
          fs->receiver = NULL;
         
        fs->contents = vp;
        fs->line_num = currentLineNum;
        if (lnmChanged)
          fs->lineNumOrder = lnm[currentLineNum]-1;
        else
          fs->lineNumOrder = lnm[currentLineNum];

        blame_info<<"STORE to(6) fs->lineNumOrder="<<fs->lineNumOrder<<" in line# "<<currentLineNum<<endl;
        allStores.push_back(fs);
                
        //added by Hui 05/10/16: for storeLineNumOrder
        (vp->storeLineNumOrder)[currentLineNum] = fs->lineNumOrder;
      }
    }

    //we don't add constant null as nodes because they are initializer
    else if (!v->hasName() && !isa<Constant>(v)) {  
      char tempBuf2[18];
      sprintf(tempBuf2, "0x%x", /*(unsigned)*/v);
      string name(tempBuf2);
      bool lnmChanged = false;
      NodeProps *vp = NULL;

      if (variables.count(name) == 0){
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(Store Reg Operand) for "<<name<<endl;
#endif
        vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;          
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
          lnmChanged = true;
        }
          
        variables[name] = vp;
        varCount++;
      }
      else
        vp = variables[name];

      //we create a FuncStore only when we met the content  
      if (opNum==0) {
        User::op_iterator op_i2 = pi->op_begin();
        op_i2++;
        Value  /**firstStr = *op_i2, */  *secondStr = op_i2->get();
#ifdef DEBUG_LLVM
        blame_info<<"STORE to(5) "<<secondStr->getName().str()<<" from "<<vp->name<<" "<<lnm[currentLineNum]<<endl;
#endif 
        FuncStores *fs = new FuncStores();
              
        if (secondStr->hasName() && variables.count(secondStr->getName().str()))
          fs->receiver = variables[secondStr->getName().str()];
        else
          fs->receiver = NULL;
        
        fs->contents = vp;
        fs->line_num = currentLineNum;
        if (lnmChanged)
          fs->lineNumOrder = lnm[currentLineNum]-1;
        else
          fs->lineNumOrder = lnm[currentLineNum];

#ifdef DEBUG_LLVM
        blame_info<<"STORE to(5) fs->lineNumOrder="<<fs->lineNumOrder<<" in line# "<<currentLineNum<<endl;
#endif 
        allStores.push_back(fs);
            
        //added by Hui 05/10/16: for storeLineNumOrder
        (vp->storeLineNumOrder)[currentLineNum] = fs->lineNumOrder;
      }
    }

    else { 
#ifdef DEBUG_LLVM
      blame_info<<"Not storing this store operand: "<<v<<endl;
#endif 
    }

    opNum++;
  } //for all operands of store
}

void FunctionBFC::ieGen_OperandsGEP(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  int opNum = 0;
  // Add operands to list of symbols, which are var nodes in AST
  //blame_info<<"NumOperands="<<pi->getNumOperands()<<endl;
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
#ifdef DEBUG_LLVM
    if (!(v->hasName())) {
      blame_info<<"Standard Operand No Name "<<" "<<v<<" ";
      printValueIDName(v);
      blame_info<<endl;
    }
#endif    
    if (v->hasName() && !isa<BasicBlock>(v)) {
      if (variables.count(v->getName().str()) == 0) {
        string name = v->getName().str();
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(2-GEP) for "<<name<<endl;
#endif 
        NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;
        
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
        }

        variables[v->getName().str()] = vp;          
        varCount++;
      }
    }
    // This is for dealing with Constants, no matter v has name or not
    else if (isa<ConstantInt>(v)) {
      ConstantInt *cv = (ConstantInt *)v;  
      int number = cv->getSExtValue();
      
      if (opNum == 0 || opNum == 1) { //ignore the first two operands
        opNum++;                    //major var and its index in array
        continue;                   //default 0 if only 1 structure isf
      }
      
      char tempBuf[64];
      //sprintf (tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, opNum, pi->getOpcode());    
      sprintf (tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, opNum, Instruction::GetElementPtr);    
      char * vN = (char *)malloc(sizeof(char)*(strlen(tempBuf)+1));
      
      strcpy(vN,tempBuf);
      vN[strlen(tempBuf)]='\0';
      const char * vName = vN;
      
      string vNameStr(vName);
      if (variables.count(vNameStr) == 0 && opNum >= 2) { //keep nodes for second index or later
        string name(vName);
        //cout<<"Creating VP for Constant "<<vName<<" in "<<getSourceFuncName()<<endl;
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(4) for "<<name<<endl;
#endif 
        NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;
        
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
        }
        
        variables[vNameStr] = vp;
        varCount++;
        //printCurrentVariables();
      }
    }
    
    else if(isa<ConstantExpr>(v)) {
#ifdef DEBUG_LLVM
      blame_info<<"Value is ConstantExpr"<<endl;
#endif 
      ConstantExpr *ce = cast<ConstantExpr>(v);  
      createNPFromConstantExpr(ce, varCount, currentLineNum, fbb);
    }

    else if (!v->hasName() && !isa<BasicBlock>(v)) {
      char tempBuf[20];
      sprintf(tempBuf, "0x%x", /*(unsigned)*/pi);
      string name(tempBuf); // Use the address of the instruction as its name ??

      if (variables.count(name) ==0) {
#ifdef DEBUG_LLVM
        blame_info<<"How come this GEP node never showed up ?? "<<name<<endl;
#endif
      }
    }

    opNum++;
  }
}


void FunctionBFC::ieGen_OperandsExtVal(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  int opNum = 0;
  // Add operands to list of symbols, which are var nodes in AST
  //blame_info<<"NumOperands="<<pi->getNumOperands()<<endl;
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
#ifdef DEBUG_LLVM
    if (!(v->hasName())) {
      blame_info<<"Standard Operand No Name "<<" "<<v<<" ";
      printValueIDName(v);
      blame_info<<endl;
    }
#endif    
    if (v->hasName() && !isa<BasicBlock>(v)) {
      if (variables.count(v->getName().str()) == 0) {
        string name = v->getName().str();
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(16) for "<<name<<endl;
#endif 
        NodeProps * vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;
        
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
        }

        variables[v->getName().str()] = vp;          
        varCount++;
      }
    }
  
    // This is for dealing with Constants, no matter v has name or not
    else if (isa<ConstantInt>(v)) {
      ConstantInt *cv = (ConstantInt *)v;  
      int number = cv->getSExtValue();
      
      char tempBuf[64];
      //sprintf (tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, opNum, pi->getOpcode());    
      sprintf (tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, opNum, Instruction::ExtractValue);    
      char * vN = (char *)malloc(sizeof(char)*(strlen(tempBuf)+1));
    
      strcpy(vN,tempBuf);
      vN[strlen(tempBuf)]='\0';
      const char * vName = vN;
      string vNameStr(vName);
    
      if (variables.count(vNameStr) == 0 && opNum >= 1) { //TC: indices start from 1 in extractValue
        string name(vName);
        //cout<<"Creating VP for Constant "<<vName<<" in "<<getSourceFuncName()<<endl;
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(17) for "<<name<<endl;
#endif 
        NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;
      
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
        }
        
        variables[vNameStr] = vp;
        varCount++;
        //printCurrentVariables();
      }
    }
    
    else if(isa<ConstantExpr>(v)) {
#ifdef DEBUG_LLVM
      blame_info<<"Value is ConstantExpr"<<endl;
#endif 
      ConstantExpr * ce = cast<ConstantExpr>(v);  
      createNPFromConstantExpr(ce, varCount, currentLineNum, fbb);
    }

    opNum++;
  }
}


void FunctionBFC::ieGen_OperandsIstVal(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  int opNum = 0;
  // Add operands to list of symbols, which are var nodes in AST
    //blame_info<<"NumOperands="<<pi->getNumOperands()<<endl;
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
#ifdef DEBUG_LLVM
    if (!(v->hasName())) {
    blame_info<<"Standard Operand No Name "<<" "<<v<<" ";
    printValueIDName(v);
    blame_info<<endl;
    }
#endif    
    if (v->hasName() &&  !isa<BasicBlock>(v)) {
    if (variables.count(v->getName().str()) == 0) {
        string name = v->getName().str();
#ifdef DEBUG_VP_CREATE
      blame_info<<"Adding NodeProps(18) for "<<name<<endl;
#endif 
      NodeProps * vp = new NodeProps(varCount,name,currentLineNum,pi);
      vp->fbb = fbb;
      
      if (currentLineNum != 0) {
      int lnm_cln = lnm[currentLineNum];
      vp->lineNumOrder = lnm_cln;
      lnm_cln++;
      lnm[currentLineNum] = lnm_cln;
      }

      variables[v->getName().str()] = vp;          
      varCount++;
    }
    }
   
      // This is for dealing with Constants, no matter v has name or not
    else if (isa<ConstantInt>(v)) {
    ConstantInt *cv = (ConstantInt *)v;  
    int number = cv->getSExtValue();
    
    char tempBuf[64];
    //sprintf (tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, opNum, pi->getOpcode());    
    sprintf (tempBuf, "Constant+%i+%i+%i+%i", number, currentLineNum, opNum, Instruction::InsertValue);    
    char *vN = (char *)malloc(sizeof(char)*(strlen(tempBuf)+1));
    
    strcpy(vN,tempBuf);
    vN[strlen(tempBuf)]='\0';
    const char * vName = vN;
      
        string vNameStr(vName);
    if (variables.count(vNameStr) == 0 && opNum >=3) { 
      string name(vName);
    //cout<<"Creating VP for Constant "<<vName<<" in "<<getSourceFuncName()<<endl;
#ifdef DEBUG_VP_CREATE
      blame_info<<"Adding NodeProps(19) for "<<name<<endl;
#endif 
      NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
      vp->fbb = fbb;
        
      if (currentLineNum != 0) {
       int lnm_cln = lnm[currentLineNum];
      vp->lineNumOrder = lnm_cln;
      lnm_cln++;
      lnm[currentLineNum] = lnm_cln;
      }
        
      variables[vNameStr] = vp;
      varCount++;
      //printCurrentVariables();
    }
    }
    else if(isa<ConstantExpr>(v)) {
#ifdef DEBUG_LLVM
      blame_info<<"Value is ConstantExpr"<<endl;
#endif 
      ConstantExpr * ce = cast<ConstantExpr>(v);  
      createNPFromConstantExpr(ce, varCount, currentLineNum, fbb);
    }

    opNum++;
  }
}

//Exactly same as ieGen_Operands(): opcode isn't treated as the first operand
void FunctionBFC::ieGen_OperandsAtomic(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  int opNum = 0;
  // Add operands to list of symbols
    User::op_iterator op_i = pi->op_begin();
  for (User::op_iterator op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
    
#ifdef DEBUG_LLVM
    if (!(v->hasName())) {
      blame_info<<"Standard Operand No Name "<<" "<<v<<" ";
      printValueIDName(v);
      blame_info<<endl;
    }
#endif      
    
    if (v->hasName() && !isa<BasicBlock>(v)) {
      if (variables.count(v->getName().str()) == 0) {
        string name = v->getName().str();
#ifdef DEBUG_VP_CREATE
        blame_info<<"Adding NodeProps(2-Atomic) for "<<name<<endl;
#endif 
        NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
        vp->fbb = fbb;
        
        if (currentLineNum != 0) {
          int lnm_cln = lnm[currentLineNum];
          vp->lineNumOrder = lnm_cln;
          lnm_cln++;
          lnm[currentLineNum] = lnm_cln;
        }

        variables[v->getName().str()] = vp;          
        varCount++;
        //printCurrentVariables();
      }
    }
    else if(isa<ConstantExpr>(v)) { //v has no name but is a constant expression
#ifdef DEBUG_LLVM
        blame_info<<"Value is ConstantExpr"<<endl;
#endif 
      ConstantExpr *ce = cast<ConstantExpr>(v);  
      createNPFromConstantExpr(ce, varCount, currentLineNum, fbb); 
    }

    opNum++;
  }
}


// Almost the same as ieDefault, except we don deal with the last operand, which is the call node
void FunctionBFC::ieDefaultPTXIntrinsic(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"In ieDefaultPTXIntrinsic"<<endl;
#endif
  
  ieGen_LHS(pi, varCount, currentLineNum, fbb);
  ieGen_OperandsPTXIntrinsic(pi, varCount, currentLineNum, fbb);
}


void FunctionBFC::ieDefault(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  if (isa<Instruction>(pi)) {
    Instruction *pipi = cast<Instruction>(pi);
#ifdef DEBUG_LLVM
    blame_info<<"In ieDefault for "<<pipi->getOpcodeName()<<endl;
#endif
  }
  else if (isa<ConstantExpr>(pi)) {
    ConstantExpr *cepi = cast<ConstantExpr>(pi);
#ifdef DEBUG_LLVM
    blame_info<<"In ieDefault for "<<cepi->getOpcodeName()<<endl;
#endif
  }
  
  ieGen_LHS(pi, varCount, currentLineNum, fbb);
  ieGen_Operands(pi, varCount, currentLineNum, fbb);
}


void FunctionBFC::ieLoad(Instruction * pi, int & varCount, int & currentLineNum, FunctionBFCBB * fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"In ieLoad for "<<pi->getName().str()<<endl;
#endif
  //we don't care about the node representing the global filename
  Value *op = *(pi->op_begin());
  if (op->hasName() && op->getName().str().find("_literal_")==0) {
#ifdef DEBUG_LLVM
    blame_info<<"Operand "<<op->getName().str()<<" is filename. Not added"<<endl;
#endif
    return;
  }

  ieGen_LHS(pi, varCount, currentLineNum, fbb);
  ieGen_Operands(pi, varCount, currentLineNum, fbb);
}


void FunctionBFC::ieGetElementPtr(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"In ieGetElementPtr for "<<pi->getName().str()<<endl;
#endif
  ieGen_LHS(pi, varCount, currentLineNum, fbb);
  ieGen_OperandsGEP(pi, varCount, currentLineNum, fbb);
}


void FunctionBFC::ieExtractValue(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"In ieExtractValue for "<<pi->getName().str()<<endl;
#endif
  ieGen_LHS(pi, varCount, currentLineNum, fbb);
  ieGen_OperandsExtVal(pi, varCount, currentLineNum, fbb);
}


void FunctionBFC::ieInsertValue(User *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"In ieInsertValue for "<<pi->getName().str()<<endl;
#endif
  ieGen_LHS(pi, varCount, currentLineNum, fbb);
  ieGen_OperandsIstVal(pi, varCount, currentLineNum, fbb);
}


void FunctionBFC::ieStore(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"In ieStore"<<endl;
#endif
  //We'll generate allStores later after both nodes of the operands are built
  ieGen_OperandsStore(pi, varCount, currentLineNum, fbb);
}

void FunctionBFC::ieSelect(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"In ieSelect"<<endl;
#endif 
  
#ifdef DEBUG_LLVM
  blame_info<<"LLVM__(examineInstruction)(Select) -- pi "<<pi->getName().str()<<" "<<pi<<" "<<pi->getOpcodeName()<<endl;
#endif     
  // Add LHS variable to list of symbols
  if (pi->hasName() && variables.count(pi->getName().str()) == 0) {
    string name = pi->getName().str();
#ifdef DEBUG_VP_CREATE
    blame_info<<"Adding NodeProps(7) for "<<name<<endl;
#endif 
    NodeProps *vp = new NodeProps(varCount,name,currentLineNum,pi);
    vp->fbb = fbb;
    if (currentLineNum != 0) {
     int lnm_cln = lnm[currentLineNum];
    vp->lineNumOrder = lnm_cln;
    lnm_cln++;
    lnm[currentLineNum] = lnm_cln;
    }
  
    variables[pi->getName().str()] = vp;
    varCount++;        
  }  
  
  int opNum = 0;
  
  // Add operands to list of symbols
  for (User::op_iterator op_i = pi->op_begin(), op_e = pi->op_end(); op_i != op_e; ++op_i) {
    Value *v = op_i->get();
    if (v->hasName() &&  !isa<BasicBlock>(v)) {
    if (variables.count(v->getName().str()) == 0) {
        string name = v->getName().str();
#ifdef DEBUG_VP_CREATE
      blame_info<<"Adding NodeProps(8) for "<<name<<endl;
#endif 
      NodeProps * vp = new NodeProps(varCount,name,currentLineNum,pi);
      vp->fbb = fbb;
    
          if (currentLineNum != 0) {
      int lnm_cln = lnm[currentLineNum];
      vp->lineNumOrder = lnm_cln;
      lnm_cln++;
      lnm[currentLineNum] = lnm_cln;
      }
      variables[v->getName().str()] = vp;          
      varCount++;
        }
    }
  
      opNum++;
  }
}


void FunctionBFC::ieBlank(Instruction *pi, int &currentLineNum)
{
#ifdef DEBUG_LLVM
  blame_info<<"In ieBlank for opcode "<<pi->getOpcodeName()<<" "<<currentLineNum<<endl;
#endif
  
}

//To be used for processing atomic memory operations
void FunctionBFC::ieMemAtomic(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"In ieMemAtomic for opocode "<<pi->getOpcodeName()<<" at "<<currentLineNum<<endl;
#endif
  if (pi->getOpcode() == Instruction::AtomicCmpXchg) { // same process as "Load"
#ifdef DEBUG_LLVM
    blame_info<<"In ieMemAtomic for cmpxchg"<<endl;
#endif                 
    ieGen_LHS(pi, varCount, currentLineNum, fbb);
    ieGen_Operands(pi, varCount, currentLineNum, fbb);
  }

  else if (pi->getOpcode() == Instruction::AtomicRMW) {
#ifdef DEBUG_LLVM
    blame_info<<"In ieMemAtomic for atomicrmw"<<endl;
#endif 
    ieGen_LHS(pi, varCount, currentLineNum, fbb);
    ieGen_OperandsAtomic(pi, varCount, currentLineNum, fbb);
  }
}

void FunctionBFC::ieBitCast(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{
  /* We need to determine which symbols deal with debug info and ignore them */  
#ifdef DEBUG_LLVM
  blame_info<<"LLVM__(examineInstruction) -- pi "<<pi->getName().str()<<" "<<pi<<" "<<pi->getOpcodeName()<<endl;
#endif
  
  //for (Value::use_iterator use_i = pi->use_begin(), use_e = pi->use_end(); use_i != use_e; ++use_i)
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
  
  ieGen_LHS(pi, varCount, currentLineNum, fbb);
  ieGen_Operands(pi, varCount, currentLineNum, fbb);
}


void FunctionBFC::ieAlloca(Instruction *pi, int &varCount, int &currentLineNum, FunctionBFCBB *fbb)
{  
  ieGen_LHS_Alloca(pi, varCount, currentLineNum, fbb);
  ieGen_Operands(pi, varCount, currentLineNum, fbb);  
}



void FunctionBFC::examineInstruction(Instruction *pi, int &varCount, int &currentLineNum, 
                        RegHashProps &variables, FunctionBFCBB *fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"Entering examineInstruction "<<pi->getOpcodeName()<<" "<<pi<<" "<<currentLineNum<<" ";
  printValueIDName(pi);
  blame_info<<endl;
#endif
  
    // We are interested in operands from
    // - Binary Ops
  // - Comparison Operations 
  // - Cast Operations
  // - Malloc/Free/Alloca
  // - Load/Store
  
#ifdef ENABLE_FORTRAN
//  if (firstGEPCheck(pi) == false) {//only useful for fortran
//        blame_info<<"firstGEPCheck return false !"<<endl;
//        return;
//    }
#endif

  genDILocationInfo(pi, currentLineNum, fbb); //generate location info of the current instruction

  // These call operations are almost exclusively for llvm.dbg.declare, the
  //  more general case with "real" data will be tackled below
  if (pi->getOpcode() == Instruction::Call) {
    // process llvm.dbg.declare intrinsic calls  
    if (parseDeclareIntrinsic(pi, currentLineNum, fbb) == true)   
      return;
  }
  
    // 1
  // TERMINATOR OPS
  if (pi->getOpcode() == Instruction::Ret)    
    ieBlank(pi, currentLineNum);
  else if (pi->getOpcode() == Instruction::Br)
    ieBlank(pi, currentLineNum);
  else if (pi->getOpcode() == Instruction::Switch)
    ieBlank(pi, currentLineNum);
  else if (pi->getOpcode() == Instruction::IndirectBr) //TC
    ieBlank(pi, currentLineNum);
  else if (pi->getOpcode() == Instruction::Invoke)
    ieInvoke(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::Resume) //TC
    ieBlank(pi, currentLineNum);
  else if (pi->getOpcode() == Instruction::Unreachable)
    ieBlank(pi, currentLineNum);
  /*else if (pi->getOpcode() == Instruction::CleanupReturn) //TC
    ieBlank(pi, currentLineNum);
  else if (pi->getOpcode() == Instruction::CatchReturn) //TC
    ieBlank(pi, currentLineNum);
  else if (pi->getOpcode() == Instruction::CatchSwitch) //TC
    ieBlank(pi, currentLineNum);*/
  // END TERMINATOR OPS
  
  // BINARY OPS 
  else if (pi->getOpcode() == Instruction::Add)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FAdd)
    ieDefault(pi, varCount, currentLineNum, fbb); //TC
  else if (pi->getOpcode() == Instruction::Sub)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FSub)    //TC
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::Mul)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FMul)    //TC
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::UDiv)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::SDiv)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FDiv)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::URem)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::SRem)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FRem)
    ieDefault(pi, varCount, currentLineNum, fbb);
  // END BINARY OPS
  
  // LOGICAL OPERATORS 
  else if (pi->getOpcode() == Instruction::Shl)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::LShr)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::AShr)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::And)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::Or)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::Xor)
    ieDefault(pi, varCount, currentLineNum, fbb);
  // END LOGICAL OPERATORS
  
  // MEMORY OPERATORS
  else if (pi->getOpcode() == Instruction::Alloca)
    ieAlloca(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::Load)
    ieLoad(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::Store)
    ieStore(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::GetElementPtr)  
    ieGetElementPtr(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::Fence)    //TC
    ieBlank(pi, currentLineNum);
  else if (pi->getOpcode() == Instruction::AtomicCmpXchg)  //TC
    ieMemAtomic(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::AtomicRMW)    //TC
    ieMemAtomic(pi, varCount, currentLineNum, fbb);

  // END MEMORY OPERATORS
  
  // CAST OPERATORS
  else if (pi->getOpcode() == Instruction::Trunc)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::ZExt)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::SExt)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FPToUI)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FPToSI)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::UIToFP)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::SIToFP)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FPTrunc)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FPExt)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::PtrToInt)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::IntToPtr)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::BitCast)
    ieBitCast(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::AddrSpaceCast) //cuda 04/24/18
    ieBitCast(pi, varCount, currentLineNum, fbb); //exactly same process as bitcast
  
  // END CAST OPERATORS
  
  // OTHER OPERATORS
  else if (pi->getOpcode() == Instruction::ICmp)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::FCmp)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::PHI)
    ieDefault(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::Call)
    ieCall(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::Select)
    ieSelect(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::ExtractValue)
    ieExtractValue(pi, varCount, currentLineNum, fbb);
  else if (pi->getOpcode() == Instruction::InsertValue)
    ieInsertValue(pi, varCount, currentLineNum, fbb);
  else //There're still other Insts unhandled as: UserOp1, UserOp2, VAArg,   //TC
  {    //IExtractElement, ShuffleVector, LandingPad    
#ifdef DEBUG_ERROR
    blame_info<<"LLVM__(examineInstruction)(Not processed inst(ie)) -- "<<pi->getOpcodeName()<<endl;
#endif      
  }      
}



void FunctionBFC::createNPFromConstantExpr(ConstantExpr *ce, int &varCount, int &currentLineNum, FunctionBFCBB  *fbb)
{
#ifdef DEBUG_LLVM
  blame_info<<"In CreateVPFromConstantExpr"<<endl;
#endif
  
  if (ce->getOpcode() == Instruction::GetElementPtr)
    ieGetElementPtr(ce, varCount, currentLineNum, fbb);
  else if (ce->getOpcode() == Instruction::ExtractValue)
    ieExtractValue(ce, varCount, currentLineNum, fbb);
  else if (ce->getOpcode() == Instruction::BitCast ||
           ce->getOpcode() == Instruction::AddrSpaceCast) {
    //No need to do anything here since for the graph we're just going to grab the first operand of cast anyway
#ifdef DEBUG_LLVM
    blame_info<<"CE is a cast inst "<<ce->getOpcodeName()<<endl; 
#endif
  }
  //else if (ce->getOpcode() == Instruction::PtrToInt)
  //ieDefault(ce, varCount, currentLineNum, fbb);
  else {
#ifdef DEBUG_LLVM
    blame_info<<"Not dealing Constant Expr cvpce for "<<ce->getOpcodeName()<<endl; //NOTHING TO DO WITH THIS ?
#endif
  }  
}


// In LLVM, each Param has the param number appended to it.  We are interested
// in the address of these params ".addr" appended to the name without
//  the number, THIS FUNCTION not used anywhere
void paramWithoutNumWithAddr(string &original)
{
  unsigned i;
  int startOfNum = -1;
  for (i=0; i < original.length(); i++)
  {
    if (original[i] >= 48 && original[i] <= 57 && startOfNum == -1)
      startOfNum = i;
    if (original[i] < 48 || original[i] > 57)
      startOfNum = -1;
  }
  
  original.replace(startOfNum, 5, ".addr");
}


bool FunctionBFC::varLengthParams()   //TO-CHECK va_arg: http://en.wikibooks.org/wiki/C++_Programming/Code/Standard_C_Library/Functions/va_arg
{
  for (Function::iterator b = func->begin(), be = func->end(); b != be; ++b) {
    for (BasicBlock::iterator i = (*b).begin(), ie = (*b).end(); i != ie; ++i) {
      Instruction *pi = &*i;
    //cout<<"Name - "<<pi->getName()<<endl;
    // If we make it to the first call without seeing argptr we know it's not variable list
      //variadic functions: a function that can take unfixed # of arguments, e.g: printf
      if (pi->getOpcode() == Instruction::Call) {
        return false;
      }
      // the final arg can be a list/array of args, which is called Varargs
      //if ( pi->getName().find("argptr") != string::npos) { //va_list argptr
      else if (pi->getOpcode() == Instruction::VAArg) {
#ifdef DEBUG_LLVM
        blame_info<<"LLVM__(varLengthParams)--Variable Length Args!"<<endl;
#endif
        return true;
      }
    }
  }
  return false;
}



