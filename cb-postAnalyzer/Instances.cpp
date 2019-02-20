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

#include "Instances.h"
#include "BlameProgram.h"
#include "BlameFunction.h"
#include "BlameModule.h"

using namespace std;
/*
void Instance::printInstance()
{
  vector<StackFrame>::iterator vec_SF_i;
  for (vec_SF_i = frames.begin(); vec_SF_i != frames.end(); vec_SF_i++) {
    if ((*vec_SF_i).lineNumber > 0){
      cout<<"At Frame "<<(*vec_SF_i).frameNumber<<" at line num "<<(*vec_SF_i).lineNumber;
      cout<<" in module "<<(*vec_SF_i).moduleName<<endl;
    }
    else {
      cout<<std::hex;
      cout<<"At Frame "<<(*vec_SF_i).frameNumber<<" at address "<<(*vec_SF_i).address<<endl;
      cout<<std::dec;
    }
  }
}

void Instance::printInstance_concise()
{
  vector<StackFrame>::iterator vec_SF_i;

  for (vec_SF_i = frames.begin(); vec_SF_i != frames.end(); vec_SF_i++) {
    stack_info<<(*vec_SF_i).frameNumber<<" "<<(*vec_SF_i).lineNumber<<" "<<
      (*vec_SF_i).moduleName<<" ";
    stack_info<<std::hex<<(*vec_SF_i).address<<std::dec;
    stack_info<<" "<<(*vec_SF_i).frameName;
    if (isForkStarWrapper((*vec_SF_i).frameName)) 
      stack_info<<" "<<(*vec_SF_i).info.callerNode<<" "<<(*vec_SF_i).info.calleeNode
      <<" "<<(*vec_SF_i).info.fid<<" "<<(*vec_SF_i).info.fork_num;
    stack_info<<endl;
  }
}


void Instance::removeRedundantFrames(ModuleHash &modules, string nodeName)
{
  stack_info<<"In removeRedundantFrames on "<<nodeName<<endl;
  vector<StackFrame>::iterator vec_SF_i, minusOne;
  vector<StackFrame> newFrames(frames);
  bool isBottomParsed = true;
  frames.clear();

  for (vec_SF_i=newFrames.begin(); vec_SF_i!=newFrames.end(); vec_SF_i++) {
    if (isForkStarWrapper((*vec_SF_i).frameName))
      continue; //always keep the frame to the final glueStacks stage
    if (isBottomParsed == true) { 
      isBottomParsed = false; 
      continue;              
    }
    minusOne = vec_SF_i - 1;
    if ((*vec_SF_i).moduleName== (*minusOne).moduleName && 
        (*vec_SF_i).lineNumber==(*minusOne).lineNumber) {
      if ((*vec_SF_i).frameName==(*minusOne).frameName) {
        stack_info<<"Redundant frame :"<<(*vec_SF_i).frameName<<
            ", delete frame# "<<(*vec_SF_i).frameNumber<<endl;
        (*vec_SF_i).toRemove = true;
      }
      else
        stack_info<<"Same file and line, but different function: curr("<<
          (*vec_SF_i).frameName<<"), prev("<<(*minusOne).frameName<<"), Kept for now"<<endl;
    }
  }

  //pick all the valid frames and push_back to "frames" again
  for (vec_SF_i=newFrames.begin(); vec_SF_i!=newFrames.end(); vec_SF_i++) {
    if ((*vec_SF_i).toRemove == false) {
        StackFrame sf(*vec_SF_i);
        frames.push_back(sf);
    }
  }
  //thorougly free the memory of newFrames
  vector<StackFrame>().swap(newFrames); 
  
}
  

// Actually this func is not necessary since wrap* funcs were deleted due to the missing bf
// Mainly used for removing fork_*_wrapper, thread_begin
void Instance::removeWrapFrames(string node, int InstNum)
{
  stack_info<<"In removeWrapFrames for instance #"<<InstNum<<" on "<<node<<endl;
  vector<StackFrame>::iterator vec_SF_i;
  vector<StackFrame> newFrames(frames);
  frames.clear();

  for (vec_SF_i=newFrames.begin(); vec_SF_i!=newFrames.end(); vec_SF_i++) {
    if ((*vec_SF_i).frameName.find("wrapcoforall")==0 
        || (*vec_SF_i).frameName.find("wrapon")==0 
        || isForkStarWrapper((*vec_SF_i).frameName) 
        || (*vec_SF_i).frameName=="thread_begin") {
        
      stack_info<<"Removable frame :"<<(*vec_SF_i).frameName<<
            ", delete frame# "<<(*vec_SF_i).frameNumber<<endl;
      (*vec_SF_i).toRemove = true;
    }
  }

  // Remove the last frame if it's chpl_gen_main TODO: Shall we keep chpl_gen_main?
  //if (*(newFrames.end()-1).frameName=="chpl_gen_main" && newFrames.size()>=2) 
  //  *(newFrames.end()-1).toRemove = true;

  //pick all the valid frames and push_back to "frames" again
  for (vec_SF_i=newFrames.begin(); vec_SF_i!=newFrames.end(); vec_SF_i++) {
    if ((*vec_SF_i).toRemove == false) {
        StackFrame sf(*vec_SF_i);
        frames.push_back(sf);
    }
  }
  //thorougly free the memory of newFrames
  vector<StackFrame>().swap(newFrames); 
  
}


//Added by Hui 12/25/15: trim the stack trace from main thread again(NOT USED ANYMORE)
void Instance::secondTrim(ModuleHash &modules, string nodeName)
{
  stack_info<<"In secondTrim on "<<nodeName<<endl;
  vector<StackFrame>::iterator vec_SF_i;
  vector<StackFrame> newFrames(frames);
  bool isBottomParsed = true;
  frames.clear();
  
  for (vec_SF_i=newFrames.begin(); vec_SF_i!=newFrames.end(); vec_SF_i++) {
    if (isForkStarWrapper((*vec_SF_i).frameName))
      continue; //always keep the frame to the final glueStacks stage
    
    BlameModule *bm = NULL;
    bm = modules[(*vec_SF_i).moduleName];
    //BlameFunction *bf = bm->findLineRange((*vec_SF_i).lineNumber);
    BlameFunction *bf = bm->getFunction((*vec_SF_i).frameName);
    if (bf==NULL) {
      stack_info<<"Weird: bf should exist for "<<(*vec_SF_i).frameName<<endl;
      return;
    }
    else {
      if (isBottomParsed == true) { //Only valid for compute instances, not for pre&fork
        isBottomParsed = false; //if it's first frame, then it doesn't
        continue;               //have to have callNodes, but later does
      }
      else {
        std::vector<VertexProps*>::iterator vec_vp_i;
        VertexProps *callNode = NULL;
        std::vector<VertexProps*> matchingCalls;
          
        for (vec_vp_i = bf->callNodes.begin(); vec_vp_i != bf->callNodes.end(); vec_vp_i++) {
          VertexProps *vp = *vec_vp_i;
          if (vp->declaredLine == (*vec_SF_i).lineNumber) {
            //just for test 
            stack_info<<"matching callNode: "<<vp->name<<endl;
            matchingCalls.push_back(vp);
          }
        }
        // we only need to check middle frames mapped to "coforall_fn/wrapcoforall_fn"
        if (matchingCalls.size() >= 1) { //exclude frame that maps to "forall/coforall" loop lines
          callNode = NULL;              //changed >1 to >=1 by Hui 03/24/16: as long as the callnode doesn't match the ...
          stack_info<<">= one call node at that line number"<<std::endl;    //previous frame, we need to remove it ..
            // figure out which call is appropriate                         //usually it's a Chapel inner func call
          vector<StackFrame>::iterator minusOne = vec_SF_i - 1;
          BlameModule *bmCheck = modules[(*minusOne).moduleName];
          if (bmCheck == NULL) {
            stack_info<<"BM of previous frame is null ! delete frame "<<(*vec_SF_i).frameNumber<<endl;
            (*vec_SF_i).toRemove = true;
          }
          else {
            //BlameFunction *bfCheck = bmCheck->findLineRange((*minusOne).lineNumber);
            BlameFunction *bf = bm->getFunction((*minusOne).frameName);
            if (bf == NULL) {
              stack_info<<"BF of previous frame is null ! delete frame "<<(*vec_SF_i).frameNumber<<endl;
              (*vec_SF_i).toRemove = true;
            }
            else {
              std::vector<VertexProps *>::iterator vec_vp_i2;
              for (vec_vp_i2 = matchingCalls.begin(); vec_vp_i2 != matchingCalls.end(); vec_vp_i2++) {
                VertexProps *vpCheck = *vec_vp_i2;
                // Look for subsets since vpCheck will have the line number concatenated
                if (vpCheck->name.find(bf->getName()) != std::string::npos)
                  callNode = vpCheck;
              }
          
              if (callNode == NULL) {
                  stack_info<<"No matching call nodes from multiple matches, delete frame "<<(*vec_SF_i).frameNumber<<endl;
                  (*vec_SF_i).toRemove = true;
              }
            }
          }
        }
      }
    }
  }

  //pick all the valid frames and push_back to "frames" again
  for (vec_SF_i=newFrames.begin(); vec_SF_i!=newFrames.end(); vec_SF_i++) {
    if ((*vec_SF_i).toRemove == false) {
        StackFrame sf(*vec_SF_i);
        frames.push_back(sf);
    }
  }
  //thorougly free the memory of newFrames
  vector<StackFrame>().swap(newFrames); 
}
//Added by Hui 12/20/15 : set all invalid frames's toRemove tag to TRUE in an instance
void Instance::trimFrames(ModuleHash &modules, int InstanceNum, string nodeName)
{
  if(this->instType == COMPUTE_INST)
    stack_info<<"Triming compute instance "<<InstanceNum<<" on "<<nodeName<<endl;
  else if(this->instType == PRESPAWN_INST)
    stack_info<<"Triming preSpawn instance "<<InstanceNum<<" on "<<nodeName<<endl;
  else if(this->instType == FORK_INST)
    stack_info<<"Triming fork instance "<<InstanceNum<<" on "<<nodeName<<endl;
  else if(this->instType == FORK_NB_INST)
    stack_info<<"Triming fork_nb instance "<<InstanceNum<<" on "<<nodeName<<endl;
  else if(this->instType == FORK_FAST_INST)
    stack_info<<"Triming fork_fast instance "<<InstanceNum<<" on "<<nodeName<<endl;
  else stack_info<<"What am I triming ? instType="<<this->instType<<endl;

  vector<StackFrame>::iterator vec_SF_i;
  vector<StackFrame> newFrames(frames);
  bool isBottomParsed = true;
  frames.clear(); //clear up this::frames after we copy everything to newFrames
  
  for (vec_SF_i=newFrames.begin(); vec_SF_i!=newFrames.end(); vec_SF_i++) {
    //if ((*vec_SF_i).lineNumber <= 0) {
    //  stack_info<<"LineNum <=0, delete frame "<<(*vec_SF_i).frameNumber<<endl;
    //  (*vec_SF_i).toRemove = true;
    //}
    //else { //lineNumber >0 Chapel has to be built with debug info
      BlameModule *bm = NULL;
      if ((*vec_SF_i).moduleName.empty()==false);
        bm = modules[(*vec_SF_i).moduleName];
      if (bm == NULL) {
        //we always keep the frame: fork*wrapper, thread_begin, polling&comm_barrier or pthread_spin_lock(if it's the last frame)
        if (isForkStarWrapper((*vec_SF_i).frameName) || (*vec_SF_i).frameName=="thread_begin"
            || (*vec_SF_i).frameName=="polling" || (*vec_SF_i).frameName=="chpl_comm_barrier"
            || ((*vec_SF_i).frameName=="pthread_spin_lock" && (vec_SF_i+1)==newFrames.end())) {
          continue;
        }
        else {
          (*vec_SF_i).toRemove = true; 
          stack_info<<"BM is NULL and it's neither important runtime functions, delete frame "<<(*vec_SF_i).frameNumber<<endl;
        }   
      }
      else { //bm != NULL, it's from user code
        // Use the combination of the module and the line number to determine the function & compare with frameName
        string fName = (*vec_SF_i).frameName;
        BlameFunction *bf = bm->getFunction(fName);
        if (bf == NULL) { //would remove frames corresponding to wrapcoforall/wrapon,etc.
          //if (fName == "chpl_gen_main") { // we keep it for now even it doesn't have bf
          //  isMainThread = true;
          //  continue;
          //}
          //else {
            stack_info<<"BF is NULL, delete frame #"<<(*vec_SF_i).frameNumber<<" "<<fName<<endl;
            (*vec_SF_i).toRemove = true;
          //}
        }
        else { //we found bf using frameName
          if (fName == "chpl_user_main" || fName == "chpl_gen_main") {
            isMainThread = true;//we know it's from the main thread if this inst has a frame of chpl_user_main
          }
          else {
            if (isBottomParsed == true) {
              isBottomParsed = false; //if it's first frame, then it doesn't
              continue;               //have to have callNodes, but later does
            }
            else {
              std::vector<VertexProps*>::iterator vec_vp_i;
              VertexProps *callNode = NULL;
              std::vector<VertexProps*> matchingCalls;
              
              for (vec_vp_i=bf->callNodes.begin(); vec_vp_i!=bf->callNodes.end(); vec_vp_i++) {
                VertexProps *vp = *vec_vp_i;
                if (vp->declaredLine == (*vec_SF_i).lineNumber) {
                  //just for test 
                  stack_info<<"matching callNode: "<<vp->name<<" at frame "<<
                      (*vec_SF_i).frameNumber<<endl;
                  matchingCalls.push_back(vp);
                }
              }
              if (matchingCalls.size() == 0) {
                stack_info<<"There's no matching callNode in this line, delete frame "
                          <<(*vec_SF_i).frameNumber<<endl;
                (*vec_SF_i).toRemove = true;
              }
            }
          }
        }
      }
    //}
  }

  //pick all the valid frames and push_back to "frames" again
  for (vec_SF_i=newFrames.begin(); vec_SF_i!=newFrames.end(); vec_SF_i++) {
    if ((*vec_SF_i).toRemove == false) {
        StackFrame sf(*vec_SF_i); //using c++ struct implicit constructor, tested work on these cases
        frames.push_back(sf);
    }
  }

  //thorougly free the memory of newFrames
  vector<StackFrame>().swap(newFrames); //here is why this works: 
                //http://prateekvjoshi.com/2013/10/20/c-vector-memory-release/
  // return if the frames is empty
  if (frames.size() < 1) {
    stack_info<<"After trimFrames, Instance #"<<InstanceNum<<" on "<<nodeName<<" is Empty."<<endl;
    return;
  }

  // modify the frames if more than 1 frame 
  else if (frames.size() > 1) 
    removeRedundantFrames(modules, nodeName);

  // Important: mark the instance that needs to glue fork* stacktraces
  if (!frames.empty()) { //very likely to be empty for insts in fork and preSpawn files
    vector<StackFrame>::reverse_iterator rsf_i = frames.rbegin();
    if ((*rsf_i).frameName == "thread_begin") { //not from the main thread of main node
      needGluePre = true;
      if (frames.size() > 1) {
        if (isForkStarWrapper((*(rsf_i+1)).frameName)) {
          needGlueFork = true;
          needGluePre =  false; //If we have fork, then we should ignore thread frame
          frames.pop_back(); //No need to keep thread_begin as we will depend on fork*wrapper
        }
      } //at least Two frames
    }
  } //at least One frame
  
  // print the new instance
  stack_info<<"After trimFrames, Instance #"<<InstanceNum<<" on "<<nodeName<<endl;
  printInstance_concise();
}
*/

void Instance::handleInstance(ModuleHash &modules, std::ostream &O, int InstanceNum, bool verbose)
{   
  cout<<"\nIn handleInstance for inst#"<<InstanceNum<<endl;
  
  // This is true in the case where we're at the last stack frame that can be parsed
  // Here, "bottom" comes from the growing style of stack since it goes down 
  // if main->foo->bar, then bar is the bottom stackframe
  bool isBottomParsed = true;
  vector<StackFrame>::iterator vec_SF_i;
  for (vec_SF_i = frames.begin(); vec_SF_i != frames.end(); vec_SF_i++) {
    if ((*vec_SF_i).toRemove == false) {
      // Get the module from the debugging information
      BlameModule *bm = modules[(*vec_SF_i).moduleName];   
      if (bm) {
        //BlameFunction *bf = bm->findLineRange((*vec_SF_i).lineNumber);
        //For CPU stack frames, it shows the real names, so we need to find its linkName
        BlameFunction *bf = bm->getFunction((*vec_SF_i).frameName); 
        if (bf) { //after previous trim&glue, it should come down here without trouble
          std::set<VertexProps *> blamedParams;
          if (bf->getBlamePoint() > 0) {
            cout<<"In function "<<bf->getName()<<" is BP="<<bf->getBlamePoint()<<endl;
            //blamePoint can be 0,1,2, why always set true(1) here ??
            bf->resolveLineNum(frames, modules, vec_SF_i, blamedParams, 
                    true, isBottomParsed, NULL, O);
            return; //once the resolveLineNum returns from a frame, then the whole instance done
          }
          else {
            cout<<"In function "<<bf->getName()<<" is not BP"<<endl;
            bf->resolveLineNum(frames, modules, vec_SF_i, blamedParams, 
                    false, isBottomParsed, NULL, O);
            return; //once the resolveLineNum returns from a frame, the whole instance is done
          }
          // TODO: Delete isBottomParsed here and everything below since we won't come here
          cout<<"THIS SHOULD NEVER EVER GETS PRINTED OUT"<<endl;
          isBottomParsed = false;
        }
        else {
          if (isBottomParsed == false) {
            cerr<<"Break in stack debugging info, BF is NULL"<<endl;
            isBottomParsed = true;
          }
          cerr<<"Error: BF NULL-- At Frame "<<(*vec_SF_i).frameNumber<<" "<<
            (*vec_SF_i).lineNumber<<" "<<(*vec_SF_i).moduleName<<" "<<(*vec_SF_i).frameName<<endl;
        }
      }
      else {
        if (isBottomParsed == false) {
          cerr<<"Break in stack debugging info, BM is NULL"<<endl;
          isBottomParsed = true;
        }
        
        cerr<<"Error: BM NULL-- At Frame "<<(*vec_SF_i).frameNumber<<" "<<
          (*vec_SF_i).lineNumber<<" "<<(*vec_SF_i).moduleName<<" "<<(*vec_SF_i).frameName<<endl;
      }
    }
    else
      cerr<<"Error: toRemove=true-- At Frame "<<(*vec_SF_i).frameNumber<<" "<<
        (*vec_SF_i).lineNumber<<" "<<(*vec_SF_i).moduleName<<" "<<(*vec_SF_i).frameName<<endl;
  }
}


void Instance::handleRuntimeInst(std::ostream &O, int InstanceNum, bool verbose)
{   
  cout<<"\nIn handleRuntimeInst for inst#"<<InstanceNum<<endl;
  //since it should only has one frame, we simply get the last frame in this instance
  StackFrame sf = *(frames.begin());

  O<<"FRAME# "<<sf.frameNumber<<" "<<sf.frameName; 
  //really not matter for the value below since we only need the frameName ("polling/comm_barrier")
  O<<" "<<sf.moduleName<<" "<<"chaple/runtime";
  O<<" "<<sf.lineNumber<<" 0 0 0"<<endl;
  //mimic user frames that have no blamed vars
  O<<"***No EV[EO, EP] found*** ["<<sf.frameName<<"]"<<endl;
}

