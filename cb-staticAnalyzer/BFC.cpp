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

#include "BFC.h"
#include "ModuleBFC.h"
#include "NodeProps.h"
#include "FunctionBFC.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include "NVPTXUtilities.h" // llvm-project/llvm/lib/Target/NVPTX
using namespace std;

ofstream struct_file;
bool gpu_code = false; //designator of whether we are running pass on gpu-side
set<string> kernels;
//populate blame relationship from externFuncions
void importSharedBFC(const char *path, ExternFuncBFCHash &externFuncInformation)
{
  ifstream bI(path);
  if (bI.is_open()) {
    string line;

    while (getline(bI, line)) {
      string buf;
      string name;
      stringstream ss(line);

      ss >> buf;
      name = buf; //assign name of the func
      if (name.empty()) //for space lines
        continue;

      ExternFunctionBFC *ef = new ExternFunctionBFC(name);

      while (ss >> buf) {
        ef->paramNums.insert(atoi(buf.c_str()));
      }
      externFuncInformation[name] = ef;

      if (externFuncInformation.size()==externFuncInformation.max_size()) {
        cerr<<"Damn it! we met the max size of externFunc hash"<<endl;
        return;
      }

#ifdef DEBUG_EXTERNFUNC
      cerr<<"Importing externFunc: "<<name<<", paramNums:"<<endl;
      set<int>::iterator pn_i;
      for(pn_i = ef->paramNums.begin(); pn_i != ef->paramNums.end(); pn_i++)
        cerr<<*pn_i<<" ";
      cerr<<endl;
#endif
    }

    bI.close();
  }
  else
    cerr<<"Shared lib path doesn't exist!"<<endl;
}

//Helper function
void exportUserFuncsInfo(FuncSigHash &kFI, ostream &O)
{
    FuncSigHash::iterator fsh_i;
    for (fsh_i = kFI.begin(); fsh_i != kFI.end(); fsh_i++) {
      string fname = fsh_i->first;
      FuncSignature *fs = fsh_i->second;
      string ftypeName = FunctionBFC::returnTypeName(fs->returnType, string(""));
      O<<"BEGIN FUNC"<<endl;
      O<<fname<<" "<<ftypeName<<endl;
      O<<fs->fLinkageName<<" "<<fs->ft<<endl; //for cuda

      vector<FuncFormalArg*>::iterator sa_i;
      for (sa_i=fs->args.begin(); sa_i!=fs->args.end(); sa_i++) {
        FuncFormalArg *arg = *sa_i;
        O<<arg->argIdx<<" "<<arg->name<<" "<<
            FunctionBFC::returnTypeName(arg->argType, string(""))<<endl;
      }
      
      O<<"END FUNC\n"<<endl;
    }
}


// check hasParOrDistCallsReal of each function to propagate that tag and produce warning 07/07/17
int calcFuncParDistTags (FuncSigHash &kFI)
{
    FuncSigHash::iterator fsh_i, fsh_e;
    int oldTotalTag = 0, newTotalTag = 0;

    for (fsh_i = kFI.begin(), fsh_e = kFI.end(); fsh_i != fsh_e; fsh_i++) {
      string fn = fsh_i->first;
      FuncSignature *fs = fsh_i->second;
        
      set<const char *, ltstr>::iterator si;
      for (si = fs->calleeNames.begin(); si != fs->calleeNames.end(); si++) {
        string callName(*si);
        
        if (kFI.count(callName)) {
          if (kFI[callName]->hasParOrDistCallsReal) {
            fs->hasParOrDistCallsReal = true;
            break;
          }
        }
      }
      newTotalTag += fs->hasParOrDistCallsReal;
    }

    // Iteratively check the hasParOrDistCallsReal
    while (newTotalTag != oldTotalTag) {
      oldTotalTag = newTotalTag; //reserve the old total tag val
      newTotalTag = 0; //recompute the new total tag val
      for (fsh_i = kFI.begin(), fsh_e = kFI.end(); fsh_i != fsh_e; fsh_i++) {
        string fn = fsh_i->first;
        FuncSignature *fs = fsh_i->second;
        
        set<const char *, ltstr>::iterator si;
        for (si = fs->calleeNames.begin(); si != fs->calleeNames.end(); si++) {
          string callName(*si);
          if (kFI.count(callName)) {
            if (kFI[callName]->hasParOrDistCallsReal) {
              fs->hasParOrDistCallsReal = true;
              break;
            }
          }
        }
        newTotalTag += fs->hasParOrDistCallsReal;
      }
    }

    return newTotalTag;
}


// specific for cuda
void populateKernelNames(string moduleName)
{
  string suffix = ".funcs";
  string middle = "-cuda-nvptx64-nvidia-cuda-sm_60"; //hard-coded here temporarily
  string gpu_func_file = moduleName.insert(moduleName.find_last_of("."), middle);
  gpu_func_file = "EXPORT/" + gpu_func_file + suffix;
  // for test only
  cout<<"gpu func file name: "<<gpu_func_file<<endl;

  ifstream ifs(gpu_func_file);
  if (!ifs.is_open())
    return;

  string line;
  char linkageName[50];
  int functionType;
    
  while (getline(ifs, line)) {
    if (line.find("BEGIN FUNC") != string::npos) {
      getline(ifs, line); // real name + return_type
      getline(ifs, line); // linkage name + func_type
      
      sscanf(line.c_str(), "%s %d", linkageName, &functionType);
      if (functionType == KERNEL)
        kernels.insert(string(linkageName));
    }
  }
}


/* Main pass that takes in a module and runs analysis on each function */
bool BFC::runOnModule(Module &M)
{
  cerr<<"IN run on module"<<endl;
  cerr<<"Module name: "<<M.getModuleIdentifier()<<"; target: "<<M.getTargetTriple()<<endl;
  if (M.getTargetTriple().find("nvidia-cuda") != string::npos) {
    gpu_code = true; //we are analyzing kernels and device funcs
    //read nvvm.annotations to extract all kernel functions
    //NamedMDNode *NMD = M.getNamedMetadata("nvvm.annotations");
    //if (NMD) {
    //  for (unsigned i=0, e=NMD->getNumOperands(); i!=e; i++) {
    //    const MDNode *elem = NMD->getOperand(i);
  }

  if (!gpu_code)  //only run on cpu code
    populateKernelNames(M.getModuleIdentifier());

  //for test only purpose
  cout<<"gpu_code = "<<gpu_code<<endl;
  for (auto k: kernels) cout<<k<<endl;

  ModuleBFC *bm = new ModuleBFC(&M); //address of a ref=address of the 
  //address of the object it refers to
  if (getenv("EXCLUSIVE_BLAME") != NULL)
    exclusive_blame = true;
  else
    exclusive_blame = false;
  ////////////////// GLOBAL VARIABLES ////////////////////////////////////////////
  vector<NodeProps *> globalVars;
  int globalNumber = 0; //global variable index
      
  Finder.processModule(M);//That's it, all DIDescriptors are stored in Vecs:
                      //CUs, SPs, GVs, TYs

  cout<<"Static Analyzing DIGlobalVariables"<<endl;
  for (Module::global_iterator gI = M.getGlobalList().begin(), 
    gE = M.getGlobalList().end(); gI != gE; ++gI) {
    
    GlobalVariable *gv = dyn_cast<GlobalVariable>(gI);
    SmallVector<DIGlobalVariableExpression *, 1> MDs;
    gv->getDebugInfo(MDs);
    if (gv->hasName() && gv->getName().str().find("llvm.")!=0 && !MDs.empty()) { //not intrinsic gv
      DIGlobalVariable *dgv = (*MDs.begin())->getVariable();
      if (!dgv->getFilename().empty() && dgv->getFilename().front() != '/') {
        string dgvName = dgv->getName().str();
        string gvName = gv->getName().str();
#ifdef DEBUG_P
        cout<<"Processing Global Variable: "<<gvName<<"; dgv name: "<<dgvName
            <<"; dgv linkage name: "<<dgv->getLinkageName().str()<<" "<<dgv->getFilename().str()
            <<" "<<dgv->getDirectory().str()<<endl;
#endif
        unsigned line_num = dgv->getLine();//the declaration linenum of this gv
        if (!gv->isConstant()) { //don't bother constants
#ifdef DEBUG_P
          cout<<"GV: "<<gvName<<" is real gv to be stored"<<endl;
#endif
          if (gv->hasInitializer()) {
            NodeProps* v = new NodeProps(globalNumber, gvName, line_num, gv->getInitializer());
            v->realName = dgvName;
            globalNumber++;
            globalVars.push_back(v);
          }
          else {
            NodeProps* v = new NodeProps(globalNumber, gvName, line_num, NULL);
            v->realName = dgvName;
            globalNumber++;
            globalVars.push_back(v);
          }
        }
      }
    }
  }


  cout<<"Static Analyzing DITypes"<<endl;
  for (DIType *dt : Finder.types()) {
    //naive way to keep analysis on user functions (others will have prefix absolute file path before the file name
    if (!dt->getFilename().empty() && dt->getFilename().front() != '/') { 
#ifdef DEBUG_P
      cout<<"Processing DIType: "<<dt->getName().str()<<" "<<dt->getFilename().str()
          <<" "<<dt->getDirectory().str()<<endl;
#endif
      bm->parseDITypes(dt);
    }
  }

  //  Make records of the information of external library functions  //
  ExternFuncBFCHash externFuncInformation;
    
  //other 3 files need to be adjusted since now return val => -1,  called func => -2
  //01/23/18, since ptx special registers are just lib funcs without params, we simply ignore those instructions for now
  char *cbPath = getenv("CUDA_BLAME_ROOT");
  string cuRtPath = string(cbPath) + "/cb-staticAnalyzer/SHARED/cuda_runtime.bs";
  importSharedBFC(cuRtPath.c_str(), externFuncInformation);
  string llInPath = string(cbPath) + "/cb-staticAnalyzer/SHARED/llvm_intrinsics.bs";
  importSharedBFC(llInPath.c_str(), externFuncInformation);
  //importSharedBFC("/homes/hzhang86/cudaBlamer/cB_StaticAnalysis/SHARED/nvvm.bs", externFuncInformation);
  //importSharedBFC("/export/home/hzhang86/cudaBlamer/cB_StaticAnalysis/SHARED/mpi.bs", externFuncInformation);
  //importSharedBFC("/export/home/hzhang86/cudaBlamer/cB_StaticAnalysis/SHARED/fortran.bs", externFuncInformation);
  //importSharedBFC("/export/home/hzhang86/cudaBlamer/cB_StaticAnalysis/SHARED/cblas.bs", externFuncInformation);
  
  // SETUP all the exports files 
  string blame_path("EXPORT/");
  string struct_path = blame_path;
  string calls_path = blame_path;
  string params_path = blame_path;
  string se_path = blame_path;
  string sea_path = blame_path;
  string alias_path = blame_path;
  string loops_path = blame_path;
  string conds_path = blame_path;
  string funcs_path = blame_path;
  
  string mod_name = M.getModuleIdentifier();
  string blame_extension(".blm");
  blame_path += mod_name;
  blame_path += blame_extension;
  ofstream blame_file(blame_path.c_str());
  
  string se_extension(".se");
  se_path += mod_name;
  se_path += se_extension;
  ofstream blame_se_file(se_path.c_str());
  
  string struct_extension(".structs");
  struct_path += mod_name;
  struct_path += struct_extension;
  //Here we make struct_file to be global in case we need to addin more later
  struct_file.open(struct_path.c_str());
  
  string calls_extension(".calls");
  calls_path += mod_name;
  calls_path += calls_extension;
  ofstream calls_file(calls_path.c_str());
  
  string params_extension(".params");
  params_path += mod_name;
  params_path += params_extension;
  ofstream params_file(params_path.c_str());
  
  string funcs_extension(".funcs");
  funcs_path += mod_name;
  funcs_path += funcs_extension;
  ofstream funcs_file(funcs_path.c_str());

  bm->exportStructs(struct_file);
    
  cout<<"Static Analyzing DISubprograms"<<endl;
  /////////////////// PRE PASS ////////////////////////////////////
  FuncSigHash knownFuncsInfo;

  for (Module::iterator fI = M.getFunctionList().begin(),
    fE = M.getFunctionList().end(); fI != fE; ++fI) {
      
    Function *F = &*fI;
    DISubprogram *dsp = F->getSubprogram();
    if (F->hasName() && !F->isIntrinsic() && dsp) { //not intrinsic function 
      //skip kernel functions in cpu code since they were analyzed in gpu code already
      if (!gpu_code) {
        if (kernels.find(F->getName().str()) != kernels.end())
          continue; 
      }
      //naive way to keep analysis on user functions (others will have prefix absolute file path before the file name
      if (!dsp->getFilename().empty() && dsp->getFilename().front() != '/') {
        string dspName = dsp->getName().str();
#ifdef DEBUG_P
        cout<<"Processing DISubprogram: "<<dspName<<" "<<dsp->getFilename().str()
            <<" "<<dsp->getDirectory().str()<<endl;
#endif
        if (!F->isDeclaration() && !dspName.empty()) {//GlobalValue->isDeclaration
#ifdef DEBUG_P
          cout<<"F: "<<dspName<<" is real func to be stored"<<endl;
#endif
          FuncSignature *funcSig = new FuncSignature();
          funcSig->fname = dspName;
          funcSig->fLinkageName = F->getName().str();//debug info's linkage name didn't always exist dsp->getLinkageName().str();
          funcSig->returnType = F->getReturnType();
          /*if (F->getCallingConv() == CallingConv::PTX_Kernel)
            funcSig->ft = KERNEL;
          else if (F->getCallingConv() == CallingConv::PTX_Device)
            funcSig->ft = DEVICE;
          else
            funcSig->ft = HOST;*/
          if (isKernelFunction(*F))
            funcSig->ft = KERNEL;
          else if (gpu_code)
            funcSig->ft = DEVICE;
          else
            funcSig->ft = HOST;

          int whichParam = 0;
          Function::arg_iterator af_i;
          for (af_i=F->arg_begin(); af_i!=F->arg_end(); af_i++) {
            Value *v = &*af_i;
            string argName = v->getName().str();
            Type *argType = v->getType();
            FuncFormalArg *arg = new FuncFormalArg();
            arg->name = argName;
            arg->argType = argType;
            arg->argIdx = whichParam;

            funcSig->args.push_back(arg);
            whichParam++;
          }
            
          knownFuncsInfo[funcSig->fname] = funcSig;
        } //not declaration
      } //name match and file name match
    } //F->hasName
  } //all subprograms

  exportUserFuncsInfo(knownFuncsInfo, funcs_file);
  //////////////// FIRST PASS ////////////////////////////////////

  int numMissing = 0;
  int numMEV = 0;
  int numMEV2 = 0;
  int numMEV3 = 0;

  //Generate relationships from first pass over functions
  for (Module::iterator fI = M.getFunctionList().begin(),
    fE = M.getFunctionList().end(); fI != fE; ++fI) {
      
    Function *F = &*fI;
    DISubprogram *dsp = F->getSubprogram();
    if (F->hasName() && !F->isIntrinsic() && dsp) {
      //skip kernel functions in cpu code since they were analyzed in gpu code already
      if (!gpu_code) {
        if (kernels.find(F->getName().str()) != kernels.end())
          continue; 
      }

      if (!dsp->getFilename().empty() && dsp->getFilename().front() != '/') {
        string dspName = dsp->getName().str();
          
        if (!F->isDeclaration() && !dspName.empty()) {
          // Run regular blame calculation
          FunctionBFC *fb = new FunctionBFC(F, knownFuncsInfo);
          fb->setModule(&M);
          fb->setModuleBFC(bm);
          fb->setModuleAndPathNames(dsp->getFilename().str(), dsp->getDirectory().str());
          fb->setRealName(dspName);
          /*if (F->getCallingConv() == CallingConv::PTX_Kernel)
             fb->ft = KERNEL;
          else if (F->getCallingConv() == CallingConv::PTX_Device)
            fb->ft = DEVICE;
            else
              fb->ft = HOST;*/
          if (isKernelFunction(*F))
            fb->ft = KERNEL;
          else if (gpu_code)
            fb->ft = DEVICE;
          else
            fb->ft = HOST;
          //change one line to 4 lines for easy parsing in addParser (in case template)
          calls_file<<"FUNCTION "<<fb->getSourceFuncName()<<endl;
          calls_file<<fb->getRealName()<<endl;
          calls_file<<fb->getModuleName()<<endl;
          calls_file<<fb->ft<<endl; 
#ifdef DEBUG_P  
          cout<<"Running firstPass on Func: "<<dspName<<endl;
#endif
          fb->firstPass(F, globalVars, externFuncInformation, 
                     blame_file, blame_se_file, calls_file, numMissing);
          calls_file<<"END FUNCTION "<<endl;

          int numMEVL = 0;
          int numMEV2L = 0;
          int numMEV3L = 0;
                        
          fb->moreThanOneEV(numMEVL, numMEV2L, numMEV3L);

          numMEV += numMEVL;
          numMEV2 += numMEV2L;
          numMEV3 += numMEV3L;

          fb->exportParams(params_file);
          params_file<<"NEVs - "<<numMEVL<<" "<<numMEV2L<<" " \
              <<numMEV3L<<" out of "<<knownFuncsInfo.size()<<endl;
          params_file<<endl;

          //store necessary information for each func to use after delete
          FuncSignature *fs = knownFuncsInfo[dspName];
          fs->calleeNames = fb->funcCallNames;
          fs->hasParOrDistCallsReal = fb->hasParOrDistCalls;
             
          //Hui 07/17/17: sizeof bf ~1.5kB, so we can keep fb for now
          delete(fb);
        }
      } 
    } 
  }


  struct_file<<"END STRUCTS"<<endl; //IMPORTANT! Moved here from exportStructs
  params_file<<"NEVs - "<<numMEV<<" "<<numMEV3<<" "<<numMEV2<<" out of "<< \
      knownFuncsInfo.size()<<endl;

  //we really should close files properly
  blame_file.close();
  blame_se_file.close();
  struct_file.close();
  calls_file.close();
  params_file.close();
  funcs_file.close();
 
  globalVars.clear();
  externFuncInformation.clear();
  knownFuncsInfo.clear();
  return false;
}
