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

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <algorithm>
#include <dirent.h> // for interating the directory

#include "BlameProgram.h"
#include "BlameFunction.h"
#include "BlameModule.h"
#include "util.h"
#include "Instances.h"
#include "Sample.h"
#include "CCT.h"

#include "BPatch.h"
#include "BPatch_process.h"
#include "BPatch_Vector.h"
#include "BPatch_statement.h"
#include "BPatch_image.h"
#include "BPatch_module.h"
#include "BPatch_point.h"
#include "BPatch_function.h"
using namespace std;
using namespace Dyninst;
 
instanceMap globInsts; //global var to store all CPU stacktraces
functionMap globFuncs; //global var to store all gpu funcs
sourceMap   globSrcs; //global var to store all source lctr info
sampleMap   globSamples; //total samples in GPU profile
CCTMap      globCCT; //calling context trees(roots are kernels)
//all processed samples, indexed by sample num
unordered_map<int, vector<Instance>> allStacks; 
unsigned int globNumSamples = 0;
// some forward-declarations
/*
//Hui for test: output file for finer stacktraces
ofstream stack_info;
nodeHash nodeIDNameMap; //Initialized in populateForkSamples
static void populateCompSamples(vector<Instance> &comp_instances, string traceName, string nodeName, BlameProgram &bp);
static void populateSamplesFromDirs(compInstanceHash &comp_instance_table, BlameProgram &bp);

static void populateCompSamples(vector<Instance> &comp_instances, string traceName, string nodeName, BlameProgram &bp)
{
  ifstream ifs_comp(traceName);
  string line;

  if (ifs_comp.is_open()) {
    getline(ifs_comp, line);
    int numInstances = atoi(line.c_str());
    cout<<"Number of instances from "<<traceName<<" is "<<numInstances<<endl;
 
    for (int i = 0; i < numInstances; i++) {
      Instance inst;
      int numFrames;
      inst.instNum = i;
      getline(ifs_comp, line);
      sscanf(line.c_str(), "%d", &numFrames);
    
      for (int j = 0; j < numFrames; j++) {
        StackFrame sf;
        getline(ifs_comp, line);
        stringstream ss(line);
        //populate stack frame info
        ss>>sf.frameNumber;
        ss>>sf.lineNumber;
        ss>>sf.moduleName;
        //special process for gpu frames since it has path
        sf.moduleName = getFileName(sf.moduleName);
        //we don't need to keep frames that are from non-user sources
        if (bp.blameModules.find(sf.moduleName) == bp.blameModules.end())
          continue;

        ss>>sf.frameName;
        //special process for cpu frames that show the real name 
        if (bp.realLinkNameMap.find(sf.frameName) != bp.realLinkNameMap.end()) {
          //the frameName is found in the real name set, means it's real name
          sf.frameName = bp.realLinkNameMap[sf.frameName];
        }
        //add this frame to the instance
        inst.frames.push_back(sf);
      }
    
      comp_instances.push_back(inst);
    }
    //Close the read file
    ifs_comp.close();
  }
  else
    cerr<<"Error: open file "<<traceName<<" fail"<<endl;
}


static void populateSamplesFromDirs(compInstanceHash &comp_instance_table, 
                                    BlameProgram &bp)
{

  DIR *dir;
  struct dirent *ent;
  string traceName;
  string dirName;
  string nodeName;
  size_t pos;

  dirName = "."; //02/06/18: for cuda, currently we only have single node
  if ((dir = opendir(dirName.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      if ((ent->d_type == DT_REG) && 
          (strstr(ent->d_name, "Input-") != NULL)) {
        traceName = dirName + "/" + std::string(ent->d_name);
        nodeName = traceName.substr(6); //cuz length of "Input-" is 6

        vector<Instance> comp_instances;
        populateCompSamples(comp_instances, traceName, nodeName, bp);
        if (comp_instance_table.count(nodeName) == 0)
          comp_instance_table[nodeName] = comp_instances; //push this node's sample
                                                        //info to the table
        else
          cerr<<"Error: This file was populated before: "<<traceName<<endl;
      }
    }
    closedir(dir);
  } 
  else 
    cerr<<"Error: couldn't open the directory "<<dirName<<endl;
  
  cout<<"Done populating all Input- files"<<endl;
}
*/

int populateFrames(Instance &inst, ifstream &ifs, BPatch_image *img, string inputFile, BlameProgram &bp)
{    
  char linebuffer[2000];
  ifs.getline(linebuffer,2000);
  string str(linebuffer);
  stringstream ss(str); // Insert the string into a stream
  //string buf;

  if (str.find("START")!=string::npos 
      || str.find("END")!=string::npos) {
    cerr<<"Shouldn't be here ! Not ADD_LINE! it's "<<str<<endl;
    return 1;
  }
  else {
    int frameNum;
    unsigned long address;  //16 bits in hex
    string frameName;
    //int stackSize = str.length()/22;
    //above: 15 for 32-bit, 18 for 48-bit, 22 for 64-bit system
    if (str.length() <= 0) {
      cerr<<"Null Stack Size"<<endl;
      return 2;
    }
    //we count the occurences of '\t' since each frame has 1 '\t'
    size_t stackSize = std::count(str.begin(), str.end(), '\t');
  
    for (int a = 0; a < stackSize; a++) {
      ss>>frameNum; // 1st: frameNum
      if (frameNum != a) {
        cerr<<"Missing a stack frame "<<frameNum<<" "<<a<<" in "
            <<inputFile<<" of inst#"<<inst.instNum<<endl;
        return 3;//break for loop, directly go to the next iter
      }

      ss>>std::hex>>address>>std::dec; // 2nd: address
      //sampled IP should points to the last instruction
      if (frameNum != 0)
        address = address - 1;  
      
      BPatch_Vector<BPatch_function*> funcs;
      //findFunction returns true if succeeds
      if (img->findFunction((Dyninst::Address)address, funcs)) { 
        frameName = funcs[0]->getName();
      }

      BPatch_Vector<BPatch_statement> sLines;
      //it can get the information associate with the address, 
      //the vector sLines contain pairs of filenames 
      //and line numbers that are associated with address
      img->getSourceLines(address, sLines); 
      //second cond added to remove all non-user funcs 01/29/18
      if (sLines.size() > 0 && sLines[0].lineNumber() != 0) { 
        StackFrame sf;
        sf.lineNumber = sLines[0].lineNumber();
        string fileN(sLines[0].fileName()); //contains the path
        string cleanFileN = getFileName(fileN);
        sf.moduleName = cleanFileN;
        sf.frameNumber = frameNum;
        sf.address = address;
        sf.frameName = frameName;
        //we only store frames with user line&file info
        if (bp.blameModules.find(sf.moduleName) 
            == bp.blameModules.end()) {
          continue;
        }
        //special process for cpu frames that show the real name 
        if (bp.realLinkNameMap.find(sf.frameName) 
            != bp.realLinkNameMap.end()) {
          //convert cpu func real names to link names
          sf.frameName = bp.realLinkNameMap[sf.frameName];
        }
        //after the previous 2 filers, the top frame should
        //be the cpu side kernel call with linkage name 
        inst.frames.push_back(sf);
      }
      //stop the stack walk till we met main in call_path
      if (frameName=="main")
        break; 
    }// end of for loop

    //trim the current frames in this instance
    //trimFrames(inst);
  }

  return 0;
}

void populateInstances(char *exeName, std::ifstream &ifs, string inputFile, BlameProgram &bp)
{
  BPatch bpatch;
  BPatch_image *img = NULL;
  BPatch_addressSpace *app = bpatch.openBinary(exeName);
  if (app == NULL) {
    cerr << "error in bpatch.openBinary" << endl;
    exit(-1);
  }

  img = app->getImage(); 
  if (img == NULL) {
    cerr << "error in app->getImage" << endl;
    exit(-1);
  }
  
  char linebuffer[2000];
  int inst_count = 0;
  //while(!ifs.eof())
  while(ifs.getline(linebuffer,1999)) {
    string str(linebuffer);
    stringstream ss(str); // Insert the string into a stream
    string buf;
   
    if (str.find("<----START") != string::npos) {
      Instance inst;
      inst.instNum = inst_count;
      inst_count++;
      
      ss>>buf; //buf = "<----START"
      ss>>buf; //buf = file name cpuStack
      ss>>inst.correlationID;
      
      int ret = populateFrames(inst, ifs, img, inputFile, bp); 

      ifs.getline(linebuffer,2000);//---->END
      string str2(linebuffer);
      if (str2.find("---->END") != string::npos && ret == 0) {
        if (globInsts.find(inst.correlationID) == globInsts.end())
          globInsts[inst.correlationID] = inst;
        else
          cerr<<"error: diff insts for same correlationID"<<endl;
      }
    }
  }// while loop
  
  cout<<"Done parsing a stack trace file: "<<inputFile<<endl;
}


void printInstances(string ofN, int which)
{
  cout<<"In printInstances--"<<which<<endl;
  if (which == 0) {
    instanceMap::iterator mi, me;
    std::ofstream ofs; //output file stream
    ofs.open(ofN);
    if (ofs.is_open()) {
      ofs<<globInsts.size()<<endl;

      //print out each processed samples
      vector<StackFrame>::iterator fi, fe;
      for (mi = globInsts.begin(), me = globInsts.end(); mi != me;
           mi++) {
        Instance inst = mi->second;
        ofs<<inst.frames.size()<<" "<<mi->first<<endl;
        for (fi = inst.frames.begin(), fe = inst.frames.end(); 
             fi != fe; fi++) {
          ofs<<(*fi).frameNumber<<" "<<(*fi).lineNumber<<" "
             <<(*fi).moduleName<<" "<<(*fi).frameName<<endl;
        } //all frames in each inst
      } //all instances
    } //if open file succeeds
  } //which == 0

  else if (which == 1) {
    int total = 0;
    unordered_map<int, vector<Instance>>::iterator mi, me;
    vector<Instance>::iterator vi, ve;
    for (mi = allStacks.begin(), me = allStacks.end(); 
         mi != me; mi++) {
      total += mi->second.size();
    }

    std::ofstream ofs; //output file stream
    ofs.open(ofN);
    if (ofs.is_open()) {
      ofs<<total<<endl;

      //print out each processed samples
      vector<StackFrame>::iterator fi, fe;
      for (mi = allStacks.begin(), me = allStacks.end(); mi != me;
           mi++) {
        for (vi = mi->second.begin(), ve = mi->second.end(); 
             vi != ve; vi++) {
          ofs<<vi->frames.size()<<" "<<vi->share<<" "
             <<vi->occurance<<endl;
          for (fi = vi->frames.begin(), fe = vi->frames.end(); 
               fi != fe; fi++) {
             ofs<<(*fi).frameNumber<<" "<<(*fi).lineNumber<<" "
                <<(*fi).moduleName<<" "<<(*fi).frameName<<endl;
          } //all frames in each inst
        } //all insts in each bucket
      } //all buckets
    } //if open file succeeds
  } //if which == 1
}


void populateSamples(std::ifstream &ifs)
{
  char linebuffer[256];
  unsigned int samp_count = 0;
  //while(!ifs.eof())
  while(ifs.getline(linebuffer,255)) {
    string str(linebuffer);
    stringstream ss(str); // Insert the string into a stream
    string buf;
   
    if (str.find("SOURCE_LOCATOR") != string::npos) {
      SourceLocator sl;
      ss>>buf; //buf = SOURCE_LOCATOR
      ss>>buf; //buf = Id
      ss>>sl.ID;
      ss>>buf; //buf = File
      ss>>sl.fileName; //complete path, may need "getFileName"
      ss>>buf; //buf = Line
      ss>>sl.line;
      
      if (globSrcs.find(sl.ID) == globSrcs.end())
        globSrcs[sl.ID] = sl;
      else
        cerr<<"err: diff srcLctr-same ID "<<sl.ID<<endl;
    }
    else if (str.find("FUNCTION") != string::npos) {
      GpuFunction gf;
      ss>>buf; //buf = FUNCTION
      ss>>buf; //buf = id
      ss>>gf.funcID; 
      ss>>buf; //buf = ctx
      ss>>gf.contextID;
      ss>>buf; //buf = moduleId
      ss>>gf.moduleID;
      ss>>buf; //buf = functionIndex
      ss>>gf.functionIndex;
      ss>>buf; //buf = name
      ss>>gf.name;

      if (globFuncs.find(gf.funcID) == globFuncs.end())
        globFuncs[gf.funcID] = gf;
      else
        cerr<<"err: diff gpuFunction-same ID "<<gf.funcID<<endl;
    }
    else if (str.find("INST_EXEC")!=string::npos || 
             str.find("PC_SAMPLING")!= string::npos) {
      unsigned int sid, cid, fid;
      ss>>buf; //buf = PC_SAMPLING
      ss>>buf; //buf = srcLctr
      ss>>sid;
      ss>>buf; //buf = corrId
      ss>>cid;
      ss>>buf; //buf = funcId
      ss>>fid;
      //ss>>buf; //buf = pc
      //ss>>sp.pcOffset; 
      samp_key_t sk(sid, cid, fid);

      if (globSamples.find(sk) == globSamples.end()) {
        Sample sp;
        sp.srcID = sid;
        sp.corrID = cid;
        sp.funcID = fid;
        //index of the 1st sample with this sample_key
        sp.sampleIdx = samp_count; 
        sp.occurance = 1; //first sample with this samp_key
        globSamples[sk] = sp;
      }
      else {
        globSamples[sk].occurance += 1; 
      }
      
      samp_count++;
    }
  }// while loop
  
  globNumSamples = samp_count; //globVar to keep total#samples
  cout<<"Done parsing GPU.txt, total #samples="<<samp_count<<endl;
  
  //////////////only for veryfication/////////////
  /*ofstream sk_file("samp_keys");
  if (sk_file.is_open()) {
    sampleMap::iterator smi, sme;
    for (smi = globSamples.begin(), sme = globSamples.end(); 
         smi != sme; smi++) {
      sk_file<<smi->first.corrID<<" "<<smi->first.srcID<<" "
             <<smi->first.funcID<<"\n";
    }
  }*/
}


//find all reversed call paths(callee->caller) from src to des
//in the calling context trees
void findAllPaths(string srcName, string desName, int &idx,
                  unordered_map<string, bool> &visited,
                  Instance &inst, int srcLine, string srcFile)
{
  //For test only purpose
  //cout<<"In findAllPaths: src="<<srcName<<", des="<<desName<<", idx="<<idx<<endl;

  StackFrame sf;
  sf.lineNumber = srcLine;
  sf.frameNumber = idx;
  sf.moduleName = getFileName(srcFile); //get clean filename
  sf.frameName = srcName;

  //Mark the current node and store it into the inst
  visited[srcName] = true;
  inst.frames.push_back(sf);
  idx++; //IMPORTANT, indicates the frameNumber
  
  //If current frame is same as destination
  if (srcName == desName) {
    Instance finishedInst = inst; //make a deep copy
    allStacks[inst.instNum].push_back(finishedInst);
    //return; //WE should NOT return here since we can have K->B(line 1)->D and K->B(line 2)->D
  }
  else {
    CCTNode *cn = globCCT[srcName];
    if (cn == NULL) {
      cout<<"Error: check why "<<srcName<<" is NULL in globCCT"<<endl;
      return;
    }
    vector<Caller*>::iterator vi, ve;
    for (vi = cn->parents.begin(), ve = cn->parents.end();
         vi != ve; vi++) {
      string vName = (*vi)->callerFunc->linkName;
      if (!visited[vName]) {
        int vLine = (*vi)->line;
        string vFile = (*vi)->file;
        //recursively call its callers
        findAllPaths(vName, desName, idx, visited, inst, 
                     vLine, vFile);
      }
    }
  }

  //Remove current frame and mark it as unvisited
  idx--;
  inst.frames.pop_back();
  visited[srcName] = false;
}


// glue CPU and GPU stacks
void glueCPUStacks() 
{
  cout<<"In glueCPUStacks"<<endl;
  
  unordered_map<int, vector<Instance>>::iterator mi, me;
  vector<Instance>::iterator vi, ve;
  //allStacks inst#(key) == globSamples samp#
  for (mi = allStacks.begin(), me = allStacks.end(); mi != me; mi++) {
    Instance cStack = globInsts[(mi->second)[0].correlationID];
    //depends on the shape of its top frame
    string kLN = cStack.frames[0].frameName; //cpu side launched kernel name
    double numAmbiguity = mi->second.size();
    cStack.frames.erase(cStack.frames.begin()); 
    for (vi = mi->second.begin(), ve = mi->second.end(); vi != ve; vi++) {
      string bFN = vi->frames.back().frameName; //bFN: bottom frame name
      if (globCCT.count(bFN) && globCCT[bFN]->ft != KERNEL) {
        cerr<<"Error: bottom frame of GPU stack "<<bFN<<" isn't a kernel!"<<endl;
        continue;
      }
      //for cases where only CPU side stacks needed
      if (kLN != bFN) { 
        //cout<<"kLN: "<<kLN<<" - bFN: "<<bFN<<endl;
        continue;
      }
    
      //vi->frames.pop_back(); //depends on the shape of cStack's top frame
      vi->frames.insert(vi->frames.end(), cStack.frames.begin(), cStack.frames.end());
      vi->share = 1.0/numAmbiguity; //set the share of each inst
    }
  }
}


void genCompleteStacks()
{
  cout<<"In genCmpleteStacks"<<endl;
  int numEC = 0, numES = 0, numEF = 0;
  sampleMap::iterator smi, sme;
  for (smi = globSamples.begin(), sme = globSamples.end(); 
       smi != sme; smi++) {
    //we need to first check whether the corresponding cID, fID, sID exists
    if (globInsts.find(smi->first.corrID) == globInsts.end()) {
      cerr<<"Error: sample#"<<smi->second.sampleIdx<<" corrID "
        <<smi->first.corrID<<" not found in globInsts! \
        We drop this sample for now"<<endl;
      numEC++;
      continue;
    }
    if (globSrcs.find(smi->first.srcID) == globSrcs.end()) {
      //cerr<<"Error: sample#"<<smi->second.sampleIdx<<" srcID "<<smi->first.srcID<<" not found in globSrcs! We drop this sample for now"<<endl;
      numES++;
      Instance tmp = globInsts[smi->first.corrID];
      tmp.instNum = smi->second.sampleIdx;
      tmp.occurance = smi->second.occurance;
      allStacks[smi->second.sampleIdx].push_back(tmp); 
      continue;
    }
    if (globFuncs.find(smi->first.funcID) == globFuncs.end()) {
      //cerr<<"Error: sample#"<<smi->second.sampleIdx<<" funcID "<<smi->first.funcID<<" not found in globSrcs! We drop this sample for now"<<endl;
      numEF++;
      Instance tmp = globInsts[smi->first.corrID];
      tmp.instNum = smi->second.sampleIdx;
      tmp.occurance = smi->second.occurance;
      allStacks[smi->second.sampleIdx].push_back(tmp); 
      continue;
    }
    
    //get all 3 necessary information
    Instance cStack = globInsts[smi->first.corrID];
    SourceLocator srcInfo = globSrcs[smi->first.srcID];
    GpuFunction gFunc = globFuncs[smi->first.funcID];
    //suppose the first frame of CPU stacktrace is kernel launch
    string kernelName = cStack.frames[0].frameName; //kernel name
    string deviceName = gFunc.name; //device func manggled name
    
    //Now we need to check whether the device/kernel funcs exist
    if (globCCT.find(kernelName) == globCCT.end()) {
      cerr<<"Error: CPU top frame "<<kernelName
          <<" not found in globCCT!"<<endl;
      continue;
    }
    if (globCCT[kernelName]->ft != KERNEL) {
      cerr<<"Error: top frame of CPU stack "<<kernelName
          <<" isn't a kernel!"<<endl;
      continue;
    }
    if (globCCT.find(deviceName) == globCCT.end()) {
      //cout<<"Check: GPU device func "<<deviceName \
          <<" not found in globCCT! We can only blame it to " \
          <<"the kernel "<<kernelName<<" level now."<<endl;
      Instance tmp = globInsts[smi->first.corrID];
      tmp.instNum = smi->second.sampleIdx;
      tmp.occurance = smi->second.occurance;
      allStacks[smi->second.sampleIdx].push_back(tmp); 
      continue;
    }
    //create an instance for this sample with filled 1st frame
    Instance inst;
    inst.instNum = smi->second.sampleIdx;
    inst.occurance = smi->second.occurance;
    inst.correlationID = smi->second.corrID;
    int idx = 0; 
    unordered_map<string, bool> visited;
    CCTMap::iterator ci, ce;
    for (ci = globCCT.begin(), ce = globCCT.end(); ci != ce; ci++)
      visited[ci->first] = false;
    
    //find all call paths, Using DFS, only works for a DAG
    //which means we cannot have recursion in gpu stack
    findAllPaths(deviceName, kernelName, idx, visited, 
                 inst, srcInfo.line, srcInfo.fileName);
  }

  cout<<"Total unresolved samples: "<<numEC+numES+numEF<<"; EC "
      <<numEC<<", ES "<<numES<<", EF "<<numEF<<endl;
  //NOW let calculates the ambiguity:  
  //(#samples that have >1 call paths /#raw samples) * 100%
  int ambiCount = 0;
  for (smi = globSamples.begin(), sme = globSamples.end(); 
       smi != sme; smi++) {
    int samp_num = smi->second.sampleIdx;
    if (allStacks.find(samp_num) == allStacks.end()){
      cerr<<"Error: sample#"<<samp_num<<" didn't find complete callpath back to its kernel"<<endl;
      continue;
    }

    if (allStacks[samp_num].size() > 1) {
      ambiCount += smi->second.occurance;
    }
  }
  double percentage = (double)(globNumSamples - ambiCount)/ 
                      (double)globNumSamples * 100;
  printf("The coverage = %3.3f\%\n", percentage);

  //glue CPU stacks
  glueCPUStacks();
}


// <altMain> <exe Name(not used)> <config file name> 
int main(int argc, char** argv)
{ 
  if (argc != 5) {
    cerr<<"Wrong Number of Arguments! "<<argc<<endl;
    cerr<<"Usage: "<<endl;
    cerr<<"AltParser <target binary> post_process_config.txt CPU.txt GPU.txt"<<endl;
    exit(0);
  }
  
  bool verbose = false;
  
  fprintf(stderr,"START - ");
  my_timestamp();
  
  BlameProgram bp;
  bp.parseConfigFile(argv[2]);
  
  //std::cout<<"Parsing structs "<<std::endl;
  bp.parseStructs();
  //std::cout<<"Printing structs"<<std::endl;
  //bp.printStructs(std::cout);
  
  //std::cout<<"Parsing side effects "<<std::endl;
  bp.parseSideEffects();
  
  //std::cout<<"Printing side effects(PRE)"<<std::endl;
  //bp.printSideEffects();
  
  //std::cout<<"Calc Rec side effects"<<std::endl;
  bp.calcRecursiveSEAliases();
  
  //std::cout<<"Printing side effects(POST)"<<std::endl;
  //bp.printSideEffects();
  //02/15/18
  //It has to be called before "grabSampledModules" since
  //it has all source locator info that indicates modules
  string inputFile = string(argv[4]); //currently GPU.txt
  ifstream ifs_gpu(inputFile);
  if (ifs_gpu.is_open()) {
    populateSamples(ifs_gpu); 
    ifs_gpu.close(); //CLOSE the file
  }
  else
    cerr<<"Failed openning "<<inputFile<<endl;

  //std::cout<<"Grab used modules "<<std::endl;
  //bp.grabUsedModulesFromDir();
  bp.grabSampledModules();
  
  //std::cout<<"Parsing program"<<std::endl;
  bp.parseProgram();

  //std::cout<<"Parsing call files."<<std::endl; //for cuda
  bp.parseCallFiles();
  
  //std::cout<<"Adding implicit blame points."<<std::endl;
  bp.addImplicitBlamePoints();

  //retrieve pids from PPA 01/18/17
  //bp.resolvePidsFromPPAs(); 
  
  //std::cout<<"Printing parsed output to 'output.txt'"<<std::endl;
  //ofstream outtie("output.txt");
  //bp.printParsed(outtie);
  
  // parse compute file then output Input-compute file
  inputFile = string(argv[3]); //currently CPU.txt
  ifstream ifs_cpu(inputFile);
  if (ifs_cpu.is_open()) {
    populateInstances(argv[1], ifs_cpu, inputFile, bp); 
    ifs_cpu.close(); //CLOSE the file
  }
  else
    cerr<<"Failed openning "<<inputFile<<endl;
  //for debug, check CPU side stacktrace
  printInstances("RAWCPU-stacks", 0);
  //now generate the complete calling context from GPU to CPU
  genCompleteStacks();
  //for debug, check CPU side stacktrace
  printInstances("PROCESSED-stacks", 1);
  
  //bp.calcSideEffects();
  //std::cout<<"Populating samples."<<std::endl;
  fprintf(stderr,"SAMPLES - ");
  my_timestamp();

  //stack_info.open("stackDebug", ofstream::out);
  //compInstanceHash  comp_instance_table;
  // Import sample info from all Input- files
  //populateSamplesFromDirs(comp_instance_table, bp);

  //compInstanceHash::iterator ch_i; //vec_I_i can be reused from above
  vector<Instance>::iterator vi, ve;
  unordered_map<int, vector<Instance>>::iterator mi, me;
  //Directly handle each instance
  char buffer[128];
  gethostname(buffer, 127);
  string outName(buffer);
  outName = "PARSED_" + outName;
  ofstream gOut(outName.c_str());
  int iCounter = 0;
  //int numUnresolvedInsts = 0;
  int emptyInst = 0;

  for (mi = allStacks.begin(), me = allStacks.end(); mi != me; mi++) {
    for (vi = mi->second.begin(), ve = mi->second.end(); vi != ve; vi++) {
      gOut<<"---INSTANCE "<<iCounter<<" "<<(*vi).share<<" "
          <<(*vi).occurance<<" ---"<<endl;
      // Result check, invalid samples end up in pthread_spin_lock or others
      if ((*vi).frames.empty())
        emptyInst++;
      else { //only handle instance while it's not empty
        // The following 'if' block was for Chapel, place reserved for cuda
        if ((*vi).frames.back().frameName=="polling" 
            || (*vi).frames.back().frameName=="chpl_comm_barrier"
            || (*vi).frames.back().frameName=="pthread_spin_lock") {
          if ((*vi).frames.size()==1)
            (*vi).handleRuntimeInst(gOut, iCounter, verbose);
          else {
            cerr<<"Weird: "<<(*vi).frames.back().frameName<<"!"<<endl;
            //numUnresolvedInsts++;
            continue;
          }
        }
        // handle the perfect instance now !
        (*vi).handleInstance(bp.blameModules, gOut, iCounter, verbose);
      }

      gOut<<"$$$INSTANCE "<<iCounter<<"  $$$"<<endl; 
      iCounter++;
    }
  }

  cout<<"#emptyInst = "<<emptyInst<<" on "<<buffer<<endl;
      
  fprintf(stderr,"DONE - ");
  my_timestamp();
}
