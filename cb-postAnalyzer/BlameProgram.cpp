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

#include "BlameProgram.h"
#include "BlameModule.h"
#include "BlameFunction.h"
#include "util.h"
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <dirent.h> //for iterating the directory
using namespace std;

extern instanceMap globInsts; //all CPU stacktraces
extern functionMap globFuncs; //all gpu funcs
extern sourceMap   globSrcs; //all source lctr info
extern sampleVec   allSamples; //total samples in GPU profile
extern CCTMap      globCCT; //calling context trees
//all processed samples, indexed by sample num
extern unordered_map<int, vector<Instance>> allStacks; 


BlameModule *BlameProgram::findOrCreateModule(const char *modName)
{
  //cout<<"In findOrCreateModule for "<<modName<<endl;
  BlameModule *bm = NULL;
  
  //bm = blameModules[modName];
  
  string modNameStr(modName);
  if (blameModules.count(modNameStr) == 0){
    //cout<<"Creating new blame module for "<<modName<<endl;
    bm = new BlameModule();
    string s(modName);
    bm->setName(s);
    blameModules[bm->getName()] = bm;
  }
  else {
    bm = blameModules[modNameStr];
    //cout<<bm->getName()<<" already created "<<endl;
  }
  
  return bm;
}


void BlameProgram::addImplicitBlamePoint(const char * checkName)
{
  BlameFunction * bf = NULL;
  FunctionHash::iterator fh_i_check;  // blameFunctions;
    string checkNameStr(checkName);
    fh_i_check = blameFunctions.find(checkNameStr);
  
  if (fh_i_check != blameFunctions.end())
  {
     bf = blameFunctions[checkNameStr];
     bf->setBlamePoint(IMPLICIT);
  }
}

void BlameProgram::addImplicitBlamePoints()
{
  // TODO: Get this from config file
  addImplicitBlamePoint("HPL_pdtest");
  addImplicitBlamePoint("HPL_pdgesv");
  addImplicitBlamePoint("HPL_pdgesv0");
  addImplicitBlamePoint("callBottom");
}

// resolve pids from PPAs by checking the EVs of callNodes for each bf
void BlameProgram::resolvePidsFromPPAs()
{
  FunctionHash::iterator fh_i;  // blameFunctions;
  for (fh_i = blameFunctions.begin(); fh_i != blameFunctions.end(); fh_i++) {
    string bfn = (*fh_i).first;
    BlameFunction *bf = (*fh_i).second;
    if (bf) {
      cout<<"In resolvePidsFromPPAs for func: "<<bfn<<endl;
      vector<VertexProps *>::iterator vi;
      for (vi = bf->callNodes.begin(); vi != bf->callNodes.end(); vi++) {
        VertexProps *cn = *vi;
        size_t pos = cn->name.find("--");
        if (pos != string::npos) { //make sure it's a callNode
          string cnfn = (*vi)->name.substr(0, pos);
          cout<<"In resolvePidsFromPPAs for "<<bfn<<"'s callNode: "<<cn->name
              <<", check cnbf: "<<cnfn<<endl;
          if (blameFunctions.find(cnfn) != blameFunctions.end()) { 
            BlameFunction *cnbf = blameFunctions[cnfn];
            if (cnbf) { //we found the bf pointed by thie callNode
              vector<FuncCall *>::iterator vec_fc_i;
              for (vec_fc_i = cn->calls.begin(); vec_fc_i != cn->calls.end(); vec_fc_i++) {
                FuncCall *fc = *vec_fc_i;
                if (fc->Node->isPid == false) { //we only care when real param wasn't pid
                  vector<VertexProps *>::iterator vi2;
                  for (vi2 = cnbf->exitVariables.begin(); vi2 != cnbf->exitVariables.end(); vi2++) {
                    VertexProps *ev = *vi2;
                    if (ev->eStatus >= EXIT_VAR_PARAM) {
                      int whichParam = ev->eStatus - EXIT_VAR_PARAM;
                      if (ev->isPid && whichParam == fc->paramNumber) {
                        fc->Node->isPid = true;
                        bf->resolvePAForParamNode(fc->Node);
                      }
                    }
                  } //end of all EVs of cnbf
                } //if fc->Node->isPid=false
              } //end of all fc
            } //if callnodeblamefunction exists
          }
        } //if real callNode name
      } //end of all callNodes
    }
  } //end of all bf in this bp
}

//Not used anymore since we don't produce Input- files now 02/15/18
void BlameProgram::grabUsedModulesFromDir()
{
  DIR *dir;
  struct dirent *ent;
  string traceName;
  string dirName;
  size_t pos;

  //dirName = "COMPUTE";
  dirName = "."; //02/06/18: for cuda, currently we only have single node
  if ((dir = opendir(dirName.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      if ((ent->d_type == DT_REG) && 
          (strstr(ent->d_name, "Input-") != NULL)) {
        traceName = dirName + "/" + string(ent->d_name);

        grabSampledModules();
      }
    }
    closedir(dir);
  }
}

void BlameProgram::grabSampledModules()
{
  sourceMap::iterator si, se;
  string moduleName;
  for (si = globSrcs.begin(), se = globSrcs.end(); si != se; si++)
  {
    moduleName = si->second.fileName;
    if (!moduleName.empty()) {
      //added on 02/06/18,cuz gpu frames have paths
      moduleName = getFileName(moduleName);
      std::pair<set<string>::iterator,bool> ret;
      ret = sampledModules.insert(moduleName);
      if (ret.second) //returns true if inserted succ
        cout<<moduleName<<" inserted in sampledModules"<<endl;
    }
  }
}

/*void BlameProgram::parseConfigFile_OA(const char * path)
{
  ifstream bI(path);
  
  string numExportFilesStr;
  string tmpStr;
  string numStructFilesStr;
  string numSEFilesStr;
  string numLoopFilesStr;
  
  getline(bI, numExportFilesStr);
  int nEF = atoi(numExportFilesStr.c_str());
  for (int a = 0; a < nEF; a++)
  {
    getline(bI, tmpStr);
    blameExportFiles.push_back(tmpStr);
  }
  
  getline(bI, numStructFilesStr);
  int nSF = atoi(numStructFilesStr.c_str());
  for (int a = 0; a < nSF; a++)
  {
    getline(bI, tmpStr);
    blameStructFiles.push_back(tmpStr);
  }
  
  getline(bI, numSEFilesStr);
  int nSEF = atoi(numSEFilesStr.c_str());
  for (int a = 0; a < nSEF; a++)
  {
    getline(bI, tmpStr);
    blameSEFiles.push_back(tmpStr);
  }
  
  getline(bI, numLoopFilesStr);
  int nLoop = atoi(numLoopFilesStr.c_str());
  for (int a = 0; a < nLoop; a++)
  {
    getline(bI, tmpStr);
    blameLoopFiles.push_back(tmpStr);
  }
  
}*/



void BlameProgram::parseConfigFile(const char * path)
{
  ifstream bI(path);
  
  string numExportFilesStr;
  string tmpStr;
  string numStructFilesStr;
  string numSEFilesStr;
  string numCallFilesStr;
  
  getline(bI, numExportFilesStr);
  int nEF = atoi(numExportFilesStr.c_str());
  for (int a = 0; a < nEF; a++) {
    getline(bI, tmpStr);
    blameExportFiles.push_back(tmpStr);
  }
  
  getline(bI, numStructFilesStr);
  int nSF = atoi(numStructFilesStr.c_str());
  for (int a = 0; a < nSF; a++) {
    getline(bI, tmpStr);
    blameStructFiles.push_back(tmpStr);
  }
  
  getline(bI, numSEFilesStr);
  int nSEF = atoi(numSEFilesStr.c_str());
  for (int a = 0; a < nSEF; a++)
  {
    getline(bI, tmpStr);
    blameSEFiles.push_back(tmpStr);
  }
  
  getline(bI, numCallFilesStr);
  int nCF = atoi(numCallFilesStr.c_str());
  for (int a = 0; a < nCF; a++) {
    getline(bI, tmpStr);
    blameCallFiles.push_back(tmpStr);
  }
}

/*
 //TODO: Obviously fix this
 bool BlameProgram::isVerbose()
 {
 cerr<<"IN VB"<<endl;
 
 if (strcmp(boolVerbose.c_str(),"VERBOSE")==0)
 {
 cerr<<"Verbose!!!"<<endl;
 return true;
 }
 else
 {
 cerr<<"Not Verbose!!!"<<endl;
 return false;
 }
 
 }
 */


void BlameProgram::addFunction(BlameFunction *bf)
{
  //cout<<"Adding bf: "<<bf->getName()<<" to bm: "<<bf->getModuleName()<<endl;

  BlameModule *bm = findOrCreateModule(bf->getModuleName().c_str());
  bm->addFunction(bf); //add to FunctionHash blameFunctions in this bm
  //bm->addFunctionSet(bf); //removed by Hui 03/29/16, not using funcBySet anymore
  
  blameFunctions[bf->getName()] = bf; //add to blameFunctions in this bp
  
}

void BlameProgram::calcSideEffects()
{
  FunctionHash::iterator fh_i;
  for (fh_i = blameFunctions.begin(); fh_i != blameFunctions.end(); fh_i++)
  {
    BlameFunction * bf = (*fh_i).second;
    if (bf)
      bf->calcSideEffectDependencies();
  }
}

void BlameProgram::parseStructs()
{
  std::vector<string>::iterator vec_str_i;
  for (vec_str_i = blameStructFiles.begin(); vec_str_i != blameStructFiles.end(); 
       vec_str_i++) {
    ifstream bI((*vec_str_i).c_str());
    cout<<"Parsing .structs file "<<(*vec_str_i)<<endl;
    string line;
    
    // BEGIN STRUCT (or END STRUCTS)
    getline(bI, line);
    
    while (line.find("BEGIN STRUCT") != string::npos) {
      StructBlame *sb = new StructBlame();
      sb->parseStruct(bI);
      blameStructs[sb->structName] = sb;
      
      // BEGIN STRUCT (or END STRUCTS)
      getline(bI, line);
    }
  }
}

void BlameProgram::printStructs(std::ostream & O)
{
  StructHash::iterator vec_sb_i;
  
  //FunctionBlame * fb;
  
  for (vec_sb_i = blameStructs.begin(); vec_sb_i != blameStructs.end(); vec_sb_i++)
  {
    O<<endl<<endl;
    StructBlame * sb = (*vec_sb_i).second;
    O<<"Struct "<<sb->structName<<endl;
    O<<"Module Path "<<sb->modulePathName<<endl;
    O<<"Module Name "<<sb->moduleName<<endl;
    O<<"Line Number "<<sb->lineNum<<endl;
    
    FieldHash::iterator vec_sf_i;
    for (vec_sf_i = sb->fields.begin(); vec_sf_i != sb->fields.end(); vec_sf_i++)
    {
      StructField * sf = (*vec_sf_i).second;
      if (sf == NULL)
        continue;
      O<<"   Field # "<<sf->fieldNum<<" ";
      O<<", Name "<<sf->fieldName<<" ";
      //cout<<", Type "<<fb->returnTypeName(sf->llvmType, string(" "));
      O<<endl;
    }
  }
}



void StructBlame::parseStruct(ifstream & bI)
{
  string line;
  
  /////////
  //BEGIN_S_NAME
  getline(bI, line);
  
  // Actual Nmae
  getline(bI, line);
  structName = line;
  
  // END S_NAME
  getline(bI, line);
  
  
  /////////
  //BEGIN_M_PATH
  getline(bI, line);
  
  // Module Path
  getline(bI, line);
  modulePathName = line;
  
  // END M_PATH
  getline(bI, line);
  
  
  
  
  /////////
  //BEGIN_M_NAME
  getline(bI, line);
  
  // Module Nmae
  getline(bI, line);
  moduleName = line;
  
  // END M_NAME
  getline(bI, line);
  
  
  
  /////////
  //BEGIN LINENUM
  getline(bI, line);
  
  // LineNum string
  getline(bI, line);
  lineNum = atoi(line.c_str());
  
  // END LINENUM
  getline(bI, line);
  
  
  // Start of fields
  // BEGIN FIELDS
  getline(bI, line);
  
  getline(bI, line);
  // BEGIN FIELD (or END FIELDS)
  while(line.find("END FIELDS") == string::npos)
  {
    //cout<<line<<endl;
    // BEGIN F_NUM (or END FIELD which would be an error)
    getline(bI, line);
    if (line.find("END FIELD") != string::npos)
    {
      // BEGIN FIELD (or END FIELDS)
      getline(bI, line);
      continue;
    }
    
    // BEGIN F_NUM
    getline(bI, line);
    int fNum = atoi(line.c_str());
    StructField *sf = new StructField(fNum);
    fields[fNum] = sf;
    
    // END F_NUM
    getline(bI, line);
    
    // BEGIN F_NAME
    getline(bI, line);
    
    // field name
    getline(bI, sf->fieldName);
    
    // END F_NAME
    getline(bI, line);
    
    // BEGIN F_TYPE
    getline(bI, line);
    
    // field type
    getline(bI, sf->fieldType);
    
    // END F_TYPE
    getline(bI, line);

    
    
    // END FIELD
    getline(bI, line);
    
    // BEGIN FIELD (or END FIELDS)
    getline(bI, line);
  }
  
  // END STRUCT
  getline(bI, line);
  
}


void BlameFunction::calcRecursiveSEAliasesRecursive(std::set<BlameFunction *> & visited)
{
  if (seAliasCalc)
    return;
  
  //cout<<"Entering calcRSEAR for "<<linkName<<endl;
  
  
  seAliasCalc = true;
  visited.insert(this);
  
  std::vector<SideEffectCall *>::iterator vec_sec_i;
  
  
  if (sideEffectCalls.size() > 0)
  {
    //cout<<"Calls:"<<endl;
    for (vec_sec_i = sideEffectCalls.begin(); vec_sec_i != sideEffectCalls.end();
         vec_sec_i++)
    {
      SideEffectCall * sec = *vec_sec_i;
      //cout<<"C: "<<sec->callNode<<endl;
      
      size_t endChar = sec->callNode.find("--");
      if (endChar != string::npos)
      {
        string callTrunc = sec->callNode.substr(0, endChar);
        //BlameFunction * bf = BP->blameFunctions[callTrunc.c_str()];
        
        BlameFunction * bf = NULL;
        FunctionHash::iterator fh_i_check;  // blameFunctions;
        fh_i_check = BP->blameFunctions.find(callTrunc);
        if (fh_i_check != BP->blameFunctions.end())
        {
          bf = BP->blameFunctions[callTrunc];
        }
        
        if (bf == NULL )
        {
          if (callTrunc.find("tmp") == string::npos)
            {
              // TODO: Look into this
              //cerr<<"Call "<<callTrunc<<" not found!"<<endl;
            //cout<<"Call "<<callTrunc<<" not found!"<<endl;
            }
        }
        else
        {
          if (this->getName() == bf->getName())
          {
            //cout<<"RECURSIVE calcSREAR call"<<endl;
            continue;
          }
        
          bf->calcRecursiveSEAliasesRecursive(visited);
          
          if (bf->sideEffectAliases.size() > 0)
          {
            //cout<<"Aliases:"<<endl;
            std::vector<SideEffectAlias *>::iterator vec_sea_i;
            for (vec_sea_i = bf->sideEffectAliases.begin(); vec_sea_i != bf->sideEffectAliases.end();
                 vec_sea_i++)
            {
              SideEffectAlias * sea = *vec_sea_i;
              //cout<<"SideEffectAlias(REC) "<<sea<<endl;
              //cout<<"A:(1) "<<endl;
              std::set<SideEffectParam *>::iterator set_s_i;        // callee iterator
              std::vector<SideEffectParam *>::iterator vec_s_i_caller; // caller iterator
              std::set<SideEffectParam *>::iterator set_s_i_2;        // callee iterator stepper
              //std::set<SideEffectParam *>::iterator set_s_i_caller_2; // caller iterator stepper
              
              
              SideEffectParam * sepCaller = NULL;
              SideEffectParam * sepCaller2 = NULL;
              string localVarName("LOCAL-VAR");
              
              
              for (set_s_i = sea->aliases.begin(); set_s_i != sea->aliases.end(); set_s_i++)
              {
                SideEffectParam * sep = *set_s_i;
                //cout<<"A:(2) "<<sep->paramName<<endl;

                //int calleeParam = sep->paramNumber;
                
                
                SideEffectParam * sep2 = NULL;
                
                set_s_i_2 = set_s_i;
                set_s_i_2++;
                for (; set_s_i_2 != sea->aliases.end(); set_s_i_2++)
                {
                  sep2 = *set_s_i_2;
                  //cout<<"A:(3) "<<sep->paramName<<" "<<sep2->paramName<<endl;

                  
                  //if (sep->paramName == sep2->paramName)
                    //continue;
                  
                  for (vec_s_i_caller = sec->params.begin(); vec_s_i_caller != sec->params.end(); vec_s_i_caller++)
                  {
                    //cout<<"A:(4) "<<(*vec_s_i_caller)->paramName<<endl;
                    //cout<<"P: "<<sep->paramNumber<<" "<<sep->paramName<<endl;
                    
                    
                    
                    if (sep->paramNumber == (*vec_s_i_caller)->calleeNumber)
                      sepCaller = (*vec_s_i_caller);
                    else if (sep2->paramNumber == (*vec_s_i_caller)->calleeNumber)
                      sepCaller2 = (*vec_s_i_caller);
                  }
                  
                }
                
                if (sep2 == NULL)
                  continue;
                
                if (sepCaller == NULL)
                {
                  sepCaller = new SideEffectParam();
                  sepCaller->paramName = localVarName;
                  sepCaller->calleeNumber = 0;
                  sepCaller->paramNumber = 0;
                }
                
                if (sepCaller2 == NULL)
                {
                  sepCaller2 = new SideEffectParam();
                  sepCaller2->paramName = localVarName;
                  sepCaller2->calleeNumber = 0;
                  sepCaller2->paramNumber = 0;
                }                
                
                
                if ((sepCaller && sepCaller2) && sepCaller->paramNumber != sepCaller2->paramNumber)
                {
                  size_t dot1 = sep->paramName.find_first_of('.');
                  size_t dot2 = sep2->paramName.find_first_of('.');
                  
                  string ali1 = sepCaller->paramName;
                  string ali2 = sepCaller2->paramName;
                  
                  if (dot1 != string::npos)
                  {
                    ali1 += sep->paramName.substr(dot1);
                  }
                  
                  if (dot2 != string::npos)
                  {
                    ali2 += sep2->paramName.substr(dot2);
                  }
                  
                  //cout<<"Alias relation between "<<ali1<<" and "<<ali2<<endl;
                  
                  SideEffectAlias * seaRes = new SideEffectAlias();
                  SideEffectParam * sepRes1 = new SideEffectParam();
                  SideEffectParam * sepRes2 = new SideEffectParam();
                  
                  sepRes1->paramNumber = sepCaller->paramNumber;
                  sepRes2->paramNumber = sepCaller2->paramNumber;
                  
                  sepRes1->paramName = ali1;
                  sepRes2->paramName = ali2;
                  
                  seaRes->aliases.insert(sepRes1);
                  seaRes->aliases.insert(sepRes2);
                  sideEffectAliases.push_back(seaRes);
                  
                }
                
                
                
                //cout<<sep->paramNumber<<" "<<sep->paramName<<endl;
              }
              //cout<<endl;
            }
          }  
        }
      }
    }                  
  }
  
  //cout<<"Exiting calcRSEAR for "<<linkName<<endl;
  
  
}


void BlameProgram::calcRecursiveSEAliases()
{
  FunctionHash::iterator fh_i;  // blameFunctions;
  
  //cout<<"Size of blameFunctions(2) is "<<blameFunctions.size()<<endl;
  for (fh_i = blameFunctions.begin(); fh_i != blameFunctions.end(); fh_i++) {
    BlameFunction *bf = (*fh_i).second;
    if (bf){  
      std::set<BlameFunction *> visited;
      visited.insert(bf);
      
      //cout<<"Calcing recursive side effect aliases for "<<bf->getName()<<endl;
      bf->calcRecursiveSEAliasesRecursive(visited);  
    }
    else{
      cerr<<"BF is NULL in calcRecursiveSEAliases"<<endl;
    }
  }
}

//std::vector<SideEffectAlias *>  sideEffectAliases;
//std::vector<SideEffectRelation *> sideEffectRelations;
//std::vector<SideEffectCall *> sideEffectCalls;
void BlameProgram::printSideEffects()
{
  FunctionHash::iterator fh_i;  // blameFunctions;
  
  cout<<"Size of blameFunctions is "<<blameFunctions.size()<<endl;
  for (fh_i = blameFunctions.begin(); fh_i != blameFunctions.end(); fh_i++)
  {
    BlameFunction * bf = (*fh_i).second;
    if (bf)
    {
      if (bf->sideEffectAliases.size() > 0 || bf->sideEffectRelations.size() > 0 
          || bf->sideEffectCalls.size() > 0)
      {
        cout<<"Function "<<bf->linkName<<" has side effects."<<endl;
        
        std::vector<SideEffectAlias *>::iterator vec_sea_i;
        std::vector<SideEffectRelation *>::iterator vec_ser_i;
        std::vector<SideEffectCall *>::iterator vec_sec_i;
        
        
        if (bf->sideEffectAliases.size() > 0)
        {
          cout<<"Aliases:"<<endl;
          for (vec_sea_i = bf->sideEffectAliases.begin(); vec_sea_i != bf->sideEffectAliases.end();
               vec_sea_i++)
          {
            SideEffectAlias * sea = *vec_sea_i;
            cout<<"A: ";
            std::set<SideEffectParam *>::iterator set_s_i;
            for (set_s_i = sea->aliases.begin(); set_s_i != sea->aliases.end(); set_s_i++)
            {
              SideEffectParam * sep = *set_s_i;
              cout<<sep->paramNumber<<" "<<sep->paramName<<endl;
            }
            cout<<endl;
          }
        }
        
        
        
        
        if (bf->sideEffectRelations.size() > 0)
        {
          cout<<"Relations:"<<endl;
          for (vec_ser_i = bf->sideEffectRelations.begin(); vec_ser_i != bf->sideEffectRelations.end();
               vec_ser_i++)
          {
            SideEffectRelation * sea = *vec_ser_i;
            cout<<"R: "<<sea->parent->paramName;
            std::set<SideEffectParam *>::iterator set_s_i;
            for (set_s_i = sea->children.begin(); set_s_i != sea->children.end(); set_s_i++)
            {
              SideEffectParam * sep = *set_s_i;
              cout<<sep->paramNumber<<" "<<sep->paramName<<endl;
            }
            cout<<endl;
          }        
        }
        
        
        
        
        
        if (bf->sideEffectCalls.size() > 0)
        {
          cout<<"Calls:"<<endl;
          for (vec_sec_i = bf->sideEffectCalls.begin(); vec_sec_i != bf->sideEffectCalls.end();
               vec_sec_i++)
          {
            SideEffectCall * sea = *vec_sec_i;
            cout<<"C: "<<sea->callNode<<endl;
            std::vector<SideEffectParam *>::iterator set_s_i;
            for (set_s_i = sea->params.begin(); set_s_i != sea->params.end(); set_s_i++)
            {
              SideEffectParam * sep = *set_s_i;
              cout<<"P: "<<sep->paramNumber<<" "<<sep->paramName<<endl;
            }
            cout<<endl;
          }                  
        }
        
        
      }
    }
    else
    {
      cout<<"BF NULL in printSideEffects"<<endl;
    }
  }
}

void BlameProgram::parseSideEffects()
{
  std::vector<string>::iterator vec_str_i;
  for (vec_str_i = blameSEFiles.begin(); vec_str_i != blameSEFiles.end();
       vec_str_i++) {

    ifstream bI((*vec_str_i).c_str());
    cout<<"Parsing .se file "<<(*vec_str_i)<<endl;
    char scratch1[6], scratch2[5], scratchName[100], scratchPN[200];
    char scratchMN[100];
    
    int bLine, eLine;
    string line;
    BlameFunction *bf = NULL;
    
    while (getline(bI, line)) {
      if (line.find("BEGIN FUNC") != string::npos) {
        sscanf(line.c_str(), "%s %s %s %s %d %d %s", scratch1, scratch2, 
               scratchName, scratchMN, &bLine, &eLine, scratchPN);
        
        bf = new BlameFunction(scratchName);
        bf->BP = this;
        
        //bf->linkName.insert(0, scratch3);
        bf->modulePathName.insert(0, scratchPN);
        string tmp(scratchMN);
        //we need clean file name, without path './'
        bf->moduleName = getFileName(tmp);
        bf->beginLineNum = bLine;
        bf->endLineNum = eLine;
        
        //cout<<"fN-"<<bf->linkName<<" MN "<<bf->moduleName<<endl;
        addFunction(bf);
      }
      else if (line.find("END FUNC") != string::npos) {
        bf = NULL;
      }
      else if (line.find("BEGIN SE_RELATIONS") != string::npos) {
        bool keepGoing = true;
        while (keepGoing) {
          getline(bI, line);
          if (line.find("END SE_RELATIONS") != string::npos)
            keepGoing = false;
          else {
            SideEffectRelation *se = new SideEffectRelation();
            
            int p1, p2;
            sscanf(line.c_str(), "%s %s %d %s %d", 
                   scratch1, scratchName, &p1, scratchMN, &p2);
            string s1(scratchName);
            string s2(scratchMN);
            
            SideEffectParam * parent = new SideEffectParam();
            SideEffectParam * child = new SideEffectParam();
            
            parent->paramNumber = p1;
            parent->paramName = s1;
            
            child->paramNumber = p2;
            child->paramName = s2;
            
            se->parent = parent;
            se->children.insert(child);
            
            if (bf!= NULL) {
              bf->sideEffectRelations.push_back(se);
              bf->hasSE = true;
            }
            else
              cerr<<"BF Null(1) in parseSideEffects"<<endl;
          }
        }
      }
      else if (line.find("BEGIN SE_ALIASES") != string::npos) {
        bool keepGoing = true;
        while (keepGoing) {
          getline(bI, line);
          if (line.find("END SE_ALIASES") != string::npos)
            keepGoing = false;
          else {
            SideEffectAlias *se = new SideEffectAlias();
            
            int p1, p2;
            sscanf(line.c_str(), "%s %s %d %s %d", 
                   scratch1, scratchName, &p1, scratchMN, &p2);
            string s1(scratchName);
            string s2(scratchMN);
            
            SideEffectParam *a1 = new SideEffectParam();
            SideEffectParam *a2 = new SideEffectParam();
            
            a1->paramNumber = p1;
            a1->paramName = s1;
            
            a2->paramNumber = p2;
            a2->paramName = s2;  
            
            se->aliases.insert(a1);
            se->aliases.insert(a2);
            
            if (bf!= NULL) {
              //std:cout<<"Pushing back "<<se<<endl;
              bf->sideEffectAliases.push_back(se);
              bf->hasSE = true;
            }
            else
              cerr<<"BF Null(2) in parseSideEffects"<<endl;
            
          }
        }        
      }
      else if (line.find("BEGIN SE_CALLS") != string::npos) {
        bool keepGoing = true;
        SideEffectCall * se = NULL;
        while (keepGoing) {
          getline(bI, line);
          if (line.find("END SE_CALLS") != string::npos)
            keepGoing = false;
          else if (line.find("C:") != string::npos) {
            se = new SideEffectCall();
            sscanf(line.c_str(), "%s %s", scratch1, scratchMN);
            string s1(scratchMN);
            se->callNode = s1;
            
            if (bf!= NULL) {
              bf->hasSE = true;
              bf->sideEffectCalls.push_back(se);
            }
            else
              cerr<<"BF Null(2) in parseSideEffects"<<endl;
            
          }
          else if (line.find("P:") != string::npos) {
            int paramNum;
            int calleeNum;
            SideEffectParam * sep = new SideEffectParam();
            sscanf(line.c_str(), "%s %d %s %d", scratchMN, &calleeNum, 
                   scratchPN, &paramNum);
            string s1(scratchPN);
            sep->paramName = s1;
            sep->paramNumber = paramNum;
            sep->calleeNumber = calleeNum;
            
            if (se != NULL)
              se->params.push_back(sep);
          }
          else {
            cerr<<"Invalid SE file in parseSideEffects"<<endl;
          }
        }  
      }
    }
  }
  //cout<<"ALL DONE!"<<endl;
}


CCTNode* BlameProgram::getOrCreateCCTNode(string fn)
{
  if (globCCT.find(fn) != globCCT.end()) 
    return globCCT[fn];

  //For test only purpose
  //cout<<"New CCTNode: "<<fn<<endl;
  CCTNode* ret = new CCTNode(fn);
  globCCT[fn] = ret;
  return ret;
}


// parse the .calls file from static analysis and build gpu side CCT
void BlameProgram::parseCallFiles()
{
  std::vector<string>::iterator vec_str_i;
  for (vec_str_i = blameCallFiles.begin(); 
       vec_str_i != blameCallFiles.end(); vec_str_i++) {

    cout<<"Parsing .calls file "<<(*vec_str_i)<<endl;
    ifstream ifs((*vec_str_i).c_str());
    if (!ifs.is_open()) {
      cerr<<"Failed open call file: "<<*vec_str_i<<endl;
      continue;
    }
    string line;
    CCTNode *func = NULL;

    while (getline(ifs, line)) {
      if (line.find("FUNCTION") == 0) {
        stringstream ss(line);
        string buf;
        //get linkage name
        ss>>buf; //buf = FUNCTION
        ss>>buf; //buf = func linkage name
        func = getOrCreateCCTNode(buf);
        //get real name
        getline(ifs, buf); //buf = func real name
        func->realName = buf;
        //get module name
        getline(ifs, buf); //buf = module name
        func->moduleName = buf;
        //get function type
        getline(ifs, buf); //buf = ft
        int val = stoi(buf); //cast from int to enum fails
        if (val == 0)
          func->ft = KERNEL;
        else if (val == 1)
          func->ft = DEVICE;
        else if (val == 2)
          func->ft = HOST;
        else
          cerr<<"ERROR: ft="<<val<<endl;
      }
      else if (line.find("END FUNCTION") != std::string::npos) {
        continue;
      } 
      else {
        //check all callees
        bool keepGoing = true;
        while (keepGoing) {
          //remove the sufix "aaaa.."
          string subline = getLineWithout_a(line);
          //cout<<"clean call: "<<subline<<endl;
          //only process if it's new in func's children
          if (func->children.find(subline) == 
              func->children.end()) {
            func->children.insert(subline);
            size_t pos = subline.find("--");
            //get call node info
            string calleeName = subline.substr(0, pos);
            int line_num = stoi(subline.substr(pos+2));
            if (calleeName.empty() || line_num == 0)
              continue; //some cases like --0, --0a
            //create CCTNode for the callee (call node), 
            //and Caller for the func then set 
            //parents relationship between it and the func
            Caller *callerNode = new Caller(line_num, func->moduleName, func);
            CCTNode *callee = getOrCreateCCTNode(calleeName);
            callee->parents.push_back(callerNode);
           
            //For test only purpose
            //cout<<"Adding: "<<func->linkName<<" to "<< \
            callee->linkName<<"'s parents"<<endl;
          }
          //check the next line in the to see if 
          getline(ifs, line);
          if (line.find("END FUNCTION") != std::string::npos) 
            keepGoing = false;
        } //end of while
      } //end of callees
    } //end of while
  } //end of for loop
}


BlameFunction *BlameProgram::getOrCreateBlameFunction(string funcName, string funcRealName, int ft, string moduleName, string modulePathName)
{
  if (realLinkNameMap.find(funcRealName) == realLinkNameMap.end())
    realLinkNameMap[funcRealName] = funcName;
  else {
    if (realLinkNameMap[funcRealName] != funcName) {
      cerr<<"Check! same realName mapped to different linkName! rn/oln/nln/nft: "
          <<funcRealName<<" "<<realLinkNameMap[funcRealName]<<" "<<funcName<<" "<<ft<<endl;
    }
  }

  BlameFunction *bf = NULL;
  FunctionHash::iterator fh_i_check;  // blameFunctions;
  fh_i_check = blameFunctions.find(funcName);
  if (fh_i_check != blameFunctions.end()) {
    bf = blameFunctions[funcName];
  }
  
  if (bf != NULL) {
    // Do a sanity check to make sure the modules are the same
    if (bf->moduleName != moduleName) {
      cerr<<"Diff moduleNames for func "<<funcName<<" M1: "<<bf->getModuleName();
      cerr<<" M2: "<<moduleName<<endl;
    }
  }
  else {
    //if it's new, we need to call addFunction here
    cerr<<"Weird: what func is new? "<<funcName<<endl;
    bf = new BlameFunction(funcName);
    bf->moduleName = moduleName;
    bf->modulePathName = modulePathName;
    addFunction(bf);
  }
    
  //New or not, we need to set realName and ft here since .se didn't have this info
  bf->realName = funcRealName;
  bf->ft = static_cast<func_type>(ft);  
  return bf;
}

void BlameProgram::parseProgram()
{
  std::vector<string>::iterator vec_str_i;
  for (vec_str_i = blameExportFiles.begin(); vec_str_i != blameExportFiles.end(); 
       vec_str_i++) {
    ifstream bI((*vec_str_i).c_str());
    
    cout<<"Parsing .blm file "<<(*vec_str_i)<<endl;
    
    string line;
    int count=0; // keep record of the number of iterations in while loop
    while (getline(bI,line)) {
      size_t foundBeg;
      // count ++;
      foundBeg = line.find("BEGIN FUNC"); 
      BlameFunction *bf;
      
      if (foundBeg != string::npos) {
        string funcName, funcRealName, moduleName, modulePathName;
        int funcT;
        // cout<<endl<<endl;
        // BEGIN_F_NAME
        getline(bI, line);
        // link Name
        getline(bI, line);
        // cout<<"Parsing Function "<<line<<endl;
        funcName = line;
        // END F_NAME
        getline(bI, line);
        
        // BEGIN_F_REAL_NAME
        getline(bI, line);
        // real Name
        getline(bI, line);
        // cout<<"Parsing Function "<<line<<endl;
        funcRealName = line;
        // END F_NAME
        getline(bI, line);
        
        // BEGIN FUNC_TYPE
        getline(bI, line);
        // func type
        getline(bI, line);
        // cout<<"Parsing Function "<<line<<endl;
        funcT = stoi(line);
        // END FUNC_TYPE
        getline(bI, line);
        
        //----
        // BEGIN M_NAME_PATH
        getline(bI, line);
        //cout<<"Should be M_NAME_PATH -- is -- "<<line<<endl;
        // Get Module Path Name()
        getline(bI, line);
        //setModulePathName(line);
        modulePathName = line;
        // END M_NAME_PATH
        getline(bI, line);
        
        //----
        // BEGIN M_NAME
        getline(bI, line);
        
        // Get Module Name
        getline(bI, line);
        //setModuleName(line);
        moduleName = line;
        //we need clean file name without path './'
        moduleName = getFileName(moduleName);
        
        // END M_NAME
        getline(bI, line);
        
        bf = getOrCreateBlameFunction(funcName, funcRealName, funcT, 
                                      moduleName, modulePathName);
        //bf = new BlameFunction(line);
        
        bf->BP = this;
        
        bool isMain = false;  
#ifdef HUI_CHPL
        string MAIN = "chpl_user_main";
        int MAINLEN = 14;
#else
        string MAIN = "main";
        int MAINLEN = 4;
#endif
        // TODO: Make this determined at runtime
        if (funcName.find(MAIN) != string::npos && funcName.length() == MAINLEN) {
          //cout<<"Setting main as a blame point"<<endl;
          isMain = true;
        }
        
        // This will not return NULL if the function in question is one
        //  of the sampled functions from the run
        if (bf->parseBlameFunction(bI) != NULL) {
          // TODO: Optimize this
          bf->calcAggregateLN();
          bf->isFull = true;
          //addFunction(bf);  
          if (isMain)
            bf->setBlamePoint(EXPLICIT); //TOCHECK: chpl_user_main
        }
      }
    }
   //  cout<<"There are "<<count<<" loops in while"<<endl;
  }
}

void BlameProgram::printParsed(std::ostream &O)
{
  //cout<<"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"<<endl;
  //std::vector<BlameModule *>::iterator bm_i;
  
  ModuleHash::iterator bm_i;
  
  for (bm_i = blameModules.begin();  bm_i != blameModules.end(); bm_i++)
  {
    BlameModule * bm = (*bm_i).second;
    if (bm)
      bm->printParsed(O);
    else
    {
      O<<"NULL Blame Module -- "<< (*bm_i).first<<endl;
    }
  }  
}
