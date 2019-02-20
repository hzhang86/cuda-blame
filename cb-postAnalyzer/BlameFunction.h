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

#ifndef BF_GUARD
#define BF_GUARD 

#include <algorithm>
#include <iostream>
#include <ostream>
//#include <boost/config.hpp>
//#include <boost/property_map.hpp>
#include <string>
#include <algorithm>
#include <set>
#include <vector>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Instances.h"
#include "util.h"
//#include "ExitVariable.h"
//#include "ExitProgram.h"
#include "VertexProps.h"
/*
#ifdef __GNUC__
#include <ext/hash_map>
#else
#include <hash_map>
#endif
*/
#include <unordered_map>

#define NO_BLAME 0
#define EXPLICIT 1
#define IMPLICIT 2

#define SE_NONE       0 // don't know when this would ever come up
#define SE_CALL       1
#define SE_RELATION   2
#define SE_ALIAS      3

class BlameModule;  // Temporary object
class StackFrame;
class BlameProgram;

typedef std::unordered_map<std::string, VertexProps *, std::hash<std::string>, eqstr> VertexHash;
typedef std::multimap<int, VertexProps *> VertexMap;

//enum func_type {KERNEL, DEVICE, HOST};

struct SideEffectParam
{
  int paramNumber; // The EV param number within the function
  int calleeNumber; // For calls, this is the number of the 
   /// param within the callee function
   
  VertexProps * vpValue; 
   
  string paramName;
};


struct SideEffectAlias
{
  // In the case of Aliases, this is the set of equivalent aliases

  set<SideEffectParam *> aliases;

};


struct SideEffectRelation 
{
  // For relationships, this is the node the children to map to
  // For calls, this is the call node
  //string parent;
  SideEffectParam * parent;
  
  
  // In case of Relations, this is all the vertices that have
  //   a relationship with the parent
  
  // In the case of Calls, this is all the parameters 
  
  set<SideEffectParam *> children;

};

struct SideEffectCall
{
  string callNode;
  vector<SideEffectParam *> params;
};

struct LoopInfo
{
  set<int> lineNumbers;
};


class BlameFunction
{
  public:
    BlameProgram *BP;
 
     BlameFunction(std::string name){linkName = name; blamePoint = NO_BLAME; isFull = false; sideEffectC = false; hasSE = true; seAliasCalc = false; }

    BlameFunction(){blamePoint = NO_BLAME; isFull = false; sideEffectC = false; hasSE = true; seAliasCalc = false; }
   
    ~BlameFunction(){ }
  
    void addBlameToFieldParents(VertexProps * vp, std::set<VertexProps*> & blamees, short fromWhere);

    void resolveLineNum(vector<StackFrame> & frames, ModuleHash & modules, vector<StackFrame>::iterator vec_SF_i, set<VertexProps *> & blamees, short isBlamePoint, bool isBottomParsed, BlameFunction * oldFunc, std::ostream &O);
                      
    std::string getFullContextName(vector<StackFrame> & frames, ModuleHash & modules, vector<StackFrame>::iterator vec_SF_i);                      
  void determineBlameHolders(std::set<VertexProps *> & blamees,std::set<VertexProps *> & oldBlamees,VertexProps * callNode,int lineNum, short isBlamePoint, std::set<VertexProps *> & localBlamees,std::set<VertexProps *> & DQblamees);
                                              void populateTempSideEffects(int lineNum, std::set<VertexProps *> & blamees);
  
    void populateTempSideEffectsRelations();
  
    void addTempFieldBlameesRecursive(VertexProps * vp, VertexProps * blamee, std::set<VertexProps *> & oldBlamees,  std::set<VertexProps *> & blamees, std::set<VertexProps *> & visited);
                                          
    void calcRecursiveSEAliasesRecursive(std::set<BlameFunction *> & visited);
  
    VertexProps *resolveSideEffectsCheckParentEV(VertexProps * vp, std::set<VertexProps *> & visited);

    VertexProps *resolveSideEffectsCheckParentLV(VertexProps * vp, std::set<VertexProps *> & visited);
                                          
    void handleTransferFunction(VertexProps * callNode, set<VertexProps *> & blamedParams);
    void resolvePAForParamNode(VertexProps *param);
    bool fieldApplies(SideEffectParam * matchSEP, std::set<VertexProps *> & blamees);

    BlameFunction *parseBlameFunction(ifstream & bI);
  
    void printParsed(std::ostream &O);

    VertexProps * findOrCreateVP(std::string &name);
  
    VertexProps * findOrCreateTempBlamees(std::set<VertexProps *> & blamees, std::string name, bool & found);
  
    void addTempFieldBlamees(std::set<VertexProps *> & blamees, std::set<VertexProps *> & oldBlamees);

    void clearTempFields(std::set<VertexProps *> & oldBlamees, BlameFunction * oldFunc);

    void calcParamInfo(std::set<VertexProps *> & blamees, VertexProps * callNode);
  
    void addPidAliasesToBlamees(std::set<VertexProps *> &blamees, std::set<VertexProps *> &localBlamees);
  
    void calcAggCallRecursive(VertexProps * ivp);

    void calcAggregateLNRecursive(VertexProps * ivp);

    void calcAggregateLN();

    void calcAggregateLN_SE_Recursive(VertexProps * ivp);

    void calcAggregateLN_SE();

    void calcSideEffectDependencies();

    void calcSEDWritesRecursive(VertexProps * ev, VertexProps * vp, std::set<VertexProps *> & visited);

    void calcSEDReadsRecursive(VertexProps * ev, VertexProps * vp, std::set<VertexProps *> & visited);

    bool transferFuncApplies(VertexProps * caller, std::set<VertexProps *> & oldBlamees,VertexProps * callNode);
                                          
    bool notARepeat(VertexProps * vp, std::set<VertexProps *> & blamees);
                                          
    void outputFrameBlamees(std::set<VertexProps *> & blamees, std::set<VertexProps *> & localBlamees, std::set<VertexProps *> &  DQBlamees, std::ostream &O);                                          
    void addExitVar(VertexProps * vp){exitVariables.push_back(vp);}
  
    void addExitProg(VertexProps * vp){exitPrograms.push_back(vp);}

    void addExitOutputs(VertexProps * vp){exitOutputs.push_back(vp);}

    void addCallNode(VertexProps * vp){callNodes.push_back(vp);}

    int getBLineNum() const {return beginLineNum;}

    int setBLineNum(int bln) {beginLineNum = bln;}

    int getELineNum() const {return endLineNum;}
  
    int setELineNum(int eln) {endLineNum = eln;}
  
    short getBlamePoint() {return blamePoint;}
  
    void setBlamePoint(short bp) {blamePoint = bp;} 

    void setModuleName(std::string name) {moduleName = name;}

    void setModulePathName(std::string name){modulePathName = name;}

    std::string getName(){return linkName;}

    std::string getRealName(){return realName;}
  
    std::string getModuleName() { return moduleName;}

    std::string getModulePathName() { return modulePathName;}
  
    std::string getFullStructName(VertexProps *vp);
  
    void outputParamInfo(std::ostream &O, VertexProps * vp);

    void clearPastData();

    /// SAMPLE CONVERGE FUNCTIONS ////
    
    BlameFunction *parseBlameFunction_SC(ifstream & bI);
  
    //////////////////////////////////////////

    ////// ALIAS ONLY FUNCTIONS /////////////////currenly removed for cuda 02/07/18
    /*
    BlameFunction *parseBlameFunction_OA(ifstream & bI);
  
    void resolveLineNum_OA(vector<StackFrame> & frames, ModuleHash & modules,vector<StackFrame>::iterator vec_SF_i, set<VertexProps *> & blamees,short isBlamePoint, bool isBottomParsed, BlameFunction * oldFunc, std::ostream &O);
                        
    void determineBlameHolders_OA(std::set<VertexProps *> & blamees,std::set<VertexProps *> & oldBlamees,VertexProps * callNode,int lineNum, short isBlamePoint, std::set<VertexProps *> & localBlamees,std::set<VertexProps *> & DQblamees);            
                
  
    void handleTransferFunction_OA(VertexProps * callNode, set<VertexProps *> & blamedParams);
                                                            
    void calcParamInfo_OA(std::set<VertexProps *> & blamees, VertexProps * callNode);
  
    void outputFrameBlamees_OA(std::set<VertexProps *> & blamees, std::set<VertexProps *> & localBlamees, std::set<VertexProps *> &  DQBlamees, std::ostream &O);  

    void addTempFieldBlamees_OA(std::set<VertexProps *> & blamees, std::set<VertexProps *> & oldBlamees);
                      
    void addTempFieldBlameesRecursive_OA(VertexProps * vp, VertexProps * blamee, std::set<VertexProps *> & oldBlamees,std::set<VertexProps *> & blamees, std::set<VertexProps *> & visited);  
                            
    bool transferFuncApplies_OA(VertexProps * caller, std::set<VertexProps *> & oldBlamees,  VertexProps * callNode);                                                  
    void addLoops_OA(ifstream & bI);

    float determineWeight(VertexProps * vp, int lineNum);

    /// For Reads
    void resolveLineNum_OAR(vector<StackFrame> & frames, ModuleHash & modules,vector<StackFrame>::iterator vec_SF_i, set<VertexProps *> & blamees,  short isBlamePoint, bool isBottomParsed, BlameFunction * oldFunc, std::ostream &O);
                      
    void determineBlameHolders_OAR(std::set<VertexProps *> & blamees,std::set<VertexProps *> & oldBlamees,VertexProps * callNode,int lineNum, short isBlamePoint, std::set<VertexProps *> & localBlamees,std::set<VertexProps *> & DQblamees);      
                
    void outputFrameBlamees_OAR(std::set<VertexProps *> & blamees, std::set<VertexProps *> & localBlamees,std::set<VertexProps *> &  DQBlamees, std::ostream &O);    */
  
    ///////////////Some Utility Functions////////////////////////////////////
    bool endWithNumber(const std::string &str);
    bool anySubstrIsNumber(const std::string &str);
    ///// For Converging Samples ///////////////////
    std::vector<FullSample *> fullSamples; 
    /////////////////////////////////////////////////////

    std::string linkName; //mangled with function signature
    std::string realName; //demangled 
    func_type ft; //whether it's a host/device/kernel function
    std::string moduleName;
    std::string modulePathName;
  
    bool seAliasCalc; // true if seAliasRecursive has been called already
    // true if side effects have already been calculated
    bool sideEffectC;
    int beginLineNum;
    int endLineNum;

    //added by Hui 03/29/16
    std::set<int> allLineNums; //all valid line# within this function

    bool isFull;  /// meaning we have full parsed information for the func
    bool hasSE;   /// some kind of sideffect
  
    int buffer;
    int buffer2;
    int buffer3;
  
    std::vector<SideEffectAlias *>  sideEffectAliases;
    std::vector<SideEffectRelation *> sideEffectRelations;
    std::vector<SideEffectCall *> sideEffectCalls;
  
    // For alias only stuff ///
    std::vector<LoopInfo *> loops;
    ////////////////////////////
  
    VertexMap   allLines;
    std::set<VertexProps *> hasParams; //all the actual args in this bf for all, if vp->params.size()>0, then vp is one of the hasParams of this fb
                                    //call nodes
  
    // Sets that hold pointers to temp variables so we can clear them later
    VertexMap   tempLines;
    std::set<VertexProps *> tempSEs;
    std::set<VertexProps *> tempSEParents;
    std::set<VertexProps *> tempParentsHolder;
    std::set<VertexProps *> tempChildrenHolder;

    std::vector<VertexProps *>  callNodes;//nodes that call other funcs in 
                                          //this BlameFunction
//private: //07/18/17 for the ease use of these variables
    short blamePoint;
    std::vector<VertexProps *>  exitVariables;
    std::vector<VertexProps *>  exitPrograms;
    std::vector<VertexProps *>  exitOutputs;

    //std::vector<VertexProps *>  allVertices;
    VertexHash  allVertices;
};

#endif
