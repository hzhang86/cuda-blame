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

#include "VertexProps.h"
#include "BlameFunction.h"
#include "BlameProgram.h"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;



void VertexProps::printParsed(std::ostream & O)
{
    O<<"BEGIN VAR  "<<std::endl;
    
    O<<"BEGIN V_NAME "<<std::endl;
    O<<name<<std::endl;
    O<<"END V_NAME "<<std::endl;
    
    O<<"BEGIN V__REAL_NAME "<<std::endl;
    O<<realName<<std::endl;
    O<<"END V_REAL_NAME "<<std::endl;
    
    O<<"BEGIN V_TYPE "<<std::endl;
    O<<eStatus<<std::endl;
    O<<"END V_TYPE "<<std::endl;
    
    O<<"BEGIN N_TYPE "<<std::endl;
    for (int a = 0; a < NODE_PROPS_SIZE; a++)
      O<<nStatus[a]<<" ";
    O<<std::endl;
    
    O<<"END N_TYPE "<<std::endl;
    
    O<<"BEGIN DECLARED_LINE"<<endl;
    O<<declaredLine<<std::endl;
    O<<"END DECLARED_LINE"<<endl;
    
    
    std::set<VertexProps *>::iterator ivp_i;
    
    O<<"BEGIN PARENTS"<<endl;
    for (ivp_i = parents.begin(); ivp_i != parents.end(); ivp_i++)
    {
      VertexProps * ivpParent = (*ivp_i);
      O<<ivpParent->name<<endl;
    }
    O<<"END PARENTS"<<endl;  
    
    O<<"BEGIN CHILDREN "<<endl;
    for (ivp_i = children.begin(); ivp_i != children.end(); ivp_i++)
    {
      VertexProps * ivpChild = (*ivp_i);
      O<<ivpChild->name<<endl;
    }
    O<<"END CHILDREN"<<std::endl;

  
    O<<"BEGIN ALIASES "<<std::endl;
    for (ivp_i = aliases.begin(); ivp_i != aliases.end(); ivp_i++)
    {
      VertexProps * ivpAlias = (*ivp_i);
      O<<ivpAlias->name<<endl;
    }
    O<<"END ALIASES "<<std::endl;
    
    O<<"BEGIN DATAPTRS "<<std::endl;
    for (ivp_i = dataPtrs.begin(); ivp_i != dataPtrs.end(); ivp_i++)
    {
      VertexProps * ivpAlias = (*ivp_i);
      O<<ivpAlias->name<<endl;
    }
    O<<"END DATAPTRS "<<std::endl;


    O<<"BEGIN CALLS"<<endl;
    std::vector<FuncCall *>::iterator ifc_i;
    for (ifc_i = calls.begin(); ifc_i != calls.end(); ifc_i++)
    {
      FuncCall *iFunc = (*ifc_i);
      O<<iFunc->Node->name<<"  "<<iFunc->paramNumber<<std::endl;
    }
    O<<"END CALLS"<<endl;
    
    std::set<int>::iterator si_i;
    O<<"BEGIN LINENUMS "<<endl;
    for (si_i = lineNumbers.begin(); si_i != lineNumbers.end(); si_i++)
    {
      O<<*si_i<<endl;
    }
    O<<"END LINENUMS "<<endl;
    
    O<<"END VAR "<<name<<std::endl;

}

void VertexProps::findSEExits(std::set<VertexProps *> & blamees)
{
  //cout<<"Enter findSEExits"<<std::endl;
  std::set<SideEffectParam *>::iterator set_sep_i;
  for (set_sep_i = tempAliases.begin(); set_sep_i != tempAliases.end(); set_sep_i++) {
    SideEffectParam *sep = *set_sep_i;
    if (sep == NULL)
      continue;
    if (sep->vpValue == NULL)
      continue;
      
    sep->vpValue->addedFromWhere = 81;
#ifdef DEBUG_BLAMEES
    cout<<"Blamees insert "<<sep->vpValue->name<<" in findSEExits(1)"<<std::endl;
#endif
    blamees.insert(sep->vpValue);
  
    if (sep->vpValue->nStatus[EXIT_VAR_FIELD] || sep->vpValue->nStatus[LOCAL_VAR_FIELD]) {
      VertexProps * upPtr = sep->vpValue->fieldUpPtr;
      while (upPtr != NULL) {
        upPtr->addedFromWhere = 91;
#ifdef DEBUG_BLAMEES
        cout<<"Blamees insert "<<upPtr->name<<" in findSEExits(2)"<<std::endl;
#endif
        blamees.insert(upPtr);
        upPtr = upPtr->fieldUpPtr;
      }
    }
  }
  //cout<<"Leaving findSEExits"<<std::endl;
}

// add blamees that are found due to some side effect relation
void VertexProps::populateSERBlamees(std::set<VertexProps *> & visited, std::set<VertexProps *> & blamees)
{
  //cout<<"Entering populateSERBlamees for "<<name<<std::endl;
  if (visited.count(this) > 0) {
    //cout<<"Exiting(1) populateSERBlamees for "<<name<<std::endl;
    return;
  }
    
  visited.insert(this);
  
  std::set<VertexProps *>::iterator set_vp_i;
  for (set_vp_i = tempRelationsParent.begin(); set_vp_i != tempRelationsParent.end(); set_vp_i++) {
    VertexProps *trp = *set_vp_i;
    if (trp->eStatus > NO_EXIT || trp->nStatus[EXIT_VAR_FIELD]) {
    //cout<<"Blamees insert(13) "<<trp->name<<std::endl;
      trp->addedFromWhere = 82;
#ifdef DEBUG_BLAMEES
        cout<<"Blamees insert "<<trp->name<<" in populateSERBlamees(1)"<<endl;
#endif
        blamees.insert(trp);
    
    if (trp->nStatus[EXIT_VAR_FIELD] || trp->nStatus[LOCAL_VAR_FIELD]) {
      VertexProps * upPtr = trp->fieldUpPtr;
      while (upPtr != NULL) {
      upPtr->addedFromWhere = 92;
#ifdef DEBUG_BLAMEES
            cout<<"Blamees insert "<<upPtr->name<<" in populateSERBlamees(2)"<<endl;
#endif
            blamees.insert(upPtr);
      upPtr = upPtr->fieldUpPtr;
        }
      }
    }
    trp->populateSERBlamees(visited, blamees);
  }
  
  //cout<<"Exiting(2) populateSERBlamees for "<<name<<std::endl;
}

// TODO: figure out if this applies to more than parent/child relationships
void VertexProps::propagateTempLineUp(std::set<VertexProps *> & visited, int lineNum)
{
  if (visited.count(this) > 0)
    return;
    
  visited.insert(this);


  std::set<VertexProps *>::iterator set_vp_i;

  for (set_vp_i = parents.begin(); set_vp_i != parents.end(); set_vp_i++)
  {
    if ((*set_vp_i)->eStatus > 0)
    {
      (*set_vp_i)->tempLine = lineNum;
      BF->tempLines.insert(pair<int, VertexProps *>(lineNum,(*set_vp_i)));
    }
    (*set_vp_i)->propagateTempLineUp(visited, lineNum);
  }
}



void VertexProps::populateTFVertices(std::set<VertexProps *> & visited)
{
  if (visited.count(this) > 0)
    return;
    
  visited.insert(this);  

  if (this->fieldUpPtr != NULL)
  {
    fieldUpPtr->populateTFVertices(visited);
  }
  
  std::set<VertexProps *>::iterator set_vp_i;

  for (set_vp_i = tempParents.begin(); set_vp_i != tempParents.end(); set_vp_i++)
  {
      VertexProps * vp = (*set_vp_i);
      vp->populateTFVertices(visited);
  }    

}
      


int VertexProps::findBlamedExits(std::set<VertexProps *> & visited, int lineNum)
{
  int total = 0;
  
  if (visited.count(this) > 0)
    return 0;

  visited.insert(this);
  
  // We don't want to end up grabbing our struct parent (due to a param thing)
  //  and using their line number
  if (this->fieldUpPtr != NULL)
    visited.insert(this->fieldUpPtr);
  
  if (lineNumbers.count(lineNum) || declaredLine == lineNum) {
#ifdef CHECK_PARAM_WRITTEN_IN_CALL
    //param is only blamed for this call when it's written in the call
    if (paramIsBlamedForTheCall) {
      //cout<<this->name<<"'s paramIsBlamedForTheCall=1 and lineNumbers includes "<<lineNum<<endl;
      return 1;
    }
#endif
  }

  std::set<VertexProps *>::iterator set_vp_i;
  std::vector<FuncCall *>::iterator vec_vp_i;
  
  // This is the key step for EVs ... the params are data pointers
  // that represent the EVs when passed into function ... unfortunately
  // if we don't represent the params correctly then we lose the link
  // to the EVs which is a bad thing
#ifdef DEBUG_BLAMED_EXITS
  cout<<"Before params, total="<<total<<" for "<<this->name<<endl;
#endif
  for (set_vp_i = params.begin(); set_vp_i != params.end(); set_vp_i++)
  {
    VertexProps * vp = (*set_vp_i);
#ifdef DEBUG_BLAMED_EXITS
    cout<<vp->name<<" is of params of "<<name<<std::endl;
#endif
    //cout<<"Total before - "<<total<<std::endl;
    //cout<<"Temp Parents size - "<<vp->tempParents.size()<<std::endl;
    if (vp->tempParents.size() == 0)
      total += vp->findBlamedExits(visited, lineNum);
    //cout<<"Total after - "<<total<<std::endl;
  }
#ifdef DEBUG_BLAMED_EXITS
  cout<<"After params, total="<<total<<" for "<<this->name<<endl;
#endif
    /*
  for (vec_vp_i = calls.begin(); vec_vp_i != calls.end(); vec_vp_i++)
  {
      FuncCall * fc = *vec_vp_i;
  }*/
  
  for (set_vp_i = fields.begin(); set_vp_i != fields.end(); set_vp_i++)
  {
    VertexProps * vp = (*set_vp_i);
#ifdef DEBUG_BLAMED_EXITS
    cout<<vp->name<<" is of fields of "<<name<<std::endl;
#endif
    if (vp->tempParents.size() == 0)
      total += vp->findBlamedExits(visited, lineNum);
  }
#ifdef DEBUG_BLAMED_EXITS
  cout<<"After fields, total="<<total<<" for "<<this->name<<endl;
#endif
  for (set_vp_i = tempChildren.begin(); set_vp_i != tempChildren.end(); set_vp_i++)
  {
    VertexProps * vp = (*set_vp_i);
#ifdef DEBUG_BLAMED_EXITS
    cout<<vp->name<<" is of tempChildren of "<<name<<std::endl;
#endif
    total += vp->findBlamedExits(visited, lineNum);
  }
#ifdef DEBUG_BLAMED_EXITS
  cout<<"After tempChildren, total="<<total<<" for "<<this->name<<endl;
#endif
  return total;
}


void VertexProps::adjustVertex()
{
  if (eStatus > NO_EXIT) //because exit vars is the top-level
  {                     //they came from outside of this frame
    fieldUpPtr = NULL;
  }
}



void VertexProps::parseVertex(ifstream & bI, BlameFunction * bf)
{
  string line;
  bool proceed = true;
  
  bool interestingVar = false;  // exit variable, local variable, or field
                               // are interestingVars
  //----
  //BEGIN V_TYPE
  getline(bI, line);
  
    
  // get variable type
  getline(bI, line);
  
  //cout<<"Parsing Var type for "<<name<<" "<<line.c_str()<<std::endl;

  eStatus = atoi(line.c_str());
  
  //cout<<"Estatus is "<<eStatus<<std::endl;
  
  //END V_TYPE
  getline(bI, line);


  if (eStatus >= EXIT_VAR_GLOBAL) {
    bf->addExitVar(this);
    }
  else if (eStatus == EXIT_PROG)
    bf->addExitProg(this);
  else if (eStatus == EXIT_OUTP)
    bf->addExitOutputs(this);
  
  //----
  // BEGIN N_TYPE
  getline(bI, line);
  
  //char arr[100];
  //int paramNum;
  
  
  getline(bI, line);
  //cout<<"Parsing nStatus line "<<line.c_str()<<std::endl;
  
  sscanf(line.c_str(), "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", 
      &(nStatus[0]), &(nStatus[1]), &(nStatus[2]), &(nStatus[3]), &(nStatus[4]), 
      &(nStatus[5]), &(nStatus[6]), &(nStatus[7]), &(nStatus[8]), &(nStatus[9]),
      &(nStatus[10]), &(nStatus[11]), &(nStatus[12]), &(nStatus[13]), &(nStatus[14]), 
      &(nStatus[15]), &(nStatus[16]), &(nStatus[17]), &(nStatus[18]), &(nStatus[19]));
      
  
  
  //sscanf(line.c_str(), "%d %d", &(nStatus[0]), &(nStatus[1]));
  
  if (nStatus[CALL_NODE])
    bf->addCallNode(this);

  // END N_TYPE
  getline(bI, line);

  //cout<<"Estatus is "<<eStatus<<std::endl;

  if (eStatus > NO_EXIT ||
      nStatus[LOCAL_VAR] || nStatus[LOCAL_VAR_FIELD] || nStatus[EXIT_VAR_FIELD])
  {
    interestingVar = true;
  }


  //----
  //BEGIN DECLARED_LINE
  getline(bI, line);
    
  // get declared line
  getline(bI, line);
  declaredLine = atoi(line.c_str());
  
  //END DECLARED_LINE
  getline(bI, line);
  
  
  
  //----
  //BEGIN IS_WRITTEN
  getline(bI, line);
    
  // get declared line
  getline(bI, line);
  isWritten = atoi(line.c_str());
  
  //END IS_WRITTEN
  getline(bI, line);


  //----
  //BEGIN CHILDREN
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END CHILDREN") != string::npos)
      proceed = false;
    else
    {
      VertexProps * child = bf->findOrCreateVP(line);
      children.insert(child);
      child->parents.insert(this);
    }
  }


  //----
  //BEGIN ALIASES
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END ALIASES") != string::npos)
      proceed = false;
    else
    {
      VertexProps * alias = bf->findOrCreateVP(line);
      aliases.insert(alias);
      //TODO: make sure what's aliasUpPtr really mean, aliases are
            //mutually inclusive, like a is b's alias also b's aliasUpPtr ??
            //alias->aliasUpPtr = this; //a->alias->aliasUpPtr = a ??? what's the logic
            //added above aliasUpPtr line back 08/06/16
    }
  }



  //----
  //BEGIN DATAPTRS
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END DATAPTRS") != string::npos)
      proceed = false;
    else
    {
      VertexProps * dataptr = bf->findOrCreateVP(line);
      dataPtrs.insert(dataptr);
      dataptr->dpUpPtr = this;
    }
  }
  
  
  //----
  //BEGIN DFALIAS
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END DFALIAS") != string::npos)
      proceed = false;
    else
    {
      VertexProps * dfalias = bf->findOrCreateVP(line);
      dfAliases.insert(dfalias);
      dfalias->dfaUpPtr = this;
    }
  }
  
  
  //----
  //BEGIN DFCHILDREN
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END DFCHILDREN") != string::npos)
      proceed = false;
    else
    {
      VertexProps * dfalias = bf->findOrCreateVP(line);
      dfChildren.insert(dfalias);
    }
  }



  //----
  //BEGIN RESOLVED_LS
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END RESOLVED_LS") != string::npos)
      proceed = false;
    else
    {
      VertexProps * resls = bf->findOrCreateVP(line);
      resolvedLS.insert(resls);
    }
  }
  
  
  
  //----
  //BEGIN RESOLVEDLS_FROM
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END RESOLVEDLS_FROM") != string::npos)
      proceed = false;
    else
    {
      VertexProps * rlsf = bf->findOrCreateVP(line);
      resolvedLSFrom.insert(rlsf);
    }
  }
  
  
  
  //----
  //BEGIN RESOLVEDLS_SE
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END RESOLVEDLS_SE") != string::npos)
      proceed = false;
    else
    {
      VertexProps * rlsse = bf->findOrCreateVP(line);
      resolvedLSSideEffects.insert(rlsse);
    }
  }
  
  
  //----
  //BEGIN STORES_TO
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END STORES_TO") != string::npos)
      proceed = false;
    else
    {
      VertexProps * stto = bf->findOrCreateVP(line);
      storesTo.insert(stto);
    }
  }



  //----
  //BEGIN FIELDS
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END FIELDS") != string::npos)
      proceed = false;
    else
    {
      VertexProps * field = bf->findOrCreateVP(line);
      
      if (field->eStatus == NO_EXIT)
      {
        fields.insert(field);
        field->fieldUpPtr = this;
      }
    }
  }
  
  // -------
  // BEGIN FIELD_ALIAS
  getline(bI, line);
    
  // get field alias
  getline(bI, line);
  
  if (line.find("NULL") != string::npos)
  {
    fieldAlias = NULL;
  }
  else
  {
    VertexProps * field = bf->findOrCreateVP(line);
    fieldAlias = field;
  }
      
  //END FIELD_ALIAS
  getline(bI, line);
  
  
  
  //----
  //BEGIN GEN_TYPE
  getline(bI, line);
    
  // get general type
  getline(bI, genType);
  
  //END GEN_TYPE
  getline(bI, line);
  
  // --
  // BEGIN STRUCTTYPE
  getline(bI, line);

  getline(bI, line);
  
  if (line.find("NULL") != string::npos)
    sType = NULL;
  else // never executed for experiment 
  {
    if (BF == NULL)
    {
      cerr<<"BF is NULL"<<std::endl;
      sType = NULL;
    }
    else
    {
      if (BF->BP == NULL)
      {
        cerr<<"BF->BP is NULL"<<std::endl;
        sType = NULL;
      }
      else
      {
        sType = BF->BP->blameStructs[line];
      }
    }
  }
  getline(bI, line);

  
  
  // --
  // BEGIN STRUCTPARENT
  getline(bI, line);
  
  //StructBlame * bs;
  
  getline(bI, line);
  
  //cout<<"SP - "<<line<<std::endl;
  if (line.find("NULL") != string::npos)
    bs = NULL;
  else // never executed for experiment
  {
    if (BF == NULL)
    {
      cerr<<"BF is NULL"<<std::endl;
      bs = NULL;
    }
    else
    {
      if (BF->BP == NULL)
      {
        cerr<<"BF->BP is NULL"<<std::endl;
        bs = NULL;
      }
      else
      {
        bs = BF->BP->blameStructs[line];
      }
    }
  }
  getline(bI, line);
  
  
  
  // - 
  // BEGIN STRUCTFIELDNUM
  getline(bI, line);

  //StructField * sField;
  getline(bI, line);
  
  //cout<<"SF - "<<line<<std::endl;
  if (line.find("NULL") != string::npos)
    sField = NULL;
  else
  {
    if (bs == NULL)
    {
      cerr<<"bs is NULL"<<std::endl;
      sField = NULL;
    }
    else
    {
      sField = bs->fields[atoi(line.c_str())];
    }
  }
  getline(bI, line);
  
  
  
  // - 
  // BEGIN STOREFROM
  getline(bI, line);

  // STOREFROM VERTEX
  getline(bI, line);
  
  if (line.find("NULL") != string::npos)
    storeFrom = NULL;
  else
    storeFrom = bf->findOrCreateVP(line);
  
  getline(bI, line);

  
  
  //----
  //BEGIN PARAMS
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END PARAMS") != string::npos)
      proceed = false;
    else
    {
      VertexProps * field = bf->findOrCreateVP(line);
      params.insert(field);
      
      if (interestingVar)
      {
        bf->hasParams.insert(this);
      }
      
    }
  }
  
  //---
  //BEGIN CALLS
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END CALLS") != string::npos)
      proceed = false;
    else
    {
      char arr[100];
      int paramNum;
      sscanf(line.c_str(), "%s %d", arr, &paramNum);
      
      string s(arr);
      VertexProps *Node = bf->findOrCreateVP(s);
      FuncCall *fc = new FuncCall(paramNum, Node);      
      
      calls.push_back(fc);
    }
  }
  
  
  
  //----
  //BEGIN DOM_LN
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END DOM_LN") != string::npos)
      proceed = false;
    else
    {
      int ln = atoi(line.c_str());
      domLineNumbers.insert(ln);
    }
  }  
  
  
  
  
  //----
  //BEGIN LINENUMS
  getline(bI, line);

  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END LINENUMS") != string::npos)
      proceed = false;
    else
    {
      int ln = atoi(line.c_str());
      lineNumbers.insert(ln);
      descLineNumbers.insert(ln);
      bf->allLines.insert(pair<int, VertexProps *>(ln,this));
    }
  }

#ifdef SUPPORT_MULTILOCALE
  //BEGIN ISPID
  getline(bI, line);
  // get isPid
  getline(bI, line);
  isPid = atoi(line.c_str());
  //END ISPID
  getline(bI, line);

  //BEGIN ISOBJ
  getline(bI, line);
  // get isObj
  getline(bI, line);
  isObj = atoi(line.c_str());
  //END ISOBJ
  getline(bI, line);

  //BEGIN MYPID
  getline(bI, line);
  // get myPid
  getline(bI, line);  
  if (line.find("NULL") != string::npos)
  {
    myPid = NULL;
  }
  else
  {
    VertexProps *temp = bf->findOrCreateVP(line);
    myPid = temp;
  }    
  //END MYPID
  getline(bI, line);

  //BEGIN MYOBJ
  getline(bI, line);
  // get myObj
  getline(bI, line);  
  if (line.find("NULL") != string::npos)
  {
    myObj = NULL;
  }
  else
  {
    VertexProps *temp = bf->findOrCreateVP(line);
    myObj = temp;
  }    
  //END MYOBJ
  getline(bI, line);

  //----
  //BEGIN PIDALIASES
  getline(bI, line);
  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END PIDALIASES") != string::npos)
      proceed = false;
    else
    {
      VertexProps *temp = bf->findOrCreateVP(line);
      pidAliases.insert(temp);
    }
  }

  //----
  //BEGIN OBJALIASES
  getline(bI, line);
  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END OBJALIASES") != string::npos)
      proceed = false;
    else
    {
      VertexProps *temp = bf->findOrCreateVP(line);
      objAliases.insert(temp);
    }
  }
  //----
  //BEGIN PPAS
  getline(bI, line);
  proceed = true;
  while (proceed)
  {
    getline(bI, line);
    if (line.find("END PPAS") != string::npos)
      proceed = false;
    else
    {
      VertexProps *temp = bf->findOrCreateVP(line);
      PPAs.insert(temp);
    }
  }
#endif

  //END VAR 
  getline(bI, line);
  line.clear();
}






