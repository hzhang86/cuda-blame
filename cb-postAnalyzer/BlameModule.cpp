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

#include "BlameModule.h"
#include "BlameFunction.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void BlameModule::printParsed(std::ostream &O)
{
    O<<"##Module "<<realName<<std::endl;
    //std::vector<BlameFunction *>::iterator bf_i;
    FunctionHash::iterator bf_i;
  
    for (bf_i = blameFunctions.begin();  bf_i != blameFunctions.end(); bf_i++)
  {
    BlameFunction * bf = (*bf_i).second;
    bf->printParsed(O);
  }  
}

void BlameModule::addFunctionSet(BlameFunction * bf)
{
  //cout<<"Module "<<getName()<<" ---- ";
  if (funcsBySet.count(bf) > 0)
  {
    //cout<<"Conflict: "<<funcsBySet.count(bf)<<"   "<<bf->getName()<<"("<<bf->getBLineNum()<<"-"<<bf->getELineNum()<<") -- ";
    
    set<BlameFunction *>::iterator set_bf_i;
    set_bf_i = funcsBySet.find(bf);
    BlameFunction * tempBF = *set_bf_i;
    
    //cout<<tempBF->getName()<<"("<<tempBF->getBLineNum()<<"-"<<tempBF->getELineNum()<<")"<<endl;
  
  
    int tcRange = bf->getELineNum() - bf->getBLineNum();
    int currRange = tempBF->getELineNum() - tempBF->getBLineNum();
  
    // big module from FORTRAN, probably a mistake, lets change it
    if (currRange > tcRange)
    {
      tempBF->setBLineNum(bf->getELineNum()+1);
      cout<<"Modifying(2) "<<tempBF->getName()<<"("<<tempBF->getBLineNum()<<"-"<<tempBF->getELineNum()<<")"<<std::endl;    
      cout<<"Inserting(2) "<<bf->getName()<<"("<<bf->getBLineNum()<<"-"<<bf->getELineNum()<<")"<<std::endl;
      addFunctionSet(bf);
    }
    else if (tcRange > currRange)
    {
      // New one is bigger, leave small one and adjust big one
      bf->setBLineNum(tempBF->getELineNum()+1);
      cout<<"Modifying(4) "<<bf->getName()<<"("<<bf->getBLineNum()<<"-"<<bf->getELineNum()<<")"<<std::endl;    
      cout<<"Inserting(4) "<<bf->getName()<<"("<<bf->getBLineNum()<<"-"<<bf->getELineNum()<<")"<<std::endl;
      addFunctionSet(bf);
    }
    else if (tcRange == currRange)
    {
      if (bf->allLines.size() > tempBF->allLines.size())
      {
        cout<<"Erasing(3) "<<tempBF->getName()<<"("<<tempBF->getBLineNum()<<"-"<<tempBF->getELineNum()<<")"<<std::endl;
        funcsBySet.erase(tempBF);
        funcsBySet.insert(bf);      
        cout<<"Inserting(3) "<<bf->getName()<<"("<<bf->getBLineNum()<<"-"<<bf->getELineNum()<<")"<<std::endl;
      }
    }
  }
  else
  {
    funcsBySet.insert(bf);      
    cout<<"Inserting(1) "<<bf->getName()<<"("<<bf->getBLineNum()<<"-"<<bf->getELineNum()<<")"<<std::endl;
  }
}


BlameFunction *BlameModule::findLineRange(int lineNum)
{
  BlameFunction *topCand = NULL;
  FunctionHash::iterator bf_i;
  
  for (bf_i = blameFunctions.begin();  bf_i != blameFunctions.end(); bf_i++) {
    BlameFunction *bf = (*bf_i).second;
    if (bf && bf->allLineNums.size() && bf->allLineNums.count(lineNum)) {
      if (topCand == NULL) {
        topCand = bf;
      }
      else {
        int tcDiff = topCand->getELineNum() - lineNum;
        int currDiff = bf->getELineNum() - lineNum;
    
        cout<<"Another function has this line("<<lineNum<<"), tcDiff="<<tcDiff<<" currDiff="<<currDiff<<endl;
        if (currDiff > tcDiff) //TOCHECK: to avoid matching the generated wrapcoforall/coforall func, we need '>'
          topCand = bf;
      }
    }
  }
  
  if (topCand == NULL)
    cout<<"Top Cand is NULL -- oh noes!!! for line#: "<<lineNum<<" file: "<<getName()<<endl;
  else
    ;//cout<<"Find func: "<<topCand->getName()<<" for lineNum: "<<lineNum<<endl;
  
  return topCand;
}


