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

#ifndef BLAME_MODULE_H
#define BLAME_MODULE_H 

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <set>

#include "Instances.h"
#include "VertexProps.h"
#include "BlameFunction.h"
/*
#ifdef __GNUC__
#include <ext/hash_map>
#else
#include <hash_map>
#endif
*/
using namespace std;

class BlameFunction;

struct ltFunc
{
  bool operator()(const BlameFunction* f1, const BlameFunction* f2) const
  {
    //return strcmp(s1, s2) < 0;                                                                                                                                                                                             
    return (f1->getBLineNum() < f2->getBLineNum()) && f1->getELineNum() < f2->getBLineNum();
  }
};

typedef std::unordered_map<std::string, BlameFunction *, std::hash<std::string>, eqstr> FunctionHash;


class BlameModule
{
 public:
  BlameModule() {dummyFunc = new BlameFunction(); }
  std::string getName() { return realName;}
  void printParsed(std::ostream &O);
  BlameFunction * findLineRange(int lineNum);
  void setName(std::string name) { realName = name;}
  void addFunction(BlameFunction * bf){blameFunctions[bf->getName()] = bf;}
  void addFunctionSet(BlameFunction * bf);
  BlameFunction *getFunction(std::string bfName){
      return blameFunctions[bfName];
  }
  set<BlameFunction *, ltFunc> funcsBySet;//not used anymore 03/29/16
  FunctionHash blameFunctions;
 
 private:
  std::string realName;
  //set<BlameFunction *, ltFunc> funcsBySet;
  BlameFunction * dummyFunc;
  
};

#endif
