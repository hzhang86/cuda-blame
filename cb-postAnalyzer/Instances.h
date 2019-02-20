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

#ifndef INSTANCES_H
#define INSTANCES_H 


#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <string.h>
#include <set>
#include <unordered_map>
#include <utility>

#include "BlameStruct.h"
#include "util.h"
/*
#ifdef __GNUC__
#include <ext/hash_map>
#else
#include <hash_map>
#endif

namespace std
{
  using namespace __gnu_cxx;
}
*/


#define COMPUTE_INST    0
#define PRESPAWN_INST   1
#define FORK_INST       2
#define FORK_NB_INST    3
#define FORK_FAST_INST  4

class BlameModule;

typedef std::unordered_map<std::string, BlameModule *, std::hash<std::string>, eqstr> ModuleHash;
typedef std::unordered_map<std::string, StructBlame *, std::hash<std::string>, eqstr> StructHash;
typedef std::unordered_map<int, StructField*> FieldHash;

struct fork_t 
{
  int callerNode;
  int calleeNode;
  int fid;
  int fork_num;

  fork_t() : callerNode(-1),calleeNode(-1),fid(-1),fork_num(-1) {}
  fork_t(int loc, int rem, int fID, int f_num) : 
    callerNode(loc),calleeNode(rem),fid(fID),fork_num(f_num) {}

};

struct StackFrame
{
  int lineNumber;
  int frameNumber;
  std::string moduleName;
  unsigned long address;
  std::string frameName;
  bool toRemove = false;
};

struct Instance
{
  std::vector<StackFrame> frames;
  int instNum; //denote which inst in this CPU file
  unsigned int correlationID = 0; //for CPU before cudaLaunch
  double share = 1; //denotes the share of this inst in a sample
  unsigned int occurance = 0; //used when glued with gpu samples
  
  void printInstance();
  void printInstance_concise();
  void handleInstance(ModuleHash & modules, std::ostream &O, int InstanceNum, bool verbose);
  void handleRuntimeInst(std::ostream &O, int InstanceNum, bool verbose);
  //void trimFrames(ModuleHash &modules, int InstanceNum, std::string nodeName); 
};

struct FullSample
{
  set<int> instanceNumbers;  // keep track of all the matching ones so we can compare against 
  /// which set to merge using a nearest neighbor style approach
  Instance * i;  // representative instance, we only record one representative one
  int frameNumber;
  int lineNumber;
};

 //NOTE: pair.second is NOT a pointer !   
typedef std::unordered_map<int, Instance> instanceMap;
typedef std::unordered_map<std::string, std::vector<Instance>, std::hash<std::string>, eqstr> compInstanceHash;
typedef std::unordered_map<std::string, instanceMap, std::hash<std::string>, eqstr> preInstanceHash;
typedef std::unordered_map<std::string, std::vector<Instance>, std::hash<std::string>, eqstr> forkInstanceHash;

//we keep a map between the chpl_nodeID and real compute node names
typedef std::unordered_map<int, std::string> nodeHash;

//typedef std::unordered_map<const fork_t, Instance, key_hash, key_equal> globalForkInstMap; //all fork instances from all nodes

#endif
