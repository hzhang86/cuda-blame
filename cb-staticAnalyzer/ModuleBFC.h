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

#ifndef _MODULE_BFC_H
#define _MODULE_BFC_H

#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Compiler.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Dwarf.h" 
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
//#include "llvm/Analysis/DominatorInternals.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Instruction.def"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/DebugInfo.h"

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
//Added by Hui to substitute hash_map
#include <unordered_map>

#include "Parameters.h"
//#include "FunctionBFC.h"
using namespace llvm;

struct StructBFC;

struct StructField {

    std::string fieldName;
	int fieldNum;
	const llvm::Type * llvmType; // not reliable
	std::string typeName;  // different from LLVM Type

	StructBFC * parentStruct;

	StructField(int fn){fieldNum = fn;}

};

struct StructBFC {

    std::string structName;
	std::vector<StructField *> fields;
	
	std::string moduleName;
	std::string modulePathName;
	
	int lineNum;
	
	void setModuleNameAndPath(llvm::DIScope *contextInfo);
	//void getPathOrName(Value * v, bool isName);
	//void setModuleName(std::string rawName);
	//void setModulePathName(std::string rawName);
	
};


// Comparators //
struct eqstr { //used in unordered_map, key type should be string
    bool operator()(std::string s1, std::string s2) const {
        return s1 == s2;
    }
};

struct ltstr { //used in set, key type can be const char*
    bool operator()(const char *s1, const char *s2) const {
        return strcmp(s1, s2) < 0;
    }
};

typedef std::unordered_map<std::string, StructBFC*, std::hash<std::string>, eqstr> StructBFCHash;


struct ModuleBFC {

	Module * M;
    
    std::vector<std::string> funcPtrTable;
	
	ModuleBFC(Module * mod) {M = mod;}

	void addStructBFC(StructBFC * sb);
	
	bool parseDerivedType(DIType *dt, StructBFC *sb, StructField *sf, bool isField);
	bool parseCompositeType(DIType *dt, StructBFC *sb, bool isPrimary);
	void parseDITypes(DIType* dt);

	void printStructs();
	void exportStructs(std::ostream & O);
    void exportOneStruct(std::ostream &O, StructBFC *sb);
	//void exportUserFuncsInfo(FuncSigHash &kFI, std::ostream &O);//added 03/16/17
	StructBFC* structLookUp(std::string & sName);
    StructBFC* findOrCreatePidArray(std::string pidArrayName, int numElems, 
                                                const llvm::Type *sbPointT);
    

	void handleStructs();
	//std::vector<StructBFC *> structs;
    StructBFCHash structs;

};


#endif
