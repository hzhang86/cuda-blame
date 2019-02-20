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

#ifndef BLAME_STRUCT_H
#define BLAME_STRUCT_H
 
 
#include <string>

//#include "Instances.h"

#include <vector>
#include <map>
#include <string>
#include <iostream>

#include <set>
/*
#ifdef __GNUC__
#include <ext/hash_map>
#else
#include <hash_map>
#endif
*/
#include <unordered_map>


namespace std
{
  using namespace __gnu_cxx;
}

using namespace std;



struct StructBlame; 
 
struct StructField {

	string fieldName;
	int fieldNum;
	string fieldType;
	//const llvm::Type * llvmType;

	StructBlame * parentStruct;

	StructField() {}
	StructField(int fn){fieldNum = fn;}

};

typedef std::unordered_map<int, StructField *> FieldHash;


struct StructBlame {

	string structName;
	//vector<StructField *> fields;
	FieldHash fields;
	
	
	string moduleName;
	string modulePathName;
	
	int lineNum;
	
	void parseStruct(ifstream & bI);
	
	//void grabModuleNameAndPath(llvm::Value * compUnit);
	//void getPathOrName(Value * v, bool isName);

	
};

#endif

