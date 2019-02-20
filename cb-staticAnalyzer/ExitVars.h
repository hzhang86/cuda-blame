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

#ifndef _EXIT_VARS_H
#define _EXIT_VARS_H

#include <string>
#include <vector>
#include <set>
#include "NodeProps.h"

class NodeProps;

enum ExitTypes { PARAM, RET, GLOBAL, STATIC, UNDEFINED };

enum ExitProgramTypes { UNREAD_RET, BLAME_HOLDER };

class ExitSuper
{
public:
    void addVertex(NodeProps * p) {vertices.push_back(p);}
    std::string realName;
	NodeProps *vertex;
    // TODO: Change this to a set
    std::vector<NodeProps *> vertices;
    //std::set<int> lineNumbers;
	int lineNum;
    //Value *llvmNode; //keep the corresponding llvm node(mainly for params)
};


class ExitOutput : public ExitSuper
{
public:
	ExitOutput() {lineNum = 0; vertex=NULL;}
	~ExitOutput() { vertices.clear(); }

};

class ExitVariable : public ExitSuper
{
public: 
    ExitVariable(std::string name){realName = name; lineNum = 0; 
                                    vertex = NULL; /*llvmNode = NULL;*/}
    ExitVariable(std::string name, ExitTypes e, int wp, bool isSP) 
        {realName = name; et = e; whichParam = wp; lineNum = 0; vertex=NULL; isStructPtr = isSP; /*llvmNode = v;*/}
	~ExitVariable() { vertices.clear(); }
    ExitTypes et;
    int whichParam;
	bool isStructPtr;
};

class ExitProgram : public ExitSuper //refers call_node of output function like printf, needs to check writeln
{
public: 
    ExitProgram(std::string name){realName = name; lineNum = 0;}
    ExitProgram(std::string name, ExitProgramTypes e) 
		{realName = name; et = e; lineNum=0; vertex=NULL;}
	~ExitProgram() {  vertices.clear(); }
    ExitProgramTypes et;
    bool isUnusedReturn;
  
    // For cases where we have a single Vertex it refers to (as opposed to vertices)
    NodeProps * pointsTo;
};

#endif
