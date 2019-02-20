/*
 *  Copyright 2014-2018 Hui Zhang
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

#ifndef _FUNCTION_H
#define _FUNCTION_H 

#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>
#include <set>
#include "util.h"
//enum func_type {KERNEL, DEVICE, HOST};

class CCTNode;

class Caller
{
  public:
    int line; //the line "this" caller called its callee
    std::string file;
    CCTNode *callerFunc;

    Caller(int ln, std::string fn, CCTNode *caller) {
      line = ln;
      file = fn;
      callerFunc = caller;
    }
};

class CCTNode
{
  public:
    std::string linkName;
    std::string realName;
    std::string moduleName;
    func_type   ft;
    
    std::set<std::string> children; //simply store truncate calls (e.g., if we have foo--6 and foo-6a, we only store foo--6)
    std::vector<Caller*> parents; //2 diff callsites of this by a same caller would make up 2 entries

    CCTNode(std::string fn) {
      linkName = fn;
    }

    ~CCTNode() {
      children.clear();
      parents.clear();
    }
};

typedef std::unordered_map<std::string, CCTNode*, std::hash<std::string>, eqstr>  CCTMap;
#endif
