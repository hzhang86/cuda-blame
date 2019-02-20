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

/* This file is not used currently
 * just keep it for future use
*/
#ifndef _UTIL_H
#define _UTIL_H

#include <iostream>
#include <string.h>
#include <time.h>
#include <sys/time.h>
//#include "Instances.h"

enum func_type {KERNEL, DEVICE, HOST};

struct eqstr
{
  bool operator() (std::string s1, std::string s2) const {
	return s1 == s2;
  }

};

bool isForkStarWrapper(std::string name);

bool isTopMainFrame(std::string name);

std::string getFileName(std::string &rawName); 

std::string getLineWithout_a(std::string origStr);

void my_timestamp();
#endif //_UTIL_H
