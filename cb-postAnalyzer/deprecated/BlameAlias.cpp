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

#include "BlameProgram.h"
#include "BlameFunction.h"
#include "BlameModule.h"
#include "BlameStruct.h"
#include "Instances.h"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

using namespace std;


void BlameProgram::parseLoops_OA()
{
	std::vector<std::string>::iterator vec_str_i;
	for (vec_str_i = blameLoopFiles.begin(); vec_str_i != blameLoopFiles.end(); vec_str_i++)
	{
		ifstream bI((*vec_str_i).c_str());
		
		std::cout<<"Parsing Loop file "<<(*vec_str_i)<<std::endl;
		
		
		string line;
		while(getline(bI,line))
		{
		
			//size_t foundBeg;
			
			//foundBeg = line.find("BEGIN FUNC");
			
			BlameFunction * bf = NULL;
			
			stringstream ss(line);
			string buf, funcName, moduleName, modulePathName;
			
			// BEGIN     FUNC          <funcName>       <moduleName>       <lineNumB> <lineNumE> <modulePath>
			ss>>buf;    ss>>buf;    ss>>funcName;  ss>>moduleName;  ss>> buf;    ss>>buf;     ss>>modulePathName;       
			
			bf = getOrCreateBlameFunction(funcName, moduleName, modulePathName);
  
			std::cout<<"Parsing function "<<funcName<<" for loops."<<std::endl;
			bf->addLoops_OA(bI);
			
		}
	}


}


void BlameFunction::addLoops_OA(ifstream & bI)
{
	string line;
	
	// either BEGIN LOOPS or END FUNC
	getline(bI, line);

	size_t foundBeg = line.find("BEGIN LOOPS");
	
	// We have not found it, so this function does not have loops
	if (foundBeg == std::string::npos)
		return;

	stringstream ss(line);
	string buf;
	int numLoops;
	
	// BEGIN    LOOPS      <numLoops>
	ss>>buf;  ss>>buf;  ss>>numLoops;

	LoopInfo * li = NULL;
	for (int a = 0; a < numLoops; a++)
	{
		getline(bI, line);
		stringstream ss2(line);
		
		LoopInfo * li = new LoopInfo();
	    // <num elements> L <el1> <el2> ...
		int numElements;
		ss2>>numElements;  ss2>>buf;
		
		cout<<"Loop of size "<<numElements<<std::endl;
		
		for (int b = 0; b < numElements; b++)
		{
			int val;
			ss2>>val;
			li->lineNumbers.insert(val);
		}
		
		loops.push_back(li);
	}
	
	getline(bI, line);
}

// write replace works for read
void BlameProgram::parseProgram_OA()
{
	std::vector<std::string>::iterator vec_str_i;
	for (vec_str_i = blameExportFiles.begin(); vec_str_i != blameExportFiles.end(); vec_str_i++)
	{
		ifstream bI((*vec_str_i).c_str());
		
		std::cout<<"Parsing file "<<(*vec_str_i)<<std::endl;
		
		string line;
		while(getline(bI,line))
		{
			size_t foundBeg;
			
			foundBeg = line.find("BEGIN FUNC");
			
			BlameFunction * bf;
			
			if (foundBeg != std::string::npos)
			{
				std::string funcName, moduleName, modulePathName;
				
				//std::cout<<endl<<endl;
				//BEGIN_F_NAME
				getline(bI, line);
				
				// Actual Nmae
				getline(bI, line);
				//std::cout<<"Parsing Function "<<line<<std::endl;
				funcName = line;
				
				// END F_NAME
				getline(bI, line);
				
				
				//----
				// BEGIN M_NAME_PATH
				getline(bI, line);
				
				//std::cout<<"Should be M_NAME_PATH -- is -- "<<line<<std::endl;
				
				// Get Module Path Name()
				getline(bI, line);
				//setModulePathName(line);
				modulePathName = line;
				
				// END M_NAME_PATH
				getline(bI, line);
				
				//----
				// BEGIN M_NAME
				getline(bI, line);
				
				// Get Module Name
				getline(bI, line);
				//setModuleName(line);
				moduleName = line;
				
				// END M_NAME
				getline(bI, line);
				
				bf = getOrCreateBlameFunction(funcName, moduleName, modulePathName);
				//bf = new BlameFunction(line);
				
				bf->BP = this;
				
				bool isMain = false;	
				
				// TODO: Make this determined at runtime
				if (funcName.find("main") != std::string::npos 
						&& funcName.length() == 4)
				{
					//std::cout<<"Setting main as a blame point"<<std::endl;
					isMain = true;
				}
				
				
				// This will not return NULL if the function in question is one
				//  of the sampled functions from the run
				if (bf->parseBlameFunction_OA(bI) != NULL)
				{
					// TODO: Optimize this
					//bf->calcAggregateLN();
					bf->isFull = true;
					
					//addFunction(bf);	
					
					if (isMain)
						bf->setBlamePoint(EXPLICIT);
				}
			}
		}
	}
}


// write replace works for read
BlameFunction * BlameFunction::parseBlameFunction_OA(ifstream & bI)
{
	std::string line;
	
	if (BP->sampledModules.count(moduleName.c_str()))
	{
		std::cout<<"MODULE "<<moduleName<<" FOUND as one containing a sampled function "<<line<<std::endl;
	}
	else
	{
		//std::cout<<"MODULE NOT FOUND, no sampled function - "<<moduleName<<" "<<strlen(moduleName.c_str())<<std::endl;
		return NULL;
	}
	
	//----
	//BEGIN F_B_LINENUM
	getline(bI, line);
	
	// Begin Line Num
	getline(bI, line);
	int bln = atoi(line.c_str());
	setBLineNum(bln);
	
	//END F_B_LINENUM
	getline(bI, line);
	
	//----
	//BEGIN F_E_LINENUM
	getline(bI, line);
	
	// End Line Num
	getline(bI, line);
	int eln = atoi(line.c_str());
	setELineNum(eln);
	
	//END F_E_LINENUM
	getline(bI, line);
	
	//----
	//BEGIN F_BPOINT
	getline(bI, line);
	
	// End Line Num
	getline(bI, line);
	int bPoint = atoi(line.c_str());
	setBlamePoint(bPoint);
	
	//END F_BPOINT
	getline(bI, line);
	
	//----
	bool proceed = true;
	while (proceed)
	{
		// SHOULD EITHER GET "BEGIN VAR" or "END FUNC"
		getline(bI, line);
		if (line.find("BEGIN VAR") != std::string::npos)
		{
			// BEGIN V_NAME
			getline(bI, line);
			
			// Get Variable Name
			getline(bI, line);
			VertexProps * vp = findOrCreateVP(line);
			
			// END V_NAME
			getline(bI, line);
			
			vp->parseVertex_OA(bI, this);
			vp->adjustVertex();
		}
		else
			proceed = false;
	}
	return this;
}


// write replace works for read
void VertexProps::parseVertex_OA(ifstream & bI, BlameFunction * bf)
{
	std::string line;
	bool proceed = true;
	
	//----
	//BEGIN V_TYPE
	getline(bI, line);
	
		
	// get variable type
	getline(bI, line);
	
	std::cout<<"Parsing Var type for "<<name<<" "<<line.c_str()<<std::endl;

	eStatus = atoi(line.c_str());
	
	//std::cout<<"Estatus is "<<eStatus<<std::endl;
	
	//END V_TYPE
	getline(bI, line);


	if (eStatus >= EXIT_VAR_GLOBAL)
		bf->addExitVar(this);
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
	//std::cout<<"Parsing nStatus line "<<line.c_str()<<std::endl;
	
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

	//std::cout<<"Estatus is "<<eStatus<<std::endl;



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
		if (line.find("END CHILDREN") != std::string::npos)
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
		if (line.find("END ALIASES") != std::string::npos)
			proceed = false;
		else
		{
			VertexProps * alias = bf->findOrCreateVP(line);
			aliases.insert(alias);
			alias->aliasUpPtr = this;
		}
	}



	//----
	//BEGIN DATAPTRS
	getline(bI, line);

	proceed = true;
	while (proceed)
	{
		getline(bI, line);
		if (line.find("END DATAPTRS") != std::string::npos)
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
		if (line.find("END DFALIAS") != std::string::npos)
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
		if (line.find("END DFCHILDREN") != std::string::npos)
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
		if (line.find("END RESOLVED_LS") != std::string::npos)
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
		if (line.find("END RESOLVEDLS_FROM") != std::string::npos)
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
		if (line.find("END RESOLVEDLS_SE") != std::string::npos)
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
		if (line.find("END STORES_TO") != std::string::npos)
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
		if (line.find("END FIELDS") != std::string::npos)
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
	
	if (line.find("NULL") != std::string::npos)
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
	
	if (line.find("NULL") != std::string::npos)
		sType = NULL;
	else
	{
		if (BF == NULL)
		{
			std::cerr<<"BF is NULL"<<std::endl;
			sType = NULL;
		}
		else
		{
			if (BF->BP == NULL)
			{
				std::cerr<<"BF->BP is NULL"<<std::endl;
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
	
	//std::cout<<"SP - "<<line<<std::endl;
	if (line.find("NULL") != std::string::npos)
		bs = NULL;
	else
	{
		if (BF == NULL)
		{
			std::cerr<<"BF is NULL"<<std::endl;
			bs = NULL;
		}
		else
		{
			if (BF->BP == NULL)
			{
				std::cerr<<"BF->BP is NULL"<<std::endl;
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
	// BEGIN STRUCTFIELD
	getline(bI, line);

	//StructField * sField;
	getline(bI, line);
	
	//std::cout<<"SF - "<<line<<std::endl;
	if (line.find("NULL") != std::string::npos)
		sField = NULL;
	else
	{
		if (bs == NULL)
		{
			std::cerr<<"bs is NULL"<<std::endl;
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
	
	if (line.find("NULL") != std::string::npos)
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
		if (line.find("END PARAMS") != std::string::npos)
			proceed = false;
		else
		{
			VertexProps * field = bf->findOrCreateVP(line);
			params.insert(field);
		}
	}
	
	//---
	//BEGIN CALLS
	getline(bI, line);

	proceed = true;
	while (proceed)
	{
		getline(bI, line);
		if (line.find("END CALLS") != std::string::npos)
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
		if (line.find("END DOM_LN") != std::string::npos)
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
		if (line.find("END LINENUMS") != std::string::npos)
			proceed = false;
		else
		{
			int ln = atoi(line.c_str());
			lineNumbers.insert(ln);
		}
	}	
	
	//----
	//BEGIN READ_L_NUMS
	getline(bI, line);

	proceed = true;
	while (proceed)
	{
		getline(bI, line);
		if (line.find("END READ_L_NUMS") != std::string::npos)
			proceed = false;
		else
		{
			int lineNum = 0;
			int lineCounts = 0;
			
			cout<<"LINE ---- "<<line.c_str()<<endl;
			sscanf(line.c_str(), "%d %d", &lineNum, &lineCounts);
			
			readLines[lineNum] = lineCounts;
			
			cout<<"Adding lineCounts "<<lineCounts<<" to lineNum "<<lineNum<<" for  variable "<<name<<endl;
			
			lineReadNumbers.insert(lineNum);
		}
	}	
	
	
	
	// END VAR
	getline(bI, line);
	line.clear();
	
}



// replace for both
void BlameFunction::resolveLineNum_OA(vector<StackFrame> & frames, ModuleHash & modules,
																	 vector<StackFrame>::iterator vec_SF_i, std::set<VertexProps *> & blamees,
																	 short isBlamePoint, bool isBottomParsed, BlameFunction * oldFunc, std::ostream &O)
{
	
	//std::set<VertexProps *> blamees;
	
	clearPastData();
	
	std::set<VertexProps *> localBlamees;
	std::set<VertexProps *> DQblamees;
	
	O<<"FRAME(OA)# "<<(*vec_SF_i).frameNumber<<" ";
	O<<getFullContextName(frames, modules, vec_SF_i);
	//O<<getName()<<" "<<getModuleName()<<" "<<getModulePathName()<<" ";
	O<<" "<<getModuleName()<<" "<<getModulePathName()<<" ";
	O<<(*vec_SF_i).lineNumber<<" "<<isBlamePoint<<" ";
	O<<beginLineNum<<" "<<endLineNum<<std::endl;
	
	std::cout<<getName()<<" LN(OA) -  "<<(*vec_SF_i).lineNumber<<" BP - ";
	std::cout<<isBlamePoint<<" Frame(OA) # "<<(*vec_SF_i).frameNumber<<std::endl;
	
	
	// Dominator call information for aliases
	populateTempSideEffects((*vec_SF_i).lineNumber, blamees);
	
	// Populate side effect relations for the blame function
	populateTempSideEffectsRelations();
	
	// We're not a blame point and we're in the middle of the stack,
	//   we're going to have to apply a transfer function to get out of this
	if (isBlamePoint == NO_BLAME && isBottomParsed == false)
	{
		std::vector<VertexProps *>::iterator vec_vp_i;
		
		VertexProps * callNode = NULL;
		
		std::vector<VertexProps *> matchingCalls;
		
		// We're only concerned with transfer functions so we look through the call nodes
		for (vec_vp_i = callNodes.begin(); vec_vp_i != callNodes.end(); vec_vp_i++)
		{
			VertexProps * vp = *vec_vp_i;
			
			if (vp->declaredLine == (*vec_SF_i).lineNumber)
			{
				std::cout<<"TRANSFER FUNC -- Line number(2) found in VP "<<vp->name<<std::endl;
				matchingCalls.push_back(vp);
				//atLeastOneFound = true;
			}
		}
		if (matchingCalls.size() == 0)
		{
			std::cerr<<"TRANSFER FUNC BA.cpp:874 - No lineNums found for sample in "<<getName();
			std::cerr<<" Line Num "<<(*vec_SF_i).lineNumber<<" Frame Number  "<<(*vec_SF_i).frameNumber<<std::endl;
			std::cout<<"TF(2) - No linenums found "<<std::endl;
		}
		else if (matchingCalls.size() == 1)
		{
			// apply transfer function
			//VertexProps * callNode = matchingCalls.front();
			callNode = matchingCalls.front();
			handleTransferFunction_OA(callNode, blamees); 
			std::set<VertexProps *>  oldBlamees = blamees;
			
			blamees.clear();
			determineBlameHolders_OA(blamees, oldBlamees, callNode, (*vec_SF_i).lineNumber, isBlamePoint, localBlamees, DQblamees);
			calcParamInfo_OA(blamees, callNode);
			
			// Go through and add temporary fields to the blamees from prior frames
			// Example
			// Frame #0
			// EF s1_addr.b
			// EV s1_addr
			// Frame #1
			// U  s1
			//  ... would be come
			// Frame #1
			//  U s1
			//  GF s1.b
			addTempFieldBlamees_OA(blamees, oldBlamees);
			clearTempFields(oldBlamees, oldFunc);
			
			outputFrameBlamees_OA(blamees, localBlamees, DQblamees, O);			
			vec_SF_i++;
			if (vec_SF_i == frames.end())
				return;
			
			if ((*vec_SF_i).lineNumber > 0)
			{
				cout<<"*****At Frame(3) "<<(*vec_SF_i).frameNumber<<" at line num "<<(*vec_SF_i).lineNumber;
				cout<<" in module "<<(*vec_SF_i).moduleName<<endl;
				
				// Get the module from the debugging information
				BlameModule * bm = modules[(*vec_SF_i).moduleName];
				
				if (bm != NULL)
				{
					// Use the combination of the module and the line number to determine the function
					BlameFunction * bf = bm->findLineRange((*vec_SF_i).lineNumber);
					
					if (bf)
					{
						bf->resolveLineNum_OA(frames, modules, vec_SF_i, blamees, 
															 bf->getBlamePoint(), false, this, O);
					}
				}
			}
		}
		else  // More than one matching call (two func calls on one line number in code)
		{
			callNode = NULL;
			std::cout<<"More than one call node at that line number"<<std::endl;
			// figure out which call is appropriate, then apply transfer function
			
			
			vector<StackFrame>::iterator minusOne = vec_SF_i - 1;
			//StackFrame * sfCheck = minusOne;
			BlameModule * bmCheck = modules[(*minusOne).moduleName];
			
			if (bmCheck == NULL)
			{
				std::cerr<<"BM null when differntiating calls"<<std::endl;
				return;
			}
			
			
			BlameFunction * bfCheck = bmCheck->findLineRange((*minusOne).lineNumber);
			if (bfCheck == NULL)
			{
				std::cerr<<"BF null when differentiating calls"<<std::endl;
				return;
			}
			
			std::vector<VertexProps *>::iterator vec_vp_i2;
			for (vec_vp_i2 = matchingCalls.begin(); vec_vp_i2 != matchingCalls.end(); vec_vp_i2++)
			{
				VertexProps * vpCheck = *vec_vp_i2;
				std::cout<<vpCheck->name<<"  "<<bfCheck->getName()<<std::endl;
				//if (vpCheck->name == bfCheck->getName())
				//	callNode = vpCheck;
				
				// Look for subsets since vpCheck will have the line number concatenated
				if (vpCheck->name.find(bfCheck->getName()) != std::string::npos)
					callNode = vpCheck;
				
			}
			
			if (callNode == NULL)
			{
				for (vec_vp_i2 = matchingCalls.begin(); vec_vp_i2 != matchingCalls.end(); vec_vp_i2++)
				{
					VertexProps * vpCheck = *vec_vp_i2;
					std::cout<<vpCheck->name<<"  (tmpCheck)  "<<bfCheck->getName()<<std::endl;
					
					// Look for function pointers resolved at runtime
					if (vpCheck->name.find("tmp") != std::string::npos)
						callNode = vpCheck;
				}
			}
			
			if (callNode == NULL)
			{
				std::cout<<"No matching call nodes from multiple matches"<<std::endl;
				std::cerr<<"No matching call nodes from multiple matches"<<std::endl;
				return;
			}
			
			
			std::cout<<"Call node that matched is "<<callNode->name<<std::endl;
			handleTransferFunction_OA(callNode, blamees); 
			std::set<VertexProps *>  oldBlamees = blamees;
			blamees.clear();
			determineBlameHolders_OA(blamees, oldBlamees, callNode, (*vec_SF_i).lineNumber, isBlamePoint, localBlamees, DQblamees);
			calcParamInfo_OA(blamees, callNode);
			
			
			// Go through and add temporary fields to the blamees from prior frames
			// Example
			// Frame #0
			// EF s1_addr.b
			// EV s1_addr
			// Frame #1
			// U  s1
			//  ... would be come
			// Frame #1
			//  U s1
			//  GF s1.b
			addTempFieldBlamees_OA(blamees, oldBlamees);
			
			clearTempFields(oldBlamees, oldFunc);
			
			
			outputFrameBlamees_OA(blamees, localBlamees, DQblamees, O);			
			vec_SF_i++;
			if (vec_SF_i == frames.end())
				return;
			
			if ((*vec_SF_i).lineNumber > 0)
			{
				cout<<"*****At Frame(4) "<<(*vec_SF_i).frameNumber<<" at line num "<<(*vec_SF_i).lineNumber;
				cout<<" in module "<<(*vec_SF_i).moduleName<<endl;
				
				// Get the module from the debugging information
				BlameModule * bm = modules[(*vec_SF_i).moduleName];
				
				if (bm != NULL)
				{
					cout<<"Checking(4) BM line range "<<bm->getName()<<std::endl;
					// Use the combination of the module and the line number to determine the function
					BlameFunction * bf = bm->findLineRange((*vec_SF_i).lineNumber);
					
					if (bf)
					{
						cout<<"Checking(4) BF line range "<<bf->getName()<<std::endl;
						bf->resolveLineNum_OA(frames, modules, vec_SF_i, blamees, 
															 bf->getBlamePoint(), false, this, O);
					}
				}
				else
				{
					cout<<"BM(4) is NULL"<<std::endl;
				}
			}
		}
	}
	// We don't apply a transfer function as it's the bottom of the readable stack
	else if (isBottomParsed == true)
	{
		std::cout<<"RLN (2) "<<std::endl;
		
		VertexProps * callNode = NULL;
		std::set<VertexProps *>  oldBlamees = blamees;
		determineBlameHolders_OA(blamees, oldBlamees, callNode,(*vec_SF_i).lineNumber, isBlamePoint, localBlamees, DQblamees);
		calcParamInfo_OA(blamees, callNode);
		
		
		outputFrameBlamees_OA(blamees, localBlamees, DQblamees, O);		
		vec_SF_i++;
		if (vec_SF_i == frames.end())
			return;
		
		if ((*vec_SF_i).lineNumber > 0)
		{
			cout<<"*****At Frame(1) "<<(*vec_SF_i).frameNumber<<" at line num "<<(*vec_SF_i).lineNumber;
			cout<<" in module "<<(*vec_SF_i).moduleName<<endl;
			
			// Get the module from the debugging information
			BlameModule * bm = modules[(*vec_SF_i).moduleName];
			
			if (bm != NULL)
			{
				// Use the combination of the module and the line number to determine the function
				BlameFunction * bf = bm->findLineRange((*vec_SF_i).lineNumber);
				
				if (bf)
				{
					bf->resolveLineNum_OA(frames, modules, vec_SF_i, blamees, 
														 bf->getBlamePoint(), false, this, O);
				}
			}
		}		
		
		
		
	}
	else if (isBlamePoint > NO_BLAME)
	{
		std::cout<<"RLN (3)"<<std::endl;
		
		
		
		std::vector<VertexProps *>::iterator vec_vp_i;
		
		VertexProps * callNode = NULL;
		
		std::vector<VertexProps *> matchingCalls;
		// We're only concerned with transfer functions so we look through the call nodes
		for (vec_vp_i = callNodes.begin(); vec_vp_i != callNodes.end(); vec_vp_i++)
		{
			VertexProps * vp = *vec_vp_i;
			if (vp->lineNumbers.count((*vec_SF_i).lineNumber))
			{
				std::cout<<"TRANSFER FUNC -- Line Number found in VP "<<vp->name<<std::endl;
				matchingCalls.push_back(vp);
				//atLeastOneFound = true;
			}
			else if (vp->declaredLine == (*vec_SF_i).lineNumber)
			{
				std::cout<<"TRANSFER FUNC -- Line number(2) found in VP "<<vp->name<<std::endl;
				matchingCalls.push_back(vp);
				//atLeastOneFound = true;
			}
		}
		if (matchingCalls.size() == 0)
		{
			std::cerr<<"TRANSFER FUNC BA.cpp:1117 - No lineNums found for sample in "<<getName();
			std::cerr<<" Line Num "<<(*vec_SF_i).lineNumber<<" Frame Number  "<<(*vec_SF_i).frameNumber<<std::endl;
			std::cout<<"TF(1) - No line nums found"<<std::endl;
		}
		else if (matchingCalls.size() == 1)
		{
			// apply transfer function
			callNode = matchingCalls.front();
			handleTransferFunction_OA(callNode, blamees); 
		}
		else  // More than one matching call (two func calls on one line number in code)
		{
			callNode = NULL;
			std::cout<<"More than one call node at that line number"<<std::endl;
			// figure out which call is appropriate, then apply transfer function
			
			
			vector<StackFrame>::iterator minusOne = vec_SF_i - 1;
			//StackFrame * sfCheck = minusOne;
			BlameModule * bmCheck = modules[(*minusOne).moduleName];
			
			if (bmCheck == NULL)
			{
				std::cerr<<"BM null when differntiating calls"<<std::endl;
				return;
			}
			
			
			BlameFunction * bfCheck = bmCheck->findLineRange((*minusOne).lineNumber);
			if (bfCheck == NULL)
			{
				std::cerr<<"BF null when differentiating calls"<<std::endl;
				return;
			}
			
			std::vector<VertexProps *>::iterator vec_vp_i2;
			for (vec_vp_i2 = matchingCalls.begin(); vec_vp_i2 != matchingCalls.end(); vec_vp_i2++)
			{
				VertexProps * vpCheck = *vec_vp_i2;
				std::cout<<vpCheck->name<<"  "<<bfCheck->getName()<<std::endl;
				//if (vpCheck->name == bfCheck->getName())
				//	callNode = vpCheck;
				
				// Look for subsets since vpCheck will have the line number concatenated
				if (vpCheck->name.find(bfCheck->getName()) != std::string::npos)
					callNode = vpCheck;
				
			}
			
			if (callNode == NULL)
			{
				for (vec_vp_i2 = matchingCalls.begin(); vec_vp_i2 != matchingCalls.end(); vec_vp_i2++)
				{
					VertexProps * vpCheck = *vec_vp_i2;
					std::cout<<vpCheck->name<<"  (tmpCheck2)  "<<bfCheck->getName()<<std::endl;
					
					// Look for function pointers resolved at runtime
					if (vpCheck->name.find("tmp") != std::string::npos)
						callNode = vpCheck;
				}
			}			
			
			if (callNode == NULL)
			{
				std::cerr<<"No matching call nodes from multiple matches"<<std::endl;
				return;
			}
			
			
			std::cout<<"Call node that matched is "<<callNode->name<<std::endl;
			handleTransferFunction_OA(callNode, blamees); 
		}
		
		std::set<VertexProps *>  oldBlamees = blamees;
		blamees.clear();
		determineBlameHolders_OA(blamees, oldBlamees, callNode,(*vec_SF_i).lineNumber, isBlamePoint, localBlamees, DQblamees);
		calcParamInfo_OA(blamees, callNode);
		
		
		// Go through and add temporary fields to the blamees from prior frames
		// Example
		// Frame #0
		// EF s1_addr.b
		// EV s1_addr
		// Frame #1
		// U  s1
		//  ... would be come
		// Frame #1
		//  U s1
		//  GF s1.b
		addTempFieldBlamees_OA(blamees, oldBlamees);
		
		clearTempFields(oldBlamees, oldFunc);
		
		
		outputFrameBlamees_OA(blamees, localBlamees, DQblamees, O);		
		//handleTransferFunction(callNode, blamedParams); 
		//blamedParams.clear();
		//determineBlameHolders_OA(blamees, (*vec_SF_i).lineNumber, isBlamePoint);
		vec_SF_i++;
		if (vec_SF_i == frames.end())
			return;
		
		if ((*vec_SF_i).lineNumber > 0)
		{
			cout<<"*****At Frame(2) "<<(*vec_SF_i).frameNumber<<" at line num "<<(*vec_SF_i).lineNumber;
			cout<<" in module "<<(*vec_SF_i).moduleName<<endl;
			
			// Get the module from the debugging information
			BlameModule * bm = modules[(*vec_SF_i).moduleName];
			
			if (bm != NULL)
			{
				// Use the combination of the module and the line number to determine the function
				BlameFunction * bf = bm->findLineRange((*vec_SF_i).lineNumber);
				
				if (bf)
				{
					if (isBlamePoint == IMPLICIT)
					{
						bf->resolveLineNum_OA(frames, modules, vec_SF_i, blamees, 
															 bf->getBlamePoint(), false, this, O);
					}
				}
			}
		}
	}
	
	// TODO:: Cases where blame point can pass up params
	// TODO:: Automatic detection of V param, V return (explicit blame points)
}



// replace for both
void BlameFunction::determineBlameHolders_OA(std::set<VertexProps *> & blamees,
																					std::set<VertexProps *> & oldBlamees,
																					VertexProps * callNode,
																					int lineNum, short isBlamePoint, std::set<VertexProps *> & localBlamees,
																					std::set<VertexProps *> & DQblamees)
{
	std::cout<<"In determineBlameHolders_OA for "<<getName()<<"  LN - "<<lineNum<<std::endl;
	
	//std::vector<VertexProps *>::iterator vec_vp_i;
	VertexHash::iterator hash_vp_i;
	
	std::set<VertexProps *> visited;
	
	bool foundOne = false;
	
	for (hash_vp_i = allVertices.begin(); hash_vp_i != allVertices.end(); hash_vp_i++)
	//for (vec_vp_i = allVertices.begin(); vec_vp_i != allVertices.end(); vec_vp_i++)
	{
		VertexProps * vp = (*hash_vp_i).second;
		//VertexProps * vp = *vec_vp_i;
		visited.clear();
		
		//std::cout<<"DBRaw VP - "<<vp->name<<" "<<vp->declaredLine<<" "<<vp->tempLine<<std::endl;
		
		if (vp->lineNumbers.count(lineNum) || vp->declaredLine == lineNum || vp->tempLine == lineNum)
		{
			//std::cout<<"DBH VP - "<<vp->name<<" "<<vp->declaredLine<<" "<<vp->tempLine<<std::endl;
			//std::cout<<"EStatus - "<<vp->eStatus<<std::endl;
			//std::cout<<"NStatus ";
			//for (int a = 0; a < NODE_PROPS_SIZE; a++)
				//std::cout<<vp->nStatus[a]<<" ";
			//std::cout<<std::endl;
			foundOne = true;
		}
		// The side effects play by a different set of rules then the others, deal with aliases
		//vp->findSEExits(blamees);
		
		if (vp->eStatus > NO_EXIT || vp->nStatus[EXIT_VAR_FIELD])
		{
			if (vp->lineNumbers.count(lineNum)  || vp->declaredLine == lineNum || vp->tempLine == lineNum) 
			{
				std::cout<<"Blamees insert(1) "<<vp->name<<std::endl;
				vp->findSEExits(blamees);
				
				// Make sure the callee EV param num matches the caller param num
				if (transferFuncApplies_OA(vp, oldBlamees, callNode) && notARepeat(vp, blamees))
				{
					blamees.insert(vp);
					vp->addedFromWhere = 1;
					addBlameToFieldParents(vp, blamees, 11);
				}
				else
				{
					vp->addedFromWhere = 61;
					DQblamees.insert(vp);
				}
			}
			else if (vp->findBlamedExits(visited, lineNum))
			{
				
				//if (getName().compare("HPL_pdtest") == 0 && excludeVP == vp)
				//continue;
				
				std::cout<<"Blamees insert(4) "<<vp->name<<std::endl;
				
				vp->findSEExits(blamees);
				
				// Make sure the callee EV param num matches the caller param num
				if (transferFuncApplies_OA(vp, oldBlamees, callNode) && notARepeat(vp, blamees))				{
					blamees.insert(vp);
					vp->addedFromWhere = 4;
					addBlameToFieldParents(vp, blamees, 14);
					
				}
				else
				{
					vp->addedFromWhere = 64;
					DQblamees.insert(vp);
				}				
			}
			
		}
		else if ( isBlamePoint )
		{
			if (vp->nStatus[EXIT_VAR_FIELD] || vp->nStatus[LOCAL_VAR] ||
					vp->nStatus[LOCAL_VAR_FIELD])
			{
				if (vp->lineNumbers.count(lineNum)) 
				{
					std::cout<<"Blamees insert(5) "<<vp->name<<std::endl;
					vp->findSEExits(blamees);
					
					
					// Make sure the callee EV param num matches the caller param num
					if (transferFuncApplies_OA(vp, oldBlamees, callNode) && notARepeat(vp, blamees))					{
						blamees.insert(vp);
						vp->addedFromWhere = 5;
						addBlameToFieldParents(vp, blamees, 15);
						
					}
					else
					{
						vp->addedFromWhere = 65;
						DQblamees.insert(vp);
					}					
				}
				else if (vp->declaredLine == lineNum)
				{
					std::cout<<"Blamees insert(6) "<<vp->name<<std::endl;
					
					vp->findSEExits(blamees);
					
					// Make sure the callee EV param num matches the caller param num
					if (transferFuncApplies_OA(vp, oldBlamees, callNode) && notARepeat(vp, blamees))					{
						blamees.insert(vp);
						vp->addedFromWhere = 6;
						addBlameToFieldParents(vp, blamees, 16);
						
					}
					else
					{
						vp->addedFromWhere = 66;
						DQblamees.insert(vp);
					}					
				}
				// This comes in handy when local variables are passed by reference,
				//  otherwise that line number would be masked by the line number of
				//  the function
				else if (vp->tempLine == lineNum)
				{
					std::cout<<"Blamees insert(7) "<<vp->name<<std::endl;
					
					vp->findSEExits(blamees);
					
					// Make sure the callee EV param num matches the caller param num
					if (transferFuncApplies_OA(vp, oldBlamees, callNode) && notARepeat(vp, blamees))					{
						blamees.insert(vp);
						vp->addedFromWhere = 7;
						addBlameToFieldParents(vp, blamees, 17);
						
					}
					else
					{
						vp->addedFromWhere = 67;
						DQblamees.insert(vp);
					}					
				}
				else if (vp->findBlamedExits(visited, lineNum))
				{
					std::cout<<"Blamees insert(8) "<<vp->name<<std::endl;
					
					vp->findSEExits(blamees);
					
					// Make sure the callee EV param num matches the caller param num
					if (transferFuncApplies_OA(vp, oldBlamees, callNode) && notARepeat(vp, blamees))					{
						blamees.insert(vp);
						vp->addedFromWhere = 8;
						addBlameToFieldParents(vp, blamees, 18);
						
					}
					else
					{
						vp->addedFromWhere = 68;
						DQblamees.insert(vp);
					}						
				}
			}		
		}
		// Straight up Local Variables that don't count towards blame
		else if (vp->nStatus[EXIT_VAR_FIELD] || vp->nStatus[LOCAL_VAR] ||
						 vp->nStatus[LOCAL_VAR_FIELD])
		{
			if (vp->lineNumbers.count(lineNum)) 
			{
							vp->findSEExits(blamees);

				std::cout<<"Blamees insert(L1) "<<vp->name<<std::endl;
				localBlamees.insert(vp);
				vp->addedFromWhere = 21;
				addBlameToFieldParents(vp, localBlamees, 31);
			}
			else if (vp->declaredLine == lineNum)
			{
							vp->findSEExits(blamees);

			
				std::cout<<"Blamees insert(L2) "<<vp->name<<std::endl;
				localBlamees.insert(vp);
				vp->addedFromWhere = 22;
				addBlameToFieldParents(vp, localBlamees, 32);
			}
			// This comes in handy when local variables are passed by reference,
			//  otherwise that line number would be masked by the line number of
			//  the function
			else if (vp->tempLine == lineNum)
			{
							vp->findSEExits(blamees);

			
				std::cout<<"Blamees insert(L3) "<<vp->name<<std::endl;
				localBlamees.insert(vp);
				vp->addedFromWhere = 23;
				addBlameToFieldParents(vp, localBlamees, 33);
			}
			else if (vp->findBlamedExits(visited, lineNum))
			{
							vp->findSEExits(blamees);

			
				std::cout<<"Blamees insert(L4) "<<vp->name<<std::endl;
				localBlamees.insert(vp);
				vp->addedFromWhere = 24;
				addBlameToFieldParents(vp, localBlamees, 34);
			}
		}				
	}	
	
	// Check for side effect relations if necessary
	// TODO: blamees.size() == 0 || one blamee and it is a retval
	
	bool proceed = false;
	
	
	if (foundOne == false)
	{
		std::cerr<<"No VPs have the line number "<<std::endl;
		std::cout<<"No VPs have the line number "<<std::endl;
	}
	
}


// might not need for either
void BlameFunction::handleTransferFunction_OA(VertexProps * callNode, std::set<VertexProps *> & blamedParams)
{
	std::cout<<"Need to do transfer function for "<<callNode->name<<std::endl;
	std::cout<<"With blamed params :";
	
	std::set<VertexProps *>::iterator vec_int_i;
	
	std::set<int> blamers; // the values feeding into the blame
	std::set<int> blamed; // the values that are blamed for call
	
	std::set<VertexProps *> blamerVP; 
	std::set<VertexProps *> blamedVP;
	
	for (vec_int_i = blamedParams.begin(); vec_int_i != blamedParams.end(); vec_int_i++)
	{
		VertexProps * vp = (*vec_int_i);
		//std::cout<<vp->name<<" ";
		
		if (vp->eStatus >= EXIT_VAR_GLOBAL)
		{
			int paramNum = vp->eStatus - EXIT_VAR_PARAM;
			if (paramNum >= 0)
			{
				blamed.insert(paramNum);
				std::cout<<paramNum<<" ";
			}	
		}
	}
	
	std::cout<<std::endl;
	
	
	std::vector<FuncCall *>::iterator vec_fc_i;
	for (vec_fc_i = callNode->calls.begin(); vec_fc_i != callNode->calls.end(); vec_fc_i++)
	{
		FuncCall *fc = (*vec_fc_i);
		if (blamed.count(fc->paramNumber) > 0)
		{
			std::cout<<"Param Num "<<fc->paramNumber<<" is blamed "<<fc->Node->name<<std::endl;
			blamedVP.insert(fc->Node);
			fc->Node->tempLine = callNode->declaredLine;
			
			// Propagate temp line up 
			std::set<VertexProps *> visited;
			fc->Node->propagateTempLineUp(visited, fc->Node->tempLine);
		}
		else
		{
			std::cout<<"Param Num "<<fc->paramNumber<<" is blamer"<<std::endl;
			blamerVP.insert(fc->Node);
		}
	}
	
	std::set<VertexProps *>::iterator set_vp_i;
	std::set<VertexProps *>::iterator set_vp_i2;
	
	for (set_vp_i = blamerVP.begin(); set_vp_i != blamerVP.end(); set_vp_i++)
	{
		VertexProps * bE = (*set_vp_i);
		for (set_vp_i2 = blamedVP.begin(); set_vp_i2 != blamedVP.end(); set_vp_i2++)
		{
			VertexProps * bD = *set_vp_i2;
			bD->tempChildren.insert(bE);
			bE->tempParents.insert(bD);
			bD->tempIsWritten = true;
			std::cout<<bE->name<<" is child (TF) to "<<bD->name<<std::endl;
		}
	}
}


// will need separate one for read (maybe write)
void BlameFunction::outputFrameBlamees_OA(std::set<VertexProps *> & blamees, std::set<VertexProps *> & localBlamees, 
																			 std::set<VertexProps *> & DQblamees, std::ostream &O)
{
	std::set<VertexProps *>::iterator set_vp_i;
	
	for (set_vp_i = blamees.begin(); set_vp_i != blamees.end(); set_vp_i++)
	{
		VertexProps * vp = (*set_vp_i);
		
		if (vp->isDerived) 
		{
			VertexProps * rootField = vp;
			std::set<VertexProps *> visited;
			
			while (rootField->fieldUpPtr != NULL && visited.count(rootField->fieldUpPtr) == 0)
			{
				visited.insert(rootField->fieldUpPtr);
				rootField = rootField->fieldUpPtr;
			}
			
			if  (rootField->eStatus >= EXIT_VAR_GLOBAL)
			{
				O<<"EDF";
				std::cout<<"EDF ";
			}
			else
			{
				O<<"LDF";
				std::cout<<"LDF ";
			}
		}
		else if (vp->eStatus == EXIT_PROG)
		{
			O<<"EP";
			std::cout<<"EP ";
		}
		else if (vp->eStatus == EXIT_OUTP)
		{
			O<<"EO";
			std::cout<<"EO ";
			
		}
		else if (vp->eStatus >= EXIT_VAR_GLOBAL)
		{
			O<<"EV";
			std::cout<<"EV ";
		}
		else if (vp->nStatus[EXIT_VAR_FIELD])
		{
			O<<"EF";
			std::cout<<"EF ";
		}
		else if (vp->nStatus[LOCAL_VAR_FIELD])
		{
			O<<"LF";
			std::cout<<"LF ";
			
		}
		else 
		{
			O<<"U";
			std::cout<<"U ";
		}
		
		O<<vp->addedFromWhere<<" ";
		
		if (vp->nStatus[EXIT_VAR_FIELD] || vp->nStatus[LOCAL_VAR_FIELD] || vp->isDerived)
		{	
			O<<getFullStructName(vp)<<"   ";
			//O<<vp->name;
		}
		else
		{
			O<<vp->name<<"   ";
			//O<<vp->name;
		}
		
		// The generic type as given by LLVM (int, double, Struct*)
		O<<"   "<<vp->genType<<" ";
		
		if (vp->sType != NULL)
		{
			O<<vp->sType->structName<<"   ";
		}
		else
		{
			O<<"NULL   ";
		}
		
		/*
		 if (vp->fieldUpPtr != NULL)
		 {
		 O<<getFullStructName(vp->fieldUpPtr)<<" ";
		 }
		 else
		 O<<"NULL ";
		 */
		
		outputParamInfo(O,vp);
		outputParamInfo(std::cout,vp);
		
		
		O<<std::endl;
		std::cout<<std::endl;
		
	}
	
	
	for (set_vp_i = localBlamees.begin(); set_vp_i != localBlamees.end(); set_vp_i++)
	{
		VertexProps * vp = (*set_vp_i);
		
		if (vp->isDerived) 
		{
			O<<"VDF";
			std::cout<<"VDF ";
		}
		if (vp->nStatus[LOCAL_VAR])
		{
			O<<"VL";
			std::cout<<"VL ";
		}
		else if (vp->nStatus[LOCAL_VAR_FIELD])
		{
			O<<"VFL";
			std::cout<<"VFL ";
		}
		else
		{
			O<<"VL";
			std::cout<<"VL ";
		}
		
		O<<vp->addedFromWhere<<" ";
		
		if ( vp->nStatus[LOCAL_VAR_FIELD] )
		{	
			O<<getFullStructName(vp)<<" ";
			//O<<vp->name;
		}
		else
		{
			O<<vp->name<<" ";
			//O<<vp->name;
		}
		
		// The generic type as given by LLVM (int, double, Struct*)
		O<<vp->genType<<" ";
		
		if (vp->sType != NULL)
		{
			O<<vp->sType->structName<<" ";
		}
		else
		{
			O<<"NULL ";
		}
		
		O<<std::endl;
		std::cout<<std::endl;
		
	}
	
	if (blamees.size() == 0)
	{
		std::cerr<<"No EV[EO, EP] found ["<<realName<<"]"<<std::endl;
		std::cout<<"No EV[EO, EP] found ["<<realName<<"]"<<std::endl;
		O<<"***No EV[EO, EP] found*** ["<<realName<<"]"<<std::endl;
		
	}
	
}


// probably won't need for write, maybe for read
void BlameFunction::addTempFieldBlameesRecursive_OA(VertexProps * vp, VertexProps * blamee, std::set<VertexProps *> & oldBlamees,
																								 std::set<VertexProps *> & blamees, std::set<VertexProps *> & visited)
{
	// The usual recursive check
	
	////std::cout<<"ATFB(1) "<<vp->name<<std::endl;
	if (visited.count(vp))
		return;
	
	visited.insert(vp);
	
	std::cout<<"ATFB(2) "<<vp->name<<std::endl;
	
	
	std::set<VertexProps *>::iterator set_vp_i;
	
	for (set_vp_i = vp->fields.begin(); set_vp_i != vp->fields.end(); set_vp_i++)
	{
		std::cout<<"ATFB(3) "<<vp->name<<" "<<(*set_vp_i)->name<<std::endl;
		
		if (oldBlamees.count(*set_vp_i) > 0)
		{
			VertexProps * vp2 = *set_vp_i;
			std::cout<<getFullStructName(blamee)<<"("<<blamee<<") maybe connected to(2) "<<getFullStructName(vp2)<<" "<<vp2->name<<" "<<vp2<<std::endl;
			
			bool found = true;
			VertexProps * newVP = findOrCreateTempBlamees(blamees, getFullStructName(vp2), found);
			
			if (found == false)
			{
				std::cout<<"Adding info for newly generated VP(2) "<<newVP->name<<std::endl;
				newVP->sType = vp2->sType;
				newVP->genType = vp2->genType;
				newVP->fieldUpPtr = blamee;
				newVP->sField = vp2->sField;
				newVP->calleePar = blamee->calleePar;
				blamee->tempFields.insert(newVP);
			}
			
			addTempFieldBlameesRecursive_OA(vp2, newVP, oldBlamees, blamees, visited);
		}
	}
	
	for (set_vp_i = vp->tempFields.begin(); set_vp_i != vp->tempFields.end(); set_vp_i++)
	{
		std::cout<<"ATFB(4) "<<vp->name<<" "<<(*set_vp_i)->name<<std::endl;
		
		if (oldBlamees.count(*set_vp_i) > 0)
		{
			VertexProps * vp2 = *set_vp_i;
			std::cout<<getFullStructName(blamee)<<"("<<blamee<<")"<<" maybe connected to(3) "<<getFullStructName(vp2)<<" "<<vp2->name<<" "<<vp2<<std::endl;
			
			bool found = true;
			VertexProps * newVP = findOrCreateTempBlamees(blamees, getFullStructName(vp2), found);
			
			if (found == false)
			{
				std::cout<<"Adding info for newly generated VP(3) "<<newVP->name<<" "<<newVP<<std::endl;
				newVP->sType = vp2->sType;
				newVP->genType = vp2->genType;
				newVP->fieldUpPtr = blamee;
				newVP->sField = vp2->sField;
				newVP->calleePar = blamee->calleePar;
				blamee->tempFields.insert(newVP);
			}
			
			addTempFieldBlameesRecursive_OA(vp2, newVP, oldBlamees, blamees, visited);
		}
	}
	
	
	
	for (set_vp_i = vp->temptempFields.begin(); set_vp_i != vp->temptempFields.end(); set_vp_i++)
	{
		std::cout<<"ATFB(5) "<<vp->name<<" "<<(*set_vp_i)->name<<std::endl;
		
		if (oldBlamees.count(*set_vp_i) > 0)
		{
			VertexProps * vp2 = *set_vp_i;
			std::cout<<getFullStructName(blamee)<<"("<<blamee<<")"<<" maybe connected to(3) "<<getFullStructName(vp2)<<" "<<vp2->name<<" "<<vp2<<std::endl;
			
			bool found = true;
			VertexProps * newVP = findOrCreateTempBlamees(blamees, getFullStructName(vp2), found);
			
			if (found == false)
			{
				std::cout<<"Adding info for newly generated VP(3) "<<newVP->name<<" "<<newVP<<std::endl;
				newVP->sType = vp2->sType;
				newVP->genType = vp2->genType;
				newVP->fieldUpPtr = blamee;
				newVP->sField = vp2->sField;
				newVP->calleePar = blamee->calleePar;
				blamee->tempFields.insert(newVP);
			}
			
			addTempFieldBlameesRecursive_OA(vp2, newVP, oldBlamees, blamees, visited);
		}
	}

	
	
}

// probably won't need for write, maybe for read
void BlameFunction::addTempFieldBlamees_OA(std::set<VertexProps *> & blamees, std::set<VertexProps *> & oldBlamees)
{
	std::set<VertexProps *>::iterator set_vp_i, set_vp_i2;
	
	for (set_vp_i = blamees.begin(); set_vp_i != blamees.end(); set_vp_i++)
	{
		VertexProps * vp = (*set_vp_i);
		// This blamee is a struct and has some fields we can attach to it from prior frames
		if (vp->genType.find("Struct") != std::string::npos)
		{
			for (set_vp_i2 = oldBlamees.begin(); set_vp_i2 != oldBlamees.end(); set_vp_i2++)
			{
				VertexProps * vp2 = (*set_vp_i2);
				std::cout<<"Comparing "<<vp->name<<" to "<<vp2->name<<std::endl;
				
				std::cout<<vp->callerPars.count(vp2->calleePar)<<"  "<<vp2->fieldUpPtr<<std::endl;
				// We're looking at the root in the previous frame that matches up
				if (vp->callerPars.count(vp2->calleePar)  && vp2->fieldUpPtr == NULL)
				{
					std::cout<<vp->sType<<" "<<vp2->sType<<std::endl;
					if (vp->sType == NULL || vp2->sType == NULL)
						continue;
					
					std::cout<<vp->sType->structName<<" "<<vp2->sType->structName<<std::endl;
					if (vp->sType->structName != vp2->sType->structName)
						continue;
					
					std::cout<<vp->name<<" maybe connected to "<<vp2->name<<std::endl;
					std::cout<<getFullStructName(vp)<<" maybe connected to "<<getFullStructName(vp2)<<" "<<vp2->name<<std::endl;
					
					std::set<VertexProps *> visited;
					addTempFieldBlameesRecursive_OA(vp2, vp, oldBlamees, blamees, visited);
					
					//if (vp2->fieldUpPtr != NULL)
					//std::cout<<" "<<vp2->fieldUpPtr->name<<std::endl;
					//else
					//std::cout<<std::endl;
				}
			}
		}
		else if (vp->genType.find("VOID") != std::string::npos)
		{
			
			for (set_vp_i2 = oldBlamees.begin(); set_vp_i2 != oldBlamees.end(); set_vp_i2++)
			{
				VertexProps * vp2 = (*set_vp_i2);
				std::cout<<"Comparing(3) "<<vp->name<<" to "<<vp2->name<<std::endl;
				
				std::cout<<vp->callerPars.count(vp2->calleePar)<<"  "<<vp2->fieldUpPtr<<std::endl;
				
				// We're looking at the root in the previous frame that matches up
				if (vp->callerPars.count(vp2->calleePar)  && vp2->fieldUpPtr == NULL)
				{
					std::cout<<"Comparing(3a) "<<vp->name<<" to "<<vp2->name<<std::endl;
					std::cout<<vp->sType<<" "<<vp2->sType<<std::endl;
					
					//if (vp->sType == NULL || vp2->sType == NULL)
					//	continue;
					
					//std::cout<<vp->sType->structName<<" "<<vp2->sType->structName<<std::endl;
					//if (vp->sType->structName != vp2->sType->structName)
					//	continue;
					
					std::cout<<vp->name<<" maybe(3) connected to "<<vp2->name<<std::endl;
					std::cout<<getFullStructName(vp)<<" maybe(3) connected to "<<getFullStructName(vp2)<<" "<<vp2->name<<std::endl;
					
					std::set<VertexProps *> visited;
					addTempFieldBlameesRecursive_OA(vp2, vp, oldBlamees, blamees, visited);
					
				}
			}		
			
		}
		else if (vp->genType.find("Opaque") != std::string::npos)
		{
			for (set_vp_i2 = oldBlamees.begin(); set_vp_i2 != oldBlamees.end(); set_vp_i2++)
			{
				VertexProps * vp2 = (*set_vp_i2);
				std::cout<<"Comparing(4) "<<vp->name<<" to "<<vp2->name<<std::endl;
				
				std::cout<<vp->callerPars.count(vp2->calleePar)<<"  "<<vp2->fieldUpPtr<<std::endl;
				
				// We're looking at the root in the previous frame that matches up
				if (vp->callerPars.count(vp2->calleePar)  && vp2->fieldUpPtr == NULL)
				{
					std::cout<<"Comparing(4a) "<<vp->name<<" to "<<vp2->name<<std::endl;
					std::cout<<vp->sType<<" "<<vp2->sType<<std::endl;
					
					//if (vp->sType == NULL || vp2->sType == NULL)
					//	continue;
					
					//std::cout<<vp->sType->structName<<" "<<vp2->sType->structName<<std::endl;
					//if (vp->sType->structName != vp2->sType->structName)
					//	continue;
					
					std::cout<<vp->name<<" maybe(4) connected to "<<vp2->name<<std::endl;
					std::cout<<getFullStructName(vp)<<" maybe(4) connected to "<<getFullStructName(vp2)<<" "<<vp2->name<<std::endl;
					
					std::set<VertexProps *> visited;
					addTempFieldBlameesRecursive_OA(vp2, vp, oldBlamees, blamees, visited);
					
				}
			}		
		}
		else if (vp->genType.find("Int") != std::string::npos)
		{
			for (set_vp_i2 = oldBlamees.begin(); set_vp_i2 != oldBlamees.end(); set_vp_i2++)
			{
				VertexProps * vp2 = (*set_vp_i2);
				std::cout<<"Comparing(5) "<<vp->name<<" to "<<vp2->name<<std::endl;
				
				std::cout<<vp->callerPars.count(vp2->calleePar)<<"  "<<vp2->fieldUpPtr<<std::endl;
				
				// We're looking at the root in the previous frame that matches up
				if (vp->callerPars.count(vp2->calleePar)  && vp2->fieldUpPtr == NULL)
				{
					std::cout<<"Comparing(5a) "<<vp->name<<" to "<<vp2->name<<std::endl;
					std::cout<<vp->sType<<" "<<vp2->sType<<std::endl;
					
					//if (vp->sType == NULL || vp2->sType == NULL)
					//	continue;
					
					//std::cout<<vp->sType->structName<<" "<<vp2->sType->structName<<std::endl;
					//if (vp->sType->structName != vp2->sType->structName)
					//	continue;
					
					std::cout<<vp->name<<" maybe(5) connected to "<<vp2->name<<std::endl;
					std::cout<<getFullStructName(vp)<<" maybe(5) connected to "<<getFullStructName(vp2)<<" "<<vp2->name<<std::endl;
					
					std::set<VertexProps *> visited;
					addTempFieldBlameesRecursive_OA(vp2, vp, oldBlamees, blamees, visited);
					
				}
			}		
		}
		else
		{
			for (set_vp_i2 = oldBlamees.begin(); set_vp_i2 != oldBlamees.end(); set_vp_i2++)
			{
				VertexProps * vp2 = (*set_vp_i2);
				std::cout<<"Comparing(2) "<<vp->name<<" to "<<vp2->name<<std::endl;
				
				std::cout<<vp->callerPars.count(vp2->calleePar)<<"  "<<vp2->fieldUpPtr<<std::endl;
				// We're looking at the root in the previous frame that matches up
				if (vp->callerPars.count(vp2->calleePar)  && vp2->fieldUpPtr == NULL)
				{
					std::cout<<"Comparing(2a) "<<vp->name<<" to "<<vp2->name<<std::endl;
					std::cout<<vp->sType<<" "<<vp2->sType<<std::endl;
					
					if (vp2->sType == NULL)
						continue;
					//if (vp->sType == NULL || vp2->sType == NULL)
					//continue;
					
					//std::cout<<vp->sType->structName<<" "<<vp2->sType->structName<<std::endl;
					std::cout<<vp2->sType->structName<<std::endl;
					
					//if (vp->sType->structName != vp2->sType->structName)
					//continue;
					
					std::cout<<vp->name<<" maybe(2) connected to "<<vp2->name<<std::endl;
					std::cout<<getFullStructName(vp)<<" maybe(2) connected to "<<getFullStructName(vp2)<<" "<<vp2->name<<std::endl;
					
					std::set<VertexProps *> visited;
					addTempFieldBlameesRecursive_OA(vp2, vp, oldBlamees, blamees, visited);
				}
			}
		}
	}
}




// might not need for either
void BlameFunction::calcParamInfo_OA(std::set<VertexProps *> & blamees, VertexProps * callNode)
{
	std::set<VertexProps *>::iterator set_vp_i;
	
	for (set_vp_i = blamees.begin(); set_vp_i != blamees.end(); set_vp_i++)
	{
		VertexProps * vp = *set_vp_i;
		if (vp->isDerived)
			continue;
		
		if (vp->calleePar < 0)
		{
			vp->calleePar = 99;
			
			if (vp->eStatus >= EXIT_VAR_RETURN)
			{
				vp->calleePar = vp->eStatus - EXIT_VAR_RETURN;
			}
			
			if (vp->nStatus[EXIT_VAR_FIELD])
			{
				VertexProps * upPtr = vp->fieldUpPtr;
				std::set<VertexProps *> visited;
				
				while (upPtr != NULL && visited.count(upPtr) == 0)
				{
					visited.insert(upPtr);
					if (upPtr->eStatus >= EXIT_VAR_RETURN)
					{
						vp->calleePar = upPtr->eStatus - EXIT_VAR_RETURN;
						break;
					}
					upPtr = upPtr->fieldUpPtr;
				}
			}
		}
		
		
		if (callNode != NULL)
		{
			std::vector<FuncCall *>::iterator vec_fc_i;
			for (vec_fc_i = callNode->calls.begin(); vec_fc_i != callNode->calls.end(); vec_fc_i++)
			{
				FuncCall *fc = (*vec_fc_i);
				if (vp->params.count(fc->Node))
				{
					vp->callerPars.insert(fc->paramNumber);
				}
			}
		}		
	}
}


// might not need for either
bool BlameFunction::transferFuncApplies_OA(VertexProps * caller, std::set<VertexProps *> & oldBlamees,
																				VertexProps * callNode)
{
	if (callNode == NULL)
		return true;
	
	
	std::vector<FuncCall *>::iterator vec_fc_i;
	for (vec_fc_i = callNode->calls.begin(); vec_fc_i != callNode->calls.end(); vec_fc_i++)
	{
		FuncCall *fc = (*vec_fc_i);
		if (caller->params.count(fc->Node))
		{
			std::cout<<"In TFA, examining "<<caller->name<<" "<<fc->paramNumber<<" "<<std::endl;
			std::set<VertexProps *>::iterator set_vp_i;
			for (set_vp_i = oldBlamees.begin(); set_vp_i != oldBlamees.end(); set_vp_i++)
			{
				VertexProps * vp = *set_vp_i;
				
				if (vp->isDerived)
					continue;
				
				int calleePar = 99;
				
				if (vp->eStatus >= EXIT_VAR_RETURN)
				{
					calleePar = vp->eStatus - EXIT_VAR_RETURN;
				}
				
				if (vp->nStatus[EXIT_VAR_FIELD])
				{
					VertexProps * upPtr = vp->fieldUpPtr;
					std::set<VertexProps *> visited;
					while (upPtr != NULL && visited.count(upPtr) == 0)
					{
						visited.insert(upPtr);
						if (upPtr->eStatus >= EXIT_VAR_RETURN)
						{
							calleePar = upPtr->eStatus - EXIT_VAR_RETURN;
							break;
						}
						upPtr = upPtr->fieldUpPtr;
					}
				}
				
				if (fc->paramNumber == calleePar)
				{	
					std::cout<<"In TFA, match found for "<<vp->name<<" and "<<caller->name<<" "<<calleePar<<std::endl;
					//std::cout<<calleePar<<" ";
					return true;
				}
			}
		}
	}
	
	std::cout<<"In TFA, no match found for "<<caller->name<<std::endl;
	return false;
}






