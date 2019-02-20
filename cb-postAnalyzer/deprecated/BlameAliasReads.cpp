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


void clearTempWeightData(set<VertexProps *> & blamees, set<VertexProps *> & localBlamees)
{
	set<VertexProps *>::iterator set_vp_i;
	
	for (set_vp_i = blamees.begin(); set_vp_i != blamees.end(); set_vp_i++)
	{
		VertexProps * vp = *set_vp_i;
		vp->weight = 1.0;
	}
	
	for (set_vp_i = localBlamees.begin(); set_vp_i != localBlamees.end(); set_vp_i++)
	{
		VertexProps * vp = *set_vp_i;
		vp->weight = 1.0;
	}

}

// replace for both
void BlameFunction::resolveLineNum_OAR(vector<StackFrame> & frames, ModuleHash & modules,
																	 vector<StackFrame>::iterator vec_SF_i, std::set<VertexProps *> & blamees,
																	 short isBlamePoint, bool isBottomParsed, BlameFunction * oldFunc, std::ostream &O)
{
	
	//std::set<VertexProps *> blamees;
	
	clearPastData();
	
	std::set<VertexProps *> localBlamees;
	std::set<VertexProps *> DQblamees;
	
	O<<"FRAME(OAR)# "<<(*vec_SF_i).frameNumber<<" ";
	O<<getFullContextName(frames, modules, vec_SF_i);
	//O<<getName()<<" "<<getModuleName()<<" "<<getModulePathName()<<" ";
	O<<" "<<getModuleName()<<" "<<getModulePathName()<<" ";
	O<<(*vec_SF_i).lineNumber<<" "<<isBlamePoint<<" ";
	O<<beginLineNum<<" "<<endLineNum<<std::endl;
	
	std::cout<<getName()<<" LN(OAR) -  "<<(*vec_SF_i).lineNumber<<" BP - ";
	std::cout<<isBlamePoint<<" Frame(OAR) # "<<(*vec_SF_i).frameNumber<<std::endl;
	
	
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
			std::cerr<<"TRANSFER FUNC BAR.cpp:98 - No lineNums found for sample in "<<getName();
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
			determineBlameHolders_OAR(blamees, oldBlamees, callNode, (*vec_SF_i).lineNumber, isBlamePoint, localBlamees, DQblamees);
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
			
			outputFrameBlamees_OAR(blamees, localBlamees, DQblamees, O);		
			clearTempWeightData(blamees, localBlamees);
					
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
						bf->resolveLineNum_OAR(frames, modules, vec_SF_i, blamees, 
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
			determineBlameHolders_OAR(blamees, oldBlamees, callNode, (*vec_SF_i).lineNumber, isBlamePoint, localBlamees, DQblamees);
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
			
			
			outputFrameBlamees_OAR(blamees, localBlamees, DQblamees, O);	
						clearTempWeightData(blamees, localBlamees);
		
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
						bf->resolveLineNum_OAR(frames, modules, vec_SF_i, blamees, 
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
		determineBlameHolders_OAR(blamees, oldBlamees, callNode,(*vec_SF_i).lineNumber, isBlamePoint, localBlamees, DQblamees);
		calcParamInfo_OA(blamees, callNode);
		
		
		outputFrameBlamees_OAR(blamees, localBlamees, DQblamees, O);		
					clearTempWeightData(blamees, localBlamees);

		
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
					bf->resolveLineNum_OAR(frames, modules, vec_SF_i, blamees, 
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
			std::cerr<<"TRANSFER FUNC BAR.cpp:348 - No lineNums found for sample in "<<getName();
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
		determineBlameHolders_OAR(blamees, oldBlamees, callNode,(*vec_SF_i).lineNumber, isBlamePoint, localBlamees, DQblamees);
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
		
		
		outputFrameBlamees_OAR(blamees, localBlamees, DQblamees, O);		
					clearTempWeightData(blamees, localBlamees);

		
		//handleTransferFunction(callNode, blamedParams); 
		//blamedParams.clear();
		//determineBlameHolders_OAR(blamees, (*vec_SF_i).lineNumber, isBlamePoint);
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
						bf->resolveLineNum_OAR(frames, modules, vec_SF_i, blamees, 
															 bf->getBlamePoint(), false, this, O);
					}
				}
			}
		}
	}
	
	// TODO:: Cases where blame point can pass up params
	// TODO:: Automatic detection of V param, V return (explicit blame points)
}


float BlameFunction::determineWeight(VertexProps * vp, int lineNum)
{

	if (vp->nStatus[LOCAL_VAR] == 0)
		return 1.0;
		
	cout<<"Looking at weights for VP "<<vp->name<<" and line number "<<lineNum<<endl;

	// Look for which loops apply	
	vector<LoopInfo *>::iterator vec_li_i;
	
	float maxLoop = 0.0;
	
	for (vec_li_i = loops.begin(); vec_li_i != loops.end(); vec_li_i++)
	{
		LoopInfo * liVar = *vec_li_i;
		if (liVar->lineNumbers.count(lineNum))
		{
			cout<<"Line Found in Loop of size "<<liVar->lineNumbers.size()<<endl;
			
			set<int>::iterator set_i_var_i;
			set<int>::iterator set_i_loop_i;
			
			int loopSize = liVar->lineNumbers.size();
			int matchSize = 0;
			for (set_i_loop_i = liVar->lineNumbers.begin(); set_i_loop_i != liVar->lineNumbers.end(); set_i_loop_i++)
			{
				matchSize += vp->lineReadNumbers.count(*set_i_loop_i);
			}
				
			float percMatch = (float) matchSize / (float) loopSize;
			cout<<"Percentage Match for this loop is "<<percMatch<<endl;

			if (maxLoop < percMatch)
				maxLoop = percMatch;

		}
	}
	
	//return maxLoop;
	// RUTAR lets see what this does
	return maxLoop*maxLoop;
	
}


void adjustWeightsBasedOnReadCount(std::set<VertexProps *> & blamees, int lineNum)
{
	int totalReads = 0;
	double numVariables = (double) blamees.size();
	double expWeight = 1.0/numVariables;

	//double normalizedWeight = 0.0;

	set<VertexProps *>	::iterator set_vp_i;
	for (set_vp_i = blamees.begin(); set_vp_i != blamees.end(); set_vp_i++)
	{
		VertexProps * vp = *set_vp_i;
		cout<<"Number of reads for variable "<<vp->name<<" at line number "<<lineNum<<" is "<<vp->readLines[lineNum]<<endl;
		totalReads += vp->readLines[lineNum];
	}
	
	for (set_vp_i = blamees.begin(); set_vp_i != blamees.end(); set_vp_i++)
	{
		VertexProps * vp = *set_vp_i;
		
		double numReads = (double) vp->readLines[lineNum];
		double actWeight = numReads/(double) totalReads;
		double normWeight = actWeight/expWeight;
		
		cout<<"Norm weight for variable "<<vp->name<<" at line number "<<lineNum<<" is "<<normWeight<<endl;
		
		vp->weight *= normWeight;
		
		//totalReads += vp->readLines[lineNum];
	}
	
	
	
	cout<<"Number of reads for line "<<lineNum<<" is "<<totalReads<<endl;
}


// replace for both
void BlameFunction::determineBlameHolders_OAR(std::set<VertexProps *> & blamees,
																					std::set<VertexProps *> & oldBlamees,
																					VertexProps * callNode,
																					int lineNum, short isBlamePoint, std::set<VertexProps *> & localBlamees,
																					std::set<VertexProps *> & DQblamees)
{
	std::cout<<"In determineBlameHolders_OAR for "<<getName()<<"  LN - "<<lineNum<<std::endl;
	
	//std::vector<VertexProps *>::iterator vec_vp_i;
	VertexHash::iterator hash_vp_i;
	
	std::set<VertexProps *> visited;
	
	bool foundOne = false;
	
	for (hash_vp_i = allVertices.begin(); hash_vp_i != allVertices.end(); hash_vp_i++)
	//for (vec_vp_i = allVertices.begin(); vec_vp_i != allVertices.end(); vec_vp_i++)
	{
		//VertexProps * vp = *vec_vp_i;
		VertexProps * vp = (*hash_vp_i).second;
		
		visited.clear();
		
		//std::cout<<"DBRaw VP - "<<vp->name<<" "<<vp->declaredLine<<" "<<vp->tempLine<<std::endl;
		
		if (vp->lineReadNumbers.count(lineNum) )
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
			if (vp->lineReadNumbers.count(lineNum) ) 
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
			else if (vp->findBlamedExits_OAR(visited, lineNum))
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
				if (vp->lineReadNumbers.count(lineNum)) 
				{
					std::cout<<"Blamees insert(5) "<<vp->name<<std::endl;
					vp->findSEExits(blamees);
					
					
					// Make sure the callee EV param num matches the caller param num
					if (transferFuncApplies_OA(vp, oldBlamees, callNode) && notARepeat(vp, blamees))					{
						blamees.insert(vp);
						vp->weight = 1.0 - determineWeight(vp, lineNum);
						
						vp->addedFromWhere = 5;
						addBlameToFieldParents(vp, blamees, 15);
						
					}
					else
					{
						vp->addedFromWhere = 65;
						DQblamees.insert(vp);
					}					
				}
				else if (vp->findBlamedExits_OAR(visited, lineNum))
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
			if (vp->lineReadNumbers.count(lineNum)) 
			{
				vp->findSEExits(blamees);

				std::cout<<"Blamees insert(L1) "<<vp->name<<std::endl;
				localBlamees.insert(vp);
				vp->weight = 1.0 - determineWeight(vp, lineNum);
				vp->addedFromWhere = 21;
				addBlameToFieldParents(vp, localBlamees, 31);
			}
			else if (vp->findBlamedExits_OAR(visited, lineNum))
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
	
	
	//  Now we adjust weights based on the number of reads for that particular blamee on that line number
	//  NEW and maybe optional functionality
	
	//adjustWeightsBasedOnReadCount(localBlamees, lineNum);
	//adjustWeightsBasedOnReadCount(blamees, lineNum);

			
									
					
	if (foundOne == false)
	{
		std::cerr<<"No VPs have the line number "<<std::endl;
		std::cout<<"No VPs have the line number "<<std::endl;
	}
	
}




int VertexProps::findBlamedExits_OAR(std::set<VertexProps *> & visited, int lineNum)
{
	int total = 0;
	
	if (visited.count(this) > 0)
		return 0;

	visited.insert(this);
	
	// We don't want to end up grabbing our struct parent (due to a param thing)
	//  and using their line number
	if (this->fieldUpPtr != NULL)
		visited.insert(this->fieldUpPtr);
	
	if (lineReadNumbers.count(lineNum))
		return 1;

	std::set<VertexProps *>::iterator set_vp_i;
	std::vector<FuncCall *>::iterator vec_vp_i;
	
	// This is the key step for EVs ... the params are data pointers
	//   that represent the EVs when passed into function ... unfortunately
	//   if we don't represent the params correctly then we lose the link
	//   to the EVs which is a bad thing
	for (set_vp_i = params.begin(); set_vp_i != params.end(); set_vp_i++)
	{
			VertexProps * vp = (*set_vp_i);
			std::cout<<"Param VP "<<vp->name<<" examined for parent "<<name<<std::endl;
			std::cout<<"Total before - "<<total<<std::endl;
			std::cout<<"Temp Parents size - "<<vp->tempParents.size()<<std::endl;
			if (vp->tempParents.size() == 0)
				total += vp->findBlamedExits_OAR(visited, lineNum);
			std::cout<<"Total after - "<<total<<std::endl;
	}		
	
	/*
	for (vec_vp_i = calls.begin(); vec_vp_i != calls.end(); vec_vp_i++)
	{
			FuncCall * fc = *vec_vp_i;
	}*/
	
	for (set_vp_i = fields.begin(); set_vp_i != fields.end(); set_vp_i++)
	{
		VertexProps * vp = (*set_vp_i);
		if (vp->tempParents.size() == 0)
			total += vp->findBlamedExits_OAR(visited, lineNum);
	}
	
	for (set_vp_i = tempChildren.begin(); set_vp_i != tempChildren.end(); set_vp_i++)
	{
			VertexProps * vp = (*set_vp_i);
			total += vp->findBlamedExits_OAR(visited, lineNum);
	}		
	
	return total;
	
}


		


// will need separate one for read (maybe write)
void BlameFunction::outputFrameBlamees_OAR(std::set<VertexProps *> & blamees, std::set<VertexProps *> & localBlamees, 
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
		else if (vp->eStatus == EXIT_VAR_GLOBAL)
		{
			O<<"EGV";
			std::cout<<"EGV";
		}
		else if (vp->eStatus > EXIT_VAR_GLOBAL)
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
		
		O<<" "<<vp->weight<<" ";
		
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
		
		O<<" "<<vp->weight;
		
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

