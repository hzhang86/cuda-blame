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

#include "ExitVars.h"
#include "FunctionBFC.h"

#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

void FunctionBFC::exportSideEffects(ostream &O)
{
	O<<"BEGIN FUNC "<<getSourceFuncName()<<" ";
	O<<getModuleName()<<" "<<getStartLineNum()<<" "<<getEndLineNum()<<" ";
	O<<getModulePathName()<<endl;

	vector< pair<NodeProps *, NodeProps *> >::iterator vec_pair_i;

	if (seRelations.size() > 0)
	{
		O<<"BEGIN SE_RELATIONS"<<endl;
		for (vec_pair_i = seRelations.begin(); vec_pair_i != seRelations.end(); vec_pair_i++)
		{
			NodeProps * fir = (*vec_pair_i).first;
			NodeProps * sec = (*vec_pair_i).second;
				
			set<NodeProps *> visited;	
			int firNum = fir->getParamNum(visited);
			
			visited.clear();
			int secNum = sec->getParamNum(visited);
		
			O<<"R: "<<fir->getFullName()<<"  "<<firNum<<" ";
			O<<sec->getFullName()<<" "<<secNum<<endl;

		}
		O<<"END SE_RELATIONS"<<endl;
	}
	
	
	if (seAliases.size() > 0)
	{
		O<<"BEGIN SE_ALIASES"<<endl;
		for (vec_pair_i = seAliases.begin(); vec_pair_i != seAliases.end(); vec_pair_i++)
		{
			NodeProps * fir = (*vec_pair_i).first;
			NodeProps * sec = (*vec_pair_i).second;
	
			string firName, secName;
			
			set<NodeProps *> visited;	
			int firNum = fir->getParamNum(visited);
			
			visited.clear();
			int secNum = sec->getParamNum(visited);
		
			O<<"A: "<<fir->getFullName()<<"  "<<firNum<<" ";
			O<<sec->getFullName()<<" "<<secNum<<endl;
		}
		O<<"END SE_ALIASES"<<endl;
	}
	
	if (seCalls.size() > 0)
	{
		O<<"BEGIN SE_CALLS"<<endl;
		vector< FuncCallSE * >::iterator vec_fcse_i;
		for (vec_fcse_i = seCalls.begin(); vec_fcse_i != seCalls.end(); vec_fcse_i++)
		{
			FuncCallSE * fcse = *vec_fcse_i;
			O<<"C: "<<fcse->callNode->name<<endl;
		
			vector<FuncCallSEElement *>::iterator vec_fcseEl_i;
			for (vec_fcseEl_i = fcse->parameters.begin(); vec_fcseEl_i != fcse->parameters.end();
					vec_fcseEl_i++)
			{
				set<NodeProps *> visited;
				FuncCallSEElement * fcseElement = *vec_fcseEl_i;
				O<<"P: "<<fcseElement->ifc->paramNumber<<" ";
				O<<fcseElement->paramEVNode->name<<" ";
				O<<fcseElement->paramEVNode->getParamNum(visited)<<endl;
			}
			
			for (vec_fcseEl_i = fcse->NEVparams.begin(); vec_fcseEl_i != fcse->NEVparams.end();
					vec_fcseEl_i++)
			{
							set<NodeProps *> visited;

				FuncCallSEElement * fcseElement = *vec_fcseEl_i;
				O<<"P: "<<fcseElement->ifc->paramNumber<<" ";
				O<<fcseElement->paramEVNode->name<<" ";
				O<<fcseElement->paramEVNode->getParamNum(visited)<<endl;
			}

			
		}
		O<<"END SE_CALLS"<<endl;
	}
	
	O<<"END FUNC "<<endl;
}



void FunctionBFC::exportEverything(ostream &O, bool reads)
{
	O<<"BEGIN FUNC "<<endl;
	
	O<<"BEGIN F_NAME"<<endl;
	O<<getSourceFuncName()<<endl;
	O<<"END F_NAME"<<endl;
	
	O<<"BEGIN F_REAL_NAME"<<endl;
	O<<getRealName()<<endl;
	O<<"END F_REAL_NAME"<<endl;
	
	O<<"BEGIN FUNC_TYPE"<<endl;
	O<<ft<<endl;
	O<<"END FUNC_TYPE"<<endl;
	
	O<<"BEGIN M_NAME_PATH "<<endl;
	O<<getModulePathName()<<endl;
	O<<"END M_NAME_PATH "<<endl;
	
	O<<"BEGIN_M_NAME"<<endl;
	O<<getModuleName()<<endl;
	O<<"END M_NAME"<<endl;
	
	O<<"BEGIN F_B_LINENUM"<<endl;//will not be useful anymore 03/28/16
	O<<getStartLineNum()<<endl;
	O<<"END F_B_LINENUM"<<endl;
			
	O<<"BEGIN F_E_LINENUM"<<endl;//will not be useful anymore 03/28/16
	O<<getEndLineNum()<<endl;
	O<<"END F_E_LINENUM"<<endl;

	O<<"BEGIN F_BPOINT"<<endl;
	O<<isBFCPoint<<endl;
	O<<"END F_BPOINT"<<endl;

    //Added by Hui 03/28/16, keep record of all line#s in this func
    set<int>::iterator s_int_i;
    O<<"BEGIN ALL_LINES"<<endl;
    for (s_int_i = allLineNums.begin(); s_int_i != allLineNums.end(); s_int_i++)
        O<<*s_int_i<<" ";
    O<<endl;
    O<<"END ALL_LINES"<<endl;

	//This is to iterate all nodes in impVertices
    set<NodeProps *>::iterator ivh_i;
	for (ivh_i = impVertices.begin(); ivh_i != impVertices.end(); ivh_i++)
	{
		NodeProps *ivp = (*ivh_i);
		
		if (!ivp->isExported)
			continue;
		
		O<<"BEGIN VAR  "<<endl;
		
		O<<"BEGIN V_NAME "<<endl;
		O<<ivp->name<<endl;
		O<<"END V_NAME "<<endl;
		
		O<<"BEGIN V_REAL_NAME "<<endl;
        if (ivp->realName.empty())
            O<<ivp->name<<endl;
        else
		    O<<ivp->realName<<endl;
		O<<"END V_REAL_NAME "<<endl;
		
		O<<"BEGIN V_TYPE "<<endl;
		O<<ivp->eStatus<<endl;
		
		O<<"END V_TYPE "<<endl;
		
		O<<"BEGIN N_TYPE "<<endl;
		for (int a = 0; a < NODE_PROPS_SIZE; a++)
			O<<ivp->nStatus[a]<<" ";
			
		O<<endl;
		
		O<<"END N_TYPE"<<endl;
				
	
		O<<"BEGIN DECLARED_LINE"<<endl;
		O<<ivp->line_num<<endl;
		O<<"END DECLARED_LINE"<<endl;
		
		O<<"BEGIN IS_WRITTEN"<<endl;
		O<<ivp->isWritten<<endl;
		O<<"END IS_WRITTEN"<<endl;
		
		//used to iterate all set members of ivp
		set<NodeProps *>::iterator set_ivp_i;
		
		
/*		O<<"BEGIN PARENTS"<<endl;
		for (set_ivp_i = ivp->parents.begin(); set_ivp_i != ivp->parents.end(); set_ivp_i++)
		{
			NodeProps * ivpParent = (*set_ivp_i);
			O<<ivpParent->name<<endl;
		}
		O<<"END PARENTS"<<endl;	
	*/
		
		O<<"BEGIN CHILDREN "<<endl;
		for (set_ivp_i = ivp->children.begin(); set_ivp_i != ivp->children.end(); set_ivp_i++)
		{
			NodeProps *ivpChild = (*set_ivp_i);
			O<<ivpChild->name<<endl;
		}
		O<<"END CHILDREN"<<endl;
	
		

		O<<"BEGIN ALIASES "<<endl;
		for (set_ivp_i = ivp->aliases.begin(); set_ivp_i != ivp->aliases.end(); set_ivp_i++)
		{
			NodeProps * ivpAlias = (*set_ivp_i);
			O<<ivpAlias->name<<endl;
		}
		O<<"END ALIASES "<<endl;
	
		
		O<<"BEGIN DATAPTRS "<<endl;
		for (set_ivp_i = ivp->dataPtrs.begin(); set_ivp_i != ivp->dataPtrs.end(); set_ivp_i++)
		{
			NodeProps * ivpAlias = (*set_ivp_i);
			O<<ivpAlias->name<<endl;
		}
		O<<"END DATAPTRS "<<endl;
		

		O<<"BEGIN DFALIAS "<<endl;
		for (set_ivp_i = ivp->dfAliases.begin(); set_ivp_i != ivp->dfAliases.end(); set_ivp_i++)
		{
			NodeProps * child = *set_ivp_i;
			O<<child->name<<endl;
		}
		O<<"END DFALIAS "<<endl;
		
		O<<"BEGIN DFCHILDREN "<<endl;
		for (set_ivp_i = ivp->dfChildren.begin(); set_ivp_i != ivp->dfChildren.end(); set_ivp_i++)
		{
			NodeProps * child = *set_ivp_i;
			O<<child->name<<endl;
		}
		O<<"END DFCHILDREN "<<endl;
		

		O<<"BEGIN RESOLVED_LS "<<endl;
		for (set_ivp_i = ivp->resolvedLS.begin(); set_ivp_i != ivp->resolvedLS.end(); set_ivp_i++)
		{
			NodeProps * child = *set_ivp_i;
			O<<child->name<<endl;
		}
		O<<"END RESOLVED_LS "<<endl;		
		
		
		O<<"BEGIN RESOLVEDLS_FROM "<<endl;
		for (set_ivp_i = ivp->resolvedLSFrom.begin(); set_ivp_i != ivp->resolvedLSFrom.end(); set_ivp_i++)
		{			
			NodeProps * child = *set_ivp_i;
			O<<child->name<<endl;
		}
		O<<"END RESOLVEDLS_FROM "<<endl;

	
		O<<"BEGIN RESOLVEDLS_SE "<<endl;
		for (set_ivp_i = ivp->resolvedLSSideEffects.begin(); set_ivp_i != ivp->resolvedLSSideEffects.end(); set_ivp_i++)
		{			
			NodeProps * child = *set_ivp_i;
			O<<child->name<<endl;
		}		
		O<<"END RESOLVEDLS_SE "<<endl;
		

		O<<"BEGIN STORES_TO" <<endl;
		for (set_ivp_i = ivp->storesTo.begin(); set_ivp_i != ivp->storesTo.end(); set_ivp_i++)
		{
			NodeProps *child = *set_ivp_i;
			O<<child->name<<endl;
		}
		O<<"END STORES_TO"<<endl;
	
		
		O<<"BEGIN FIELDS "<<endl;
		for (set_ivp_i = ivp->fields.begin(); set_ivp_i != ivp->fields.end(); set_ivp_i++)
		{
			NodeProps * ivpAlias = (*set_ivp_i);
			O<<ivpAlias->name<<endl;
		}
		O<<"END FIELDS "<<endl;
		
		O<<"BEGIN FIELD_ALIAS"<<endl;
		if (ivp->fieldAlias == NULL)
			O<<"NULL"<<endl;
		else 
			O<<ivp->fieldAlias->name<<endl;
		O<<"END FIELD_ALIAS"<<endl;
		

/*
		O<<"BEGIN ALIASEDFROM "<<endl;
		if (ivp->pointsTo == NULL)
			O<<"NULL"<<endl;
		else
			O<<ivp->pointsTo->name<<endl;
		O<<"END ALIASEDFROM "<<endl;
	*/
	
		O<<"BEGIN GEN_TYPE"<<endl;
		const llvm::Type * origT = 0;	
		
		bool assigned = false;
		
		if (ivp->sField != NULL)
		{
			 if (ivp->sField->typeName.find("VOID") != string::npos)
			 {
				blame_info<<"Void for "<<ivp->name<<endl;
				 O<<ivp->sField->typeName<<endl;
				 assigned = true;
			 }
		}
			
			
		if (assigned)
		{
		
		}
		else if (ivp->llvm_inst != NULL && isa<Instruction>(ivp->llvm_inst))
		{
			Instruction * pi = cast<Instruction>(ivp->llvm_inst);	
			
            if (pi == NULL)
			{
				O<<"UNDEF"<<endl;
			}
			else
			{
				origT = pi->getType();					
				string origTStr = returnTypeName(origT, string(" "));
				O<<origTStr<<endl;
			}
		}
		else
		{
            if(ivp->llvm_inst != NULL){//it could be a constant(when ivp is a gv)
                origT = ivp->llvm_inst->getType();
                string origTStr = returnTypeName(origT, string(" "));
                O<<origTStr<<endl;
            }
            else
			    O<<"UNDEF"<<endl;
		}
		
		O<<"END GEN_TYPE"<<endl;
		
	
		O<<"BEGIN STRUCTTYPE"<<endl;
		if (ivp->sBFC == NULL) {
            /*if(ivp->llvm_inst != NULL){ //added by Hui 01/21/16
                llvm::Type *origT = ivp->llvm_inst->getType();
                unsigned typeVal = origT->getTypeID();
                if(typeVal == Type::StructTyID){
                    const llvm::StructType *type = cast<StructType>(origT);
                    string structNameReal = type->getStructName().str();
                    blame_info<<"Create structName("<<structNameReal<<
                        ") for "<<ivp->name<<endl;
                    O<<structNameReal<<endl;
                }
                else O<<"NULL"<<endl;
            }*/
            //If you really need chpl_** struct, you can first make sure
            //the GEN_TYPE has "Struct" and its llvm_inst is either instruction
            //or constantExpr, then get the operand(0) of it and apply above method
            O<<"NULL"<<endl;
        }
		else
			O<<ivp->sBFC->structName<<endl;
		O<<"END STRUCTTYPE "<<endl;
	
			
		O<<"BEGIN STRUCTPARENT "<<endl;
		if (ivp->sField == NULL)
			O<<"NULL"<<endl;
		else if (ivp->sField->parentStruct == NULL)
			O<<"NULL"<<endl;
		else
			O<<ivp->sField->parentStruct->structName<<endl;
		O<<"END STRUCTPARENT "<<endl;
		
		O<<"BEGIN STRUCTFIELDNUM "<<endl;
		if (ivp->sField == NULL)
			O<<"NULL"<<endl;
		else
			O<<ivp->sField->fieldNum<<endl;
			
		O<<"END STRUCTFIELDNUM "<<endl;
		
		O<<"BEGIN STOREFROM "<<endl;
		if (ivp->storeFrom == NULL)
			O<<"NULL"<<endl;
		else
			O<<ivp->storeFrom->name<<endl;//changed by Hui 01/29/16, before it was just 'storeFrom'
			
		O<<"END STOREFROM "<<endl;


/*
		O<<"BEGIN EXITV "<<endl;
		if (ivp->exitV == NULL)
			O<<"NULL"<<endl;
		else
			O<<ivp->exitV->name<<endl;
		O<<"END EXITV "<<endl;
	*/
	
	/*
		O<<"BEGIN PARAMS"<<endl;
		for (set_ivp_i = ivp->descParams.begin(); set_ivp_i != ivp->descParams.end(); set_ivp_i++)
		{
			NodeProps * ivpAlias = (*set_ivp_i);
			O<<ivpAlias->name<<endl;
		}
		O<<"END PARAMS"<<endl;
		*/
		
		
		O<<"BEGIN PARAMS"<<endl;
		for (set_ivp_i = ivp->descParams.begin(); set_ivp_i != ivp->descParams.end(); set_ivp_i++)
		{
			NodeProps * ivpAlias = (*set_ivp_i);
			O<<ivpAlias->name<<endl;
		}
		O<<"END PARAMS"<<endl;

		
			
		O<<"BEGIN CALLS"<<endl;
		set<ImpFuncCall *>::iterator ifc_i;
		for (ifc_i = ivp->calls.begin(); ifc_i != ivp->calls.end(); ifc_i++)
		{
			ImpFuncCall * iFunc = (*ifc_i);
			O<<iFunc->callNode->name<<"  "<<iFunc->paramNumber<<endl;
		}
		O<<"END CALLS"<<endl;
		
		set<int>::iterator si_i;
		O<<"BEGIN DOM_LN "<<endl;
			for (si_i = ivp->domLineNumbers.begin(); si_i != ivp->domLineNumbers.end(); si_i++)
		{
			O<<*si_i<<endl;
		}
		O<<"END DOM_LN "<<endl;

		
		O<<"BEGIN LINENUMS "<<endl;
			for (si_i = ivp->descLineNumbers.begin(); si_i != ivp->descLineNumbers.end(); si_i++)
		{
			O<<*si_i<<endl;
		}
		O<<"END LINENUMS "<<endl;
		
		
		if (reads)
		{
			O<<"BEGIN READ_L_NUMS "<<endl;
			LineReadHash::iterator lrh_i;
			
			for (lrh_i = ivp->readLines.begin(); lrh_i != ivp->readLines.end(); lrh_i++)
			{
				int lineNum = lrh_i->first;
				int numReads = lrh_i->second;
				O<<lineNum<<" "<<numReads<<endl;
				//blame_info<<"For EV "<<ivp->name<<" --- Line num "<<lineNum<<" has "<<numReads<<" Reads."<<endl;
			}
			O<<"END READ_L_NUMS "<<endl;
		}

#ifdef ADD_MULTI_LOCALE
		O<<"BEGIN ISPID"<<endl;
	    O<<ivp->isPid<<endl;;
	    O<<"END ISPID"<<endl;
	
		O<<"BEGIN ISOBJ"<<endl;
        O<<ivp->isObj<<endl;
	    O<<"END ISOBJ"<<endl;

		O<<"BEGIN MYPID"<<endl;
        if (ivp->myPid == NULL)
            O<<"NULL"<<endl;
        else
	        O<<ivp->myPid->name<<endl;;
	    O<<"END MYPID"<<endl;
	
		O<<"BEGIN MYOBJ"<<endl;
        if (ivp->myObj == NULL)
            O<<"NULL"<<endl;
        else
            O<<ivp->myObj->name<<endl;
	    O<<"END MYOBJ"<<endl;

		O<<"BEGIN PIDALIASES "<<endl;
		for (set_ivp_i = ivp->pidAliases.begin(); set_ivp_i != ivp->pidAliases.end(); set_ivp_i++)
		{
			NodeProps *ivpPidAlias = (*set_ivp_i);
			O<<ivpPidAlias->name<<endl;
		}
		O<<"END PIDALIASES "<<endl;

		O<<"BEGIN OBJALIASES "<<endl;
		for (set_ivp_i = ivp->objAliases.begin(); set_ivp_i != ivp->objAliases.end(); set_ivp_i++)
		{
			NodeProps *ivpObjAlias = (*set_ivp_i);
			O<<ivpObjAlias->name<<endl;
		}
		O<<"END OBJALIASES "<<endl;

		O<<"BEGIN PPAS "<<endl;
		for (set_ivp_i = ivp->PPAs.begin(); set_ivp_i != ivp->PPAs.end(); set_ivp_i++)
		{
			NodeProps *ivpPPA = (*set_ivp_i);
			O<<ivpPPA->name<<endl;
		}
		O<<"END PPAS "<<endl;
#endif
		O<<"END VAR "<<ivp->name<<endl;
	}
	
	O<<"END FUNC "<<getSourceFuncName()<<endl;
}


void FunctionBFC::exportParams(ostream &O)
{
	O<<"FUNCTION "<<getSourceFuncName()<<" "<<voidReturn<<" "<<numPointerParams<<endl;
	vector<ExitVariable *>::iterator v_ev_i;
	for (v_ev_i = exitVariables.begin(); v_ev_i != exitVariables.end(); v_ev_i++)
	{
		ExitVariable * ev = *v_ev_i;
		if (ev->whichParam >= 0 && ev->vertex != NULL)//changed by Hui '>0'-->'>=0'
		{
			O<<ev->whichParam<<" "<<ev->realName<<" "<<ev->isStructPtr<<" "<<ev->vertex->descLineNumbers.size()<<endl;
			if (ev->vertex->descLineNumbers.size() > 0 && ev->vertex->descLineNumbers.size() < 20)
			{
				set<int>::iterator s_i_i;
				for (s_i_i = ev->vertex->descLineNumbers.begin(); s_i_i != ev->vertex->descLineNumbers.end(); s_i_i++)
					O<<*s_i_i<<" ";
				O<<endl;
			}
		}
	}
	O<<"END FUNCTION"<<endl;
}


void FunctionBFC::exportCalls(ostream &O, ExternFuncBFCHash & efInfo)
{
  //cout<<"Entering exportCalls"<<endl;
  /*hasParOrDistCalls = false; //Initialization for each function

  set<const char *, ltstr>::iterator s_ch_i;
  for (s_ch_i = funcCallNames.begin(); s_ch_i != funcCallNames.end(); s_ch_i++)	{
	O<<*s_ch_i;
    string ss(*s_ch_i);
	if (knownFuncsInfo.count(ss))
	  O<<" K ";
	else if (efInfo[ss])
	  O<<" E ";
	else if (isLibraryOutput(*s_ch_i))
	  O<<" O ";
	else if (strstr(*s_ch_i, "tmp"))
	  O<<" P ";
	else
	  O<<" U ";
			
	O<<endl;

    //Added by Hui 07/07/17: to tag the functions with parallel construct calls
    if(ss.compare("_waitEndCount")==0 || ss.find("chpl_executeOn")!=string::npos)
      hasParOrDistCalls = true;
  }*/
  set<FuncCall *>::iterator fc_i = funcCalls.begin(), fc_e = funcCalls.end();
  for (; fc_i != fc_e; fc_i++) {
    FuncCall *fc = *fc_i;
    //only care about call node
    if (fc->paramNumber == -2) { 
      O<<fc->funcName; //funcName is mangled with "--#'aaa'"
      O<<endl;
    }
  }
}

/* Prints out a series of dot files */
void FunctionBFC::printDotFiles(const char * strExtension, bool printImplicit)
{
  string s("DOT/");
 	
  string s_name = func->getName();
 	
	string extension(strExtension);
	
  s += s_name;
  s += extension;
 	
  ofstream ofs(s.c_str());
 	
  if (!printImplicit)
	printToDot(ofs, NO_PRINT, PRINT_INST_TYPE, PRINT_LINE_NUM, NULL, 0);	
  else
	printToDot(ofs, PRINT_IMPLICIT, PRINT_INST_TYPE, PRINT_LINE_NUM, NULL, 0 );

  string s2("DOTP/");
  s2 += s_name;
  s2 += extension;

  ofstream ofs2(s2.c_str());

  printToDotPretty(ofs2, PRINT_IMPLICIT, PRINT_INST_TYPE, PRINT_LINE_NUM, NULL, 0 );

	
  //int x[] = {Instruction::Load, Instruction::Store};
  //printToDot(ofs_ls, NO_PRINT, PRINT_INST_TYPE, PRINT_LINE_NUM, x, 2 );
}



void FunctionBFC::printFinalDot(bool printAllLines, string ext)
{
	string s("DOT/");
 	
  string s_name = func->getName();
 	
	
	  s += s_name;
	  s += ext;
	
	if (printAllLines)
	{
			string extension("_FINAL_AL.dot");
	  s += extension;
	}
	else
	{
			string extension("_FINAL.dot");
		  s += extension;
	}
 	
  ofstream O(s.c_str());
	
	property_map<MyTruncGraphType, edge_iore_t>::type edge_type
	= get(edge_iore, G_trunc);
	
  O<<"digraph G {"<<endl;
  graph_traits<MyTruncGraphType>::vertex_iterator i, v_end;
  for(tie(i,v_end) = vertices(G_trunc); i != v_end; ++i) 
	{
		
		NodeProps * v = get(get(vertex_props, G_trunc),*i);
		if (!v)
		{
#ifdef DEBUG_ERROR		
			blame_info<<"Null IVP for "<<*i<<" in printFinalDot<<"<<endl;
			cerr<<"Null V in printFinalDot\n";
#endif			
			continue;
		}
		
		//int v_index = v->number;
		
		int lineNum = v->line_num;
		int lineNumOrder = v->lineNumOrder;
		//if (v->vp != NULL)
		//	lineNum = v->vp->line_num;
		
		// Print out the nodes
		if (v->eStatus >= EXIT_VAR_PARAM) //changed by Hui 03/15/16, from > to >=, 
		{                               //same changes to other EXIT_VAR_PARAM
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<")"<<"P# "<<v->eStatus - EXIT_VAR_PARAM;
			O<<"\",shape=invtriangle, style=filled, fillcolor=green]\n";		
		}
		else if (v->eStatus == EXIT_VAR_RETURN)
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=invtriangle, style=filled, fillcolor=yellow]\n";		
		}
		else if (v->eStatus == EXIT_VAR_GLOBAL)
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=invtriangle, style=filled, fillcolor=purple]\n";		
		}
		else if (v->nStatus[EXIT_PROG])
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=invtriangle, style=filled, fillcolor=dodgerblue]\n";	
		}
		else if (v->nStatus[EXIT_VAR_PTR])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				O<<"("<<v->pointsTo->name<<")";
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";		
			if (v->isWritten)
				O<<"\",shape=Mdiamond, style=filled, fillcolor=yellowgreen]\n";	
			else
				O<<"\",shape=Mdiamond, style=filled, fillcolor=wheat3]\n";	

		}
		else if (v->nStatus[EXIT_VAR_ALIAS])
		{
		if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				O<<"("<<v->pointsTo->name<<")";			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=seagreen4]\n";	
		}
		else if (v->nStatus[EXIT_VAR_A_ALIAS])
		{
		if (v->storeFrom == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				O<<"("<<v->storeFrom->name<<")";			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=powderblue]\n";	
		}
		else if (v->nStatus[EXIT_VAR_FIELD])
		{
			if (v->pointsTo == NULL)
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}
			}
			else
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						//string fullName;
						//getStructName(v,fullName);
						//O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<fullName;
					
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}				
				O<<"("<<v->pointsTo->name<<")";
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";						
			O<<"\",shape=Mdiamond, style=filled, fillcolor=grey]\n";	
		}
		else if (v->nStatus[EXIT_VAR_FIELD_ALIAS])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				
				if (v->pointsTo->sField == NULL)
					O<<"("<<v->pointsTo->name<<")";
				else
				{
					O<<"("<<v->pointsTo->sField->parentStruct->structName;
					O<<"."<<v->pointsTo->sField->fieldName<<")";
				}
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";						
			O<<"\",shape=Mdiamond, style=filled, fillcolor=grey38]\n";	
		}
		else if (v->eStatus == EXIT_OUTP)
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=invtriangle, style=filled, fillcolor=orange]\n";
		}
		else if (v->nStatus[LOCAL_VAR] )
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=rectangle, style=filled, fillcolor=pink]\n";
		}
		else if (v->nStatus[LOCAL_VAR_FIELD])
		{
			if (v->pointsTo == NULL)
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}
			}
			else
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}				
				O<<"(&"<<v->pointsTo->name<<")("<<v->name<<")";
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";						
			O<<"\",shape=parallelogram, style=filled, fillcolor=pink]\n";	
		}
		else if (v->nStatus[LOCAL_VAR_FIELD_ALIAS])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				
				if (v->pointsTo->sField == NULL)
					O<<"("<<v->pointsTo->name<<")";
				else
				{
					O<<"("<<v->pointsTo->sField->parentStruct->structName;
					O<<"."<<v->pointsTo->sField->fieldName<<")";
				}
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";						
			O<<"\",shape=parallelogram, style=filled, fillcolor=pink]\n";	
		}
		else if (v->nStatus[LOCAL_VAR_PTR])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				O<<"("<<v->pointsTo->name<<")";
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";		
			if (v->isWritten)
				O<<"\",shape=Mdiamond, style=filled, fillcolor=pink]\n";	
			else
				O<<"\",shape=Mdiamond, style=filled, fillcolor=cornsilk]\n";			
		}
		else if (v->nStatus[LOCAL_VAR_ALIAS])
		{
		if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				O<<"("<<v->pointsTo->name<<")";			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=pink]\n";	
		}
		else if (v->nStatus[LOCAL_VAR_A_ALIAS])
		{
			if (v->storeFrom == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				O<<"("<<v->storeFrom->name<<")";			
				O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=skyblue3]\n";	
		}
		else if (v->nStatus[CALL_NODE])
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=tripleoctagon, style=filled, fillcolor=red]\n";
		}
		else if (v->nStatus[CALL_PARAM])
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=doubleoctagon, style=filled, fillcolor=red]\n";
		}
		else if (v->nStatus[CALL_RETURN] )
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=octagon, style=filled, fillcolor=red]\n";
		}
        //added by Hui 03/22/16 to take care of important registers
        else if (v->nStatus[IMP_REG])
        {
			if (v->storeFrom == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				O<<"("<<v->storeFrom->name<<")";			
				O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=square, style=filled, fillcolor=brown]\n";	
        }
		else
		{
		#ifdef DEBUG_ERROR
			blame_info<<"ERROR! why didn't this node count toward something for "<<v->name<<" "<<v->eStatus<<endl;
			cerr<<"ERROR! why didn't this node count toward something for "<<v->name<<" "<<v->eStatus<<endl;
	#endif
		}

	}
	
	
	//  a -> b [label="hello", style=dashed];
	graph_traits<MyGraphType>::edge_iterator ei, edge_end;
	for(tie(ei, edge_end) = edges(G_trunc); ei != edge_end; ++ei) 
	{
		int opCode = get(get(edge_iore, G_trunc),*ei);
	
		if (opCode == DF_CHILD_EDGE)
			continue;
				
		int paramNum = -1; //TOCHECK: should it be -2 ? Hui 12/31/15 
		if (opCode >= CALL_EDGE)
			paramNum = opCode - CALL_EDGE;
		
		int sourceV = get(get(vertex_index, G_trunc), source(*ei, G_trunc));
		int targetV = get(get(vertex_index, G_trunc), target(*ei, G_trunc));
		
		O<< sourceV  << "->" << targetV;
		
		if (opCode == ALIAS_EDGE )
			O<< "[label=\""<<"A"<<"\", color=powderblue]";
		//else if (opCode == CHILD_EDGE)
			//O<< "[label=\""<<"C"<<"\", color=green]";
		else if (opCode == PARENT_EDGE )
			O<< "[label=\""<<"P"<<"\", color=orange]";
		else if (opCode == DATA_EDGE )
			O<< "[label=\""<<"D"<<"\", color=green]";
		else if (opCode == FIELD_EDGE )
			O<< "[label=\""<<"F"<<"\", color=grey]";
		else if (opCode == DF_ALIAS_EDGE )
			O<< "[label=\""<<"DFA "<<"\", color=powderblue]";
	    else if (opCode == DF_CHILD_EDGE )
			O<< "[label=\""<<"DFC "<<"\", color=purple]";
		else if (opCode == DF_INST_EDGE )
			O<< "[label=\""<<"S"<<"\", color=powderblue, style=dashed]";
		else if (opCode == CALL_PARAM_EDGE )
			O<< "[label=\""<<"CP"<<"\", color=red, style=dashed]";
		else if (opCode >= CALL_EDGE )
			O<< "[label=\""<<"Call Param "<<paramNum<<"\", color=red]";


		
		O<<" ;"<<endl;	
	}
	
	if (printAllLines)
	{
		int iS = impVertices.size();
		for(tie(i,v_end) = vertices(G_trunc); i != v_end; ++i) 
		{
			NodeProps * v = get(get(vertex_props, G_trunc),*i);

			
			O<<iS+v->impNumber+1<<"[label=\"";
			
			set<int>::iterator set_i_i;
		
			//for (set_i_i = v->lineNumbers.begin(); set_i_i != v->lineNumbers.end(); set_i_i++)
			for (set_i_i = v->descLineNumbers.begin(); set_i_i != v->descLineNumbers.end(); set_i_i++)
			{
				O<<(*set_i_i)<<" ";
			}
			O<<"\", shape = rectangle]\n";
			
			O<<v->impNumber<<"->"<<iS+v->impNumber+1<<endl;
			
		}
	}
	O<<"}";
}


void FunctionBFC::printFinalDotPretty(bool printAllLines, string ext)
{
	string s("DOTP/");
	//printAllLines = false;
 	
  string s_name = func->getName();
 	
	
	  s += s_name;
	  s += ext;
	
	if (printAllLines)
	{
			string extension("_FINAL_AL.dot");
	  s += extension;
	}
	else
	{
			string extension("_FINAL.dot");
		  s += extension;
	}
 	
  ofstream O(s.c_str());
	
	property_map<MyTruncGraphType, edge_iore_t>::type edge_type
	= get(edge_iore, G_trunc);
	
  O<<"digraph G {"<<endl;
  graph_traits<MyTruncGraphType>::vertex_iterator i, v_end;
  for(tie(i,v_end) = vertices(G_trunc); i != v_end; ++i) 
	{
		
		NodeProps * v = get(get(vertex_props, G_trunc),*i);
		if (!v)
		{
#ifdef DEBUG_ERROR		
			blame_info<<"Null IVP for "<<*i<<" in printFinalDot<<"<<endl;
			cerr<<"Null V in printFinalDot\n";
#endif			
			continue;
		}
		
		//int v_index = v->number;
		
		//int lineNum = v->line_num;
		//int lineNumOrder = v->lineNumOrder;
		//if (v->vp != NULL)
		//	lineNum = v->vp->line_num;
		
		// Print out the nodes
		if (v->eStatus >= EXIT_VAR_PARAM)
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<"P# "<<v->eStatus - EXIT_VAR_PARAM;
			O<<"\",shape=invtriangle, style=filled, fillcolor=green]\n";		
		}
		else if (v->eStatus == EXIT_VAR_RETURN)
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<"\",shape=invtriangle, style=filled, fillcolor=yellow]\n";		
		}
		else if (v->eStatus == EXIT_VAR_GLOBAL)
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<"\",shape=invtriangle, style=filled, fillcolor=purple]\n";		
		}
		else if (v->nStatus[EXIT_PROG])
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<"\",shape=invtriangle, style=filled, fillcolor=dodgerblue]\n";	
		}
		else if (v->nStatus[EXIT_VAR_PTR])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				//O<<"("<<v->pointsTo->name<<")";
			}
			if (v->isWritten)
				O<<"\",shape=Mdiamond, style=filled, fillcolor=yellowgreen]\n";	
			else
				O<<"\",shape=Mdiamond, style=filled, fillcolor=wheat3]\n";	

		}
		else if (v->nStatus[EXIT_VAR_ALIAS])
		{
		if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				//O<<"("<<v->pointsTo->name<<")";			
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=seagreen4]\n";	
		}
		else if (v->nStatus[EXIT_VAR_A_ALIAS])
		{
		if (v->storeFrom == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				//O<<"("<<v->storeFrom->name<<")";			
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=powderblue]\n";	
		}
		else if (v->nStatus[EXIT_VAR_FIELD])
		{
			if (v->pointsTo == NULL)
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}
			}
			else
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						//string fullName;
						//getStructName(v,fullName);
						//O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<fullName;
					
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}				
				O<<"("<<v->pointsTo->name<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=grey]\n";	
		}
		else if (v->nStatus[EXIT_VAR_FIELD_ALIAS])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				
				if (v->pointsTo->sField == NULL)
					O<<"("<<v->pointsTo->name<<")";
				else
				{
					O<<"("<<v->pointsTo->sField->parentStruct->structName;
					O<<"."<<v->pointsTo->sField->fieldName<<")";
				}
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=grey38]\n";	
		}
		else if (v->eStatus == EXIT_OUTP)
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<"\",shape=invtriangle, style=filled, fillcolor=orange]\n";
		}
		else if (v->nStatus[LOCAL_VAR] )
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<"\",shape=rectangle, style=filled, fillcolor=pink]\n";
		}
		else if (v->nStatus[LOCAL_VAR_FIELD])
		{
			if (v->pointsTo == NULL)
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}
			}
			else
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}				
				//O<<"(&"<<v->pointsTo->name<<")("<<v->name<<")";
				O<<"("<<v->name<<")";

			}
			O<<"\",shape=parallelogram, style=filled, fillcolor=pink]\n";	
		}
		else if (v->nStatus[LOCAL_VAR_FIELD_ALIAS])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				
				if (v->pointsTo->sField == NULL)
					O<<"("<<v->pointsTo->name<<")";
				else
				{
					O<<"("<<v->pointsTo->sField->parentStruct->structName;
					O<<"."<<v->pointsTo->sField->fieldName<<")";
				}
			}
			O<<"\",shape=parallelogram, style=filled, fillcolor=pink]\n";	
		}
		else if (v->nStatus[LOCAL_VAR_PTR])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				//O<<"("<<v->pointsTo->name<<")";
			}
			if (v->isWritten)
				O<<"\",shape=Mdiamond, style=filled, fillcolor=pink]\n";	
			else
				O<<"\",shape=Mdiamond, style=filled, fillcolor=cornsilk]\n";			
		}
		else if (v->nStatus[LOCAL_VAR_ALIAS])
		{
		if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				//O<<"("<<v->pointsTo->name<<")";			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=pink]\n";	
		}
		else if (v->nStatus[LOCAL_VAR_A_ALIAS])
		{
			if (v->storeFrom == NULL)
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
				//O<<"("<<v->storeFrom->name<<")";			
				//O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=skyblue3]\n";	
		}
		else if (v->nStatus[CALL_NODE])
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			//O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=tripleoctagon, style=filled, fillcolor=red]\n";
		}
		else if (v->nStatus[CALL_PARAM])
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			//O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=doubleoctagon, style=filled, fillcolor=red]\n";
		}
		else if (v->nStatus[CALL_RETURN] )
		{
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			//O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=octagon, style=filled, fillcolor=red]\n";
		}
        //added by Hui 03/22/16 to take care of important registers
        else if (v->nStatus[IMP_REG])
        {
			O<<get(get(vertex_index, G_trunc),*i)<<"[label=\""<<v->name;
			O<<"\",shape=square, style=filled, fillcolor=brown]\n";	
        }
		else
		{
		#ifdef DEBUG_ERROR
			blame_info<<"ERROR! why didn't this node count toward something for "<<v->name<<" "<<v->eStatus<<endl;
			cerr<<"ERROR! why didn't this node count toward something for "<<v->name<<" "<<v->eStatus<<endl;
	#endif
		}

	}
	
	
	//  a -> b [label="hello", style=dashed];
	graph_traits<MyGraphType>::edge_iterator ei, edge_end;
	for(tie(ei, edge_end) = edges(G_trunc); ei != edge_end; ++ei) 
	{
		int opCode = get(get(edge_iore, G_trunc),*ei);
	
		if (opCode == DF_CHILD_EDGE || opCode == DF_ALIAS_EDGE || opCode == DF_CHILD_EDGE)
			continue;
				
		int paramNum = -1;//TOCHECK: should it be -2? Hui 12/31/15
		if (opCode >= CALL_EDGE)
			paramNum = opCode - CALL_EDGE;
		
		int sourceV = get(get(vertex_index, G_trunc), source(*ei, G_trunc));
		int targetV = get(get(vertex_index, G_trunc), target(*ei, G_trunc));
		
		O<< sourceV  << "->" << targetV;
		
		if (opCode == ALIAS_EDGE )
			O<< "[label=\""<<"A"<<"\", color=powderblue]";
		else if (opCode == PARENT_EDGE )
			O<< "[label=\""<<"B"<<"\", color=orange]";
		else if (opCode == DATA_EDGE )
			O<< "[label=\""<<"D"<<"\", color=green]";
		else if (opCode == FIELD_EDGE )
			O<< "[label=\""<<"F"<<"\", color=grey]";
		//else if (opCode == DF_ALIAS_EDGE )
			//O<< "[label=\""<<"DFA "<<"\", color=powderblue]";
	    //else if (opCode == DF_CHILD_EDGE )
		//	O<< "[label=\""<<"DFC "<<"\", color=purple]";
		else if (opCode == DF_INST_EDGE )
			O<< "[label=\""<<"S"<<"\", color=powderblue, style=dashed]";
		else if (opCode == CALL_PARAM_EDGE )
			O<< "[label=\""<<"CP"<<"\", color=red, style=dashed]";
		else if (opCode >= CALL_EDGE )
			O<< "[label=\""<<"Call Param "<<paramNum<<"\", color=red]";


		
		O<<" ;"<<endl;	
	}
	
	if (printAllLines)
	{
		int iS = impVertices.size();
		for(tie(i,v_end) = vertices(G_trunc); i != v_end; ++i) 
		{
			NodeProps * v = get(get(vertex_props, G_trunc),*i);

			
			O<<iS+v->impNumber+1<<"[label=\"";
			
			set<int>::iterator set_i_i;
		
			//for (set_i_i = v->lineNumbers.begin(); set_i_i != v->lineNumbers.end(); set_i_i++)
			for (set_i_i = v->descLineNumbers.begin(); set_i_i != v->descLineNumbers.end(); set_i_i++)
			{
				O<<(*set_i_i)<<" ";
			}
			O<<"\", shape = rectangle]\n";
			
			O<<v->impNumber<<"->"<<iS+v->impNumber+1<<endl;
			
		}
	}
	O<<"}";
}



void FunctionBFC::printFinalDotAbbr(string ext)
{
	bool printAllLines = true;

	string s("DOT/");
 	
  string s_name = func->getName();
 	
	
	  s += s_name;
	  s += ext;

	
		string extension("_FINAL_abbr.dot");
	  s += extension;
	
 	
  ofstream O(s.c_str());
	
	property_map<MyTruncGraphType, edge_iore_t>::type edge_type
	= get(edge_iore, G_abbr);
	
  O<<"digraph G {"<<endl;
  graph_traits<MyTruncGraphType>::vertex_iterator i, v_end;
  for(tie(i,v_end) = vertices(G_abbr); i != v_end; ++i) 
	{
		
		NodeProps * v = get(get(vertex_props, G_abbr),*i);
		if (!v)
		{
#ifdef DEBUG_ERROR		
			blame_info<<"Null IVP for "<<*i<<" in printFinalDot<<"<<endl;
			cerr<<"Null V in printFinalDot\n";
#endif			
			continue;
		}
		
		//int v_index = v->number;
		
		int lineNum = v->line_num;
		int lineNumOrder = v->lineNumOrder;
		//if (v->vp != NULL)
		//	lineNum = v->vp->line_num;
		
		// Print out the nodes
		if (v->eStatus >= EXIT_VAR_PARAM)
		{
			O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<")"<<"P# "<<v->eStatus - EXIT_VAR_PARAM;
			O<<"\",shape=invtriangle, style=filled, fillcolor=green]\n";		
		}
		else if (v->eStatus == EXIT_VAR_RETURN)
		{
			O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=invtriangle, style=filled, fillcolor=yellow]\n";		
		}
		else if (v->eStatus == EXIT_VAR_GLOBAL)
		{
			O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=invtriangle, style=filled, fillcolor=purple]\n";		
		}/*
		else if (v->nStatus[EXIT_PROG])
		{
			O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=invtriangle, style=filled, fillcolor=dodgerblue]\n";	
		}
		else if (v->nStatus[EXIT_VAR_PTR])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
				O<<"("<<v->pointsTo->name<<")";
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";		
			if (v->isWritten)
				O<<"\",shape=Mdiamond, style=filled, fillcolor=yellowgreen]\n";	
			else
				O<<"\",shape=Mdiamond, style=filled, fillcolor=wheat3]\n";	

		}
		else if (v->nStatus[EXIT_VAR_ALIAS])
		{
		if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
				O<<"("<<v->pointsTo->name<<")";			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=seagreen4]\n";	
		}
		else if (v->nStatus[EXIT_VAR_A_ALIAS])
		{
		if (v->storeFrom == NULL)
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
				O<<"("<<v->storeFrom->name<<")";			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=powderblue]\n";	
		}*/
		else if (v->nStatus[EXIT_VAR_FIELD])
		{
			if (v->pointsTo == NULL)
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}
			}
			else
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}				
				O<<"("<<v->pointsTo->name<<")";
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";						
			if (v->isWritten)
				O<<"\",shape=Mdiamond, style=filled, fillcolor=grey]\n";	
			else
				O<<"\",shape=Mdiamond, style=filled, fillcolor=cadetblue]\n";	
		}
		else if (v->nStatus[EXIT_VAR_FIELD_ALIAS])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
				
				if (v->pointsTo->sField == NULL)
					O<<"("<<v->pointsTo->name<<")";
				else
				{
					O<<"("<<v->pointsTo->sField->parentStruct->structName;
					O<<"."<<v->pointsTo->sField->fieldName<<")";
				}
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";						
			O<<"\",shape=Mdiamond, style=filled, fillcolor=grey38]\n";	
		}
		else if (v->eStatus == EXIT_OUTP)
		{
			O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=invtriangle, style=filled, fillcolor=orange]\n";
		}
		else if (v->nStatus[LOCAL_VAR] )
		{
			O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=rectangle, style=filled, fillcolor=pink]\n";
		}
		else if (v->nStatus[LOCAL_VAR_FIELD])
		{
			if (v->pointsTo == NULL)
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}
			}
			else
			{
				if (v->sField == NULL)
				{
					O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<"NFP_"<<v->name;
				}
				else
				{
					if (v->sField->parentStruct == NULL)
					{
						O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<"NSP_"<<v->sField->fieldName;
					}
					else
					{
						O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->sField->parentStruct->structName;
						O<<"."<<v->sField->fieldName;
					}
				}				
				O<<"(&"<<v->pointsTo->name<<")("<<v->name<<")";
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";						
			if (v->isWritten)
				O<<"\",shape=parallelogram, style=filled, fillcolor=pink]\n";	
			else
				O<<"\",shape=parallelogram, style=filled, fillcolor=lightcyan]\n";	
		}
		else if (v->nStatus[LOCAL_VAR_FIELD_ALIAS])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
				
				if (v->pointsTo->sField == NULL)
					O<<"("<<v->pointsTo->name<<")";
				else
				{
					O<<"("<<v->pointsTo->sField->parentStruct->structName;
					O<<"."<<v->pointsTo->sField->fieldName<<")";
				}
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";						
			O<<"\",shape=parallelogram, style=filled, fillcolor=pink]\n";	
		}/*
		else if (v->nStatus[LOCAL_VAR_PTR])
		{
			if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
				O<<"("<<v->pointsTo->name<<")";
			}
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";		
			if (v->isWritten)
				O<<"\",shape=Mdiamond, style=filled, fillcolor=pink]\n";	
			else
				O<<"\",shape=Mdiamond, style=filled, fillcolor=wheat3]\n";			
		}
		else if (v->nStatus[LOCAL_VAR_ALIAS])
		{
		if (v->pointsTo == NULL)
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
				O<<"("<<v->pointsTo->name<<")";			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=pink]\n";	
		}
		else if (v->nStatus[LOCAL_VAR_A_ALIAS])
		{
			if (v->storeFrom == NULL)
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			else
			{
				O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
				O<<"("<<v->storeFrom->name<<")";			
				O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			}
			O<<"\",shape=Mdiamond, style=filled, fillcolor=skyblue3]\n";	
		}*/
		else if (v->nStatus[CALL_NODE])
		{
			O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=tripleoctagon, style=filled, fillcolor=red]\n";
		}
		else if (v->nStatus[CALL_PARAM])
		{
			O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=doubleoctagon, style=filled, fillcolor=red]\n";
		}
		else if (v->nStatus[CALL_RETURN] )
		{
			O<<get(get(vertex_index, G_abbr),*i)<<"[label=\""<<v->name;
			O<<":("<<lineNum<<":"<<lineNumOrder<<")";
			O<<"\",shape=octagon, style=filled, fillcolor=red]\n";
		}

	}
	
	
	//  a -> b [label="hello", style=dashed];
	graph_traits<MyGraphType>::edge_iterator ei, edge_end;
	for(tie(ei, edge_end) = edges(G_abbr); ei != edge_end; ++ei) 
	{
		
		int opCode = get(get(edge_iore, G_abbr),*ei);
		
		int paramNum = -1; //TOCHECK: should it be -2? Hui: 12/31/15
		if (opCode >= CALL_EDGE)
			paramNum = opCode - CALL_EDGE;
		
		int sourceV = get(get(vertex_index, G_abbr), source(*ei, G_abbr));
		int targetV = get(get(vertex_index, G_abbr), target(*ei, G_abbr));
		
		O<< sourceV  << "->" << targetV;
		
		if (opCode == ALIAS_EDGE )
			O<< "[label=\""<<"A"<<"\", color=powderblue]";
		//else if (opCode == CHILD_EDGE)
			//O<< "[label=\""<<"C"<<"\", color=green]";
		else if (opCode == PARENT_EDGE )
			O<< "[label=\""<<"P"<<"\", color=orange]";
		else if (opCode == DATA_EDGE )
			O<< "[label=\""<<"D"<<"\", color=green]";
		else if (opCode == FIELD_EDGE )
			O<< "[label=\""<<"F"<<"\", color=grey]";
		else if (opCode == DF_ALIAS_EDGE )
			O<< "[label=\""<<"DFA "<<"\", color=powderblue]";
		else if (opCode == DF_CHILD_EDGE )
			O<< "[label=\""<<"DFC "<<"\", color=purple]";
		else if (opCode == DF_INST_EDGE )
			O<< "[label=\""<<"S"<<"\", color=powderblue, style=dashed]";
		else if (opCode == CALL_PARAM_EDGE )
			O<< "[label=\""<<"CP"<<"\", color=red, style=dashed]";
		else if (opCode >= CALL_EDGE )
			O<< "[label=\""<<"Call Param "<<paramNum<<"\", color=red]";


		
		O<<" ;"<<endl;	
	}
	
	if (printAllLines)
	{
		int iS = impVertices.size();
		for(tie(i,v_end) = vertices(G_abbr); i != v_end; ++i) 
		{
			NodeProps * ivp = get(get(vertex_props, G_abbr),*i);
			if (isTargetNode(ivp))
			{

				O<<iS+ivp->impNumber+1<<"[label=\"";
			
				set<int>::iterator set_i_i;
		
				//for (set_i_i = v->lineNumbers.begin(); set_i_i != v->lineNumbers.end(); set_i_i++)
				for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++)
				{
					O<<(*set_i_i)<<" ";
				}
				
				O<<"\", shape = rectangle]\n";
				O<<ivp->impNumber<<"->"<<iS+ivp->impNumber+1<<endl;
			}
			
		}
	}
	O<<"}";
}



void FunctionBFC::printFinalLineNums(ostream &O)
{
		
	O<<"EXIT VARIABLES"<<endl;
	
	graph_traits<MyTruncGraphType>::vertex_iterator i, v_end;

	for(tie(i,v_end) = vertices(G_abbr); i != v_end; ++i) 
	{
		NodeProps * ivp = get(get(vertex_props, G_abbr),*i);
		if (isTargetNode(ivp))
		{
			if (ivp->eStatus >= EXIT_VAR_GLOBAL)
			{
				O<<ivp->name<<endl;
				set<int>::iterator set_i_i;		
				for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++)
				{
					O<<(*set_i_i)<<" ";
				}				
				O<<endl;
			}
		}
	}
	
	
	O<<"EXIT VAR FIELDS"<<endl;
	for(tie(i,v_end) = vertices(G_abbr); i != v_end; ++i) 
	{
		NodeProps * ivp = get(get(vertex_props, G_abbr),*i);
		if (isTargetNode(ivp))
		{
			if (ivp->nStatus[EXIT_VAR_FIELD])
			{
				O<<ivp->getFullName()<<endl;
				set<int>::iterator set_i_i;		
				for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++)
				{
					O<<(*set_i_i)<<" ";
				}				
				O<<endl;
			}
		}
	}

	
	
	O<<"LOCAL VARIABLES"<<endl;
	for(tie(i,v_end) = vertices(G_abbr); i != v_end; ++i) 
	{
		NodeProps * ivp = get(get(vertex_props, G_abbr),*i);
		if (isTargetNode(ivp))
		{
			if (ivp->nStatus[LOCAL_VAR])
			{
				if (ivp->isFakeLocal)
					continue;
			
				O<<ivp->name<<endl;
				set<int>::iterator set_i_i;		
				for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++)
				{
					O<<(*set_i_i)<<" ";
				}				
				O<<endl;
			}
		}
	}
	
	O<<"LOCAL VAR FIELDS"<<endl;
	for(tie(i,v_end) = vertices(G_abbr); i != v_end; ++i) 
	{
		NodeProps * ivp = get(get(vertex_props, G_abbr),*i);
		if (isTargetNode(ivp))
		{
			if (ivp->nStatus[LOCAL_VAR_FIELD])
			{
				set<NodeProps *> visited;
			
				NodeProps * fUpPtr = ivp->fieldUpPtr;
				NodeProps * topField = NULL;
				while (fUpPtr != NULL)
				{
					topField = fUpPtr;
					fUpPtr = fUpPtr->fieldUpPtr;
					if (visited.count(topField))
						fUpPtr = NULL;
					else
						visited.insert(topField);
				}
				
				if (topField == NULL)
					continue;
				
				if (topField->isFakeLocal)
					continue;
			
				O<<ivp->getFullName()<<endl;
				set<int>::iterator set_i_i;		
				for (set_i_i = ivp->descLineNumbers.begin(); set_i_i != ivp->descLineNumbers.end(); set_i_i++)
				{
					O<<(*set_i_i)<<" ";
				}				
				O<<endl;
			}
		}
	}

	//cout<<"Leaving printFinalLineNums for"<<endl;
	
	
}




void FunctionBFC::printTruncDotFiles(const char * strExtension, bool printImplicit)
{
  string s("DOT/");
 	
  string s_name = func->getName();
 	
	string extension(strExtension);
	
  s += s_name;
  s += extension;
 	
  ofstream ofs(s.c_str());
 	
	
	printToDotTrunc(ofs);
	
  //int x[] = {Instruction::Load, Instruction::Store};
  //printToDot(ofs_ls, NO_PRINT, PRINT_INST_TYPE, PRINT_LINE_NUM, x, 2 );
}


void FunctionBFC::printToDotTrunc(ostream &O)
{
	bool printLineNum = true;
	//int opSetSize = 0;
	
	
	property_map<MyGraphType, edge_iore_t>::type edge_type
	= get(edge_iore, G);
	
  O<<"digraph G {"<<endl;
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
  for(tie(i,v_end) = vertices(G); i != v_end; ++i) 
	{
		
		NodeProps * v = get(get(vertex_props, G),*i);
		//int v_index = v->number;
				
		if (!v)
		{
			//cerr<<"Null V in printToDot\n";
			continue;
		}
		
		//if (v->fbb != NULL)
			//cout<<"BB is "<<v->fbb->getName()<<endl;

#ifdef DEBUG_OUTPUT		
		blame_info<<"V "<<v->name<<" exit Status is "<<v->eStatus;
		blame_info<<v->nStatus[CALL_NODE]<<" "<<v->nStatus[CALL_PARAM];
		blame_info<<" "<<v->nStatus[CALL_RETURN]<<endl;
#endif

		if (v->eStatus >= EXIT_VAR_PARAM)
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
				if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")"<<"P# "<<v->eStatus - EXIT_VAR_PARAM;
				O<<"\",shape=invtriangle, style=filled, fillcolor=green]\n";		
		}
		else if (v->eStatus == EXIT_VAR_RETURN)
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
				if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=invtriangle, style=filled, fillcolor=yellow]\n";		
		}
		else if (v->eStatus == EXIT_VAR_GLOBAL)
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
				if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=invtriangle, style=filled, fillcolor=purple]\n";		
		}
		else if (v->nStatus[EXIT_PROG])
		{
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
								if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
				if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=invtriangle, style=filled, fillcolor=dodgerblue]\n";	
		}
		else if (v->nStatus[EXIT_VAR_PTR] && v->pointsTo != NULL)
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name<<"(&"<<v->pointsTo->name<<")("<<v->name<<")";
			if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";		
			if (v->isWritten)
				O<<"\",shape=invtriangle, style=filled, fillcolor=yellowgreen]\n";	
			else
				O<<"\",shape=invtriangle, style=filled, fillcolor=wheat3]\n";	

		}
		else if (v->nStatus[EXIT_VAR_ALIAS] && v->pointsTo != NULL)
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name<<"~("<<v->pointsTo->name<<")";
							if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";						
				
				O<<"\",shape=invtriangle, style=filled, fillcolor=seagreen4]\n";	
		}
		else if (v->nStatus[EXIT_VAR_A_ALIAS] && v->storeFrom != NULL)
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name<<"~("<<v->storeFrom->name<<")";
							if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";						
				
				O<<"\",shape=invtriangle, style=filled, fillcolor=powderblue]\n";	
		}
		else if (v->nStatus[EXIT_VAR_FIELD] && v->pointsTo != NULL)
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name<<"~("<<v->pointsTo->name<<")";
							if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";						
				
				O<<"\",shape=invtriangle, style=filled, fillcolor=grey]\n";	
		}
		else if (v->nStatus[EXIT_VAR_FIELD_ALIAS] && v->pointsTo != NULL)
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name<<"~("<<v->pointsTo->name<<")";
							if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";						
				
				O<<"\",shape=invtriangle, style=filled, fillcolor=grey58]\n";	
		}
		else if (v->eStatus == EXIT_OUTP)
		{
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
								if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
				if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=invtriangle, style=filled, fillcolor=orange]\n";
		}
		else if (v->nStatus[CALL_NODE])
		{
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
								if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=tripleoctagon, style=filled, fillcolor=red]\n";
		}
		else if (v->nStatus[CALL_PARAM])
		{
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
								if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=doubleoctagon, style=filled, fillcolor=red]\n";
		}
		else if (v->nStatus[CALL_RETURN])
		{
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
								if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=octagon, style=filled, fillcolor=red]\n";
		}
		else if (v->nStatus[LOCAL_VAR_PTR])
		{
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				if (v->pointsTo != NULL)
					O<<"("<<v->pointsTo->name<<")";
				if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
			if (v->isWritten)
				O<<"\",shape=Mdiamond, style=filled, fillcolor=pink]\n";	
			else
				O<<"\",shape=Mdiamond, style=filled, fillcolor=cornsilk]\n";	

		}
		else if (v->nStatus[LOCAL_VAR])
		{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				
				if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=rectangle, style=filled, fillcolor=pink]\n";		
		
		}
		else if (v->nStatus[LOCAL_VAR_ALIAS] || v->nStatus[LOCAL_VAR_FIELD] ||
							v->nStatus[LOCAL_VAR_FIELD_ALIAS])
		{
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				if (v->pointsTo != NULL)
					O<<"("<<v->pointsTo->name<<")";
				if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=Mdiamond, style=filled, fillcolor=pink]\n";		
		}
		else if (v->llvm_inst != NULL)
		{	
			if (isa<Instruction>(v->llvm_inst))
			{
				Instruction * pi = cast<Instruction>(v->llvm_inst);	
				
				
				const llvm::Type * t = pi->getType();
				unsigned typeVal = t->getTypeID();
				
				if (pi->getOpcode() == Instruction::Alloca)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					if (v->fbb != NULL)
						O<<":("<<v->fbb->getName()<<")";
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					if (v->isLocalVar == true)
						O<<"\",shape=Mdiamond, style=filled, fillcolor=white]\n";
					else if (v->isFormalArg)
						O<<"\",shape=Mdiamond, style=filled, fillcolor=white]\n";
					else
						O<<"\",shape=Mdiamond, style=filled, fillcolor=white]\n";
				}	
				//else if (pi->getOpcode() == Instruction::GetElementPtr)
				else if (v->pointsTo != NULL)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name<<"(&"<<v->pointsTo->name<<")";
									if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\",shape=Mdiamond, style=filled, fillcolor=white]\n";
				}
				else if(pi->getOpcode() == Instruction::Call || pi->getOpcode() == Instruction::Invoke)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
									if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\",shape=box, style=filled, fillcolor=white]\n";		
				}
				else if (typeVal == Type::PointerTyID)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
									if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\",shape=Mdiamond, style=filled, fillcolor=yellow]\n";
				}
				else
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
									if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\"];"<<"\n";
					
				}
			}
			else
			{
				//cerr<<"Not an instruction "<<v->name<<"\n";
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				
				if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";

				if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";

				O<<"\",shape=Mdiamond, style=filled, fillcolor=purple]\n";	
				O.flush();
				
			}
		}
		else
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				if (v->fbb != NULL)
					O<<":("<<v->fbb->getName()<<")";
			if (printLineNum)
				O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
			O<<"\",shape=Mdiamond, style=filled, fillcolor=white]\n";
			O.flush();
		}
	}
	
  //  a -> b [label="hello", style=dashed];
  graph_traits<MyGraphType>::edge_iterator ei, edge_end;
  for(tie(ei, edge_end) = edges(G); ei != edge_end; ++ei) {
		
    int opCode = get(get(edge_iore, G),*ei);
		/*
    if (opSetSize)
		{
			for (int a = 0; a < opSetSize; a++)
			{
				if (opSet[a] == opCode)
	      {
					O<< get(get(vertex_index, G), source(*ei, G)) << "->" << get(get(vertex_index, G), target(*ei, G));
					
					if ( opCode && printInstType)
						if (opCode != ALIAS_OP)
							O<< "[label=\""<<Instruction::getOpcodeName(get(get(edge_iore, G),*ei))<<"\"]";
						else
							O<< "[label=\""<<"ALIAS"<<"\"]";
					
					O<<" ;"<<endl;	
	      }
			}
		}
		*/
		bool printImplicit = true;
		bool printInstType = true;
    
			if ( opCode || printImplicit )	
			{
				int sourceV = get(get(vertex_index, G), source(*ei, G));
				int targetV = get(get(vertex_index, G), target(*ei, G));
				
				
				NodeProps * sourceVP = get(get(vertex_props, G), source(*ei, G));
				NodeProps * targetVP = get(get(vertex_props, G), target(*ei, G));
				
							
				O<< sourceV  << "->" << targetV;
				
				if (!opCode && printImplicit)
					O<< "[color=grey, style=dashed]";
				else if ( opCode && printInstType)
	      {
					if (opCode == ALIAS_OP)
						O<< "[label=\""<<"ALIAS"<<"\"]";
					else if (opCode == GEP_BASE_OP )
						O<< "[label=\""<<"P--BASE"<<"\", color=powderblue]";
					else if (opCode == GEP_OFFSET_OP )
						O<< "[label=\""<<"P__OFFSET"<<"\", color=powderblue]";
					else if (opCode == GEP_S_FIELD_VAR_OFF_OP )
						O<< "[label=\""<<"P__ARR_OFF"<<"\", color=powderblue]";		
					else if (opCode == RESOLVED_L_S_OP )
						O<< "[label=\""<<"LS"<<"\", color=blue]";		
					else if (opCode >= GEP_S_FIELD_OFFSET_OP )
						O<< "[label=\""<<"P__FIELD # "<<opCode - GEP_S_FIELD_OFFSET_OP<<"\", color=powderblue]";
					else if (opCode == RESOLVED_OUTPUT_OP)
						O<< "[label=\""<<"R_OUTPUT"<<"\", color=green]";
					else if (opCode == RESOLVED_EXTERN_OP )
						O<< "[label=\""<<"R_EXTERN"<<"\", color=pink]";
					else if (opCode == RESOLVED_MALLOC_OP )
						O<< "[label=\""<<"R_MALLOC"<<"\", color=red]";
					else if (opCode == Instruction::Call )
					{
						int paramNum = MAX_PARAMS + 1;
						//cerr<<"Call from "<<sourceVP->name<<" to "<<targetVP->name<<endl;
						
						set<FuncCall *>::iterator fc_i = sourceVP->funcCalls.begin();
						
						for (; fc_i != sourceVP->funcCalls.end(); fc_i++)
						{
							FuncCall * fc = *fc_i;
							
							if (fc->funcName == targetVP->name)
							{
								paramNum = fc->paramNumber;
								break;
							}
							
							
							//cerr<<"     PN is "<<fc->paramNumber<<" for func "<<fc->funcName<<endl;
						}
						
						O<< "[color=red, label=\""<<Instruction::getOpcodeName(opCode)<<" "<<paramNum<<"\"]";
					}
					else if (opCode == Instruction::Store )
					{
						O<<"[label=\"";
						set<int>::iterator set_int_i;
						
						for (set_int_i = targetVP->storeLines.begin(); set_int_i != targetVP->storeLines.end(); set_int_i++)
							O<<(*set_int_i)<<" ";
						
						O<<"\", color = green]";
					}
					else
						O<< "[label=\""<<Instruction::getOpcodeName(opCode)<<"\"]";
	      }
				O<<" ;"<<endl;	
			}
		
  }
  O<<"}";
}


/* Prints blame relationships to dot file
 printImplicit - also prints implicit relationship
 printInstType - also prints the type for each instruction 
 printLineNum - also prints linenum the instruction generated from
 */
void FunctionBFC::printToDot(ostream &O, bool printImplicit, bool printInstType, 
															 bool printLineNum,int * opSet, int opSetSize)
{  
  property_map<MyGraphType, edge_iore_t>::type edge_type
	= get(edge_iore, G);
	
  O<<"digraph G {"<<endl;
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
  for(tie(i,v_end) = vertices(G); i != v_end; ++i) 
	{
	#ifdef DEBUG_OUTPUT		
		blame_info<<"Getting v for "<<*i<<endl;
#endif
		
		NodeProps * v = get(get(vertex_props, G),*i);
		
		if (!v)
		{
#ifdef DEBUG_ERROR				
			blame_info<<"V is NULL"<<endl;
			cerr<<"Null V in printToDot\n";
			#endif
			continue;
		}
		
		int v_index = v->number;
		int in_d = in_degree(v_index, G);
		int out_d = out_degree(v_index, G);
		
		

		
		if (in_d == 0 && out_d == 0)
		{

			bool shouldIContinue = true;
			
			if (v->isLocalVar)
				shouldIContinue = false;
		
			//if (!(v->isLocalVar))
				//continue;
			
			
			//if (isa<Instruction>(v->llvm_inst))
		//	{
			//	Instruction * pi = cast<Instruction>(v->llvm_inst);	
				//if (pi->getOpcode() == Instruction::Call)
					//shouldIContinue = false;
			//}	
			
			if (v->name.find("--") != string::npos)
				shouldIContinue = false;
			
			if (shouldIContinue)
				continue;
		}
		
		if (v->llvm_inst != NULL)
		{	
			if (isa<Instruction>(v->llvm_inst))
			{
				Instruction * pi = cast<Instruction>(v->llvm_inst);	
								
				const llvm::Type * t = pi->getType();
				unsigned typeVal = t->getTypeID();
				
				if (pi->getOpcode() == Instruction::Alloca)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					if (v->isLocalVar == true)
						O<<"\",shape=Mdiamond, style=filled, fillcolor=forestgreen]\n";
					else if (v->isFormalArg)
						O<<"\",shape=Mdiamond, style=filled, fillcolor=green]\n";
					else
						O<<"\",shape=Mdiamond, style=filled, fillcolor=darkseagreen]\n";
				}	
				//else if (pi->getOpcode() == Instruction::GetElementPtr)
				else if (v->pointsTo != NULL)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<"&"<<v->pointsTo->name<<"("<<v->name<<")";
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\",shape=Mdiamond, style=filled, fillcolor=powderblue]\n";
				}
				else if(pi->getOpcode() == Instruction::Call || pi->getOpcode() == Instruction::Invoke)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\",shape=box, style=filled, fillcolor=red]\n";		
				}
				else if (typeVal == Type::PointerTyID)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\",shape=Mdiamond, style=filled, fillcolor=yellow]\n";
				}
				else
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\"];"<<"\n";
					
				}
			}
			else
			{
				//cerr<<"Not an instruction "<<v->name<<"\n";
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=Mdiamond, style=filled, fillcolor=purple]\n";	
				
			}
		}
		else
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
			if (printLineNum)
				O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
			O<<"\",shape=Mdiamond, style=filled, fillcolor=purple]\n";
		}
	}
	
  //  a -> b [label="hello", style=dashed];
  graph_traits<MyGraphType>::edge_iterator ei, edge_end;
  for(tie(ei, edge_end) = edges(G); ei != edge_end; ++ei) {
		
    int opCode = get(get(edge_iore, G),*ei);
		
    if (opSetSize)
	{
	    for (int a = 0; a < opSetSize; a++)
		{
			if (opSet[a] == opCode) 
            {
				O<< get(get(vertex_index, G), source(*ei, G)) << "->" << get(get(vertex_index, G), target(*ei, G));
					
				if ( opCode && printInstType) {
					if (opCode != ALIAS_OP)
						O<< "[label=\""<<Instruction::getOpcodeName(get(get(edge_iore, G),*ei))<<"\"]";
					else
						O<< "[label=\""<<"ALIAS"<<"\"]";
                }
				O<<" ;"<<endl;	
	        }
		}
	}
		
    else
		{
			if ( opCode || printImplicit )	
			{
				int sourceV = get(get(vertex_index, G), source(*ei, G));
				int targetV = get(get(vertex_index, G), target(*ei, G));
				
				
				NodeProps * sourceVP = get(get(vertex_props, G), source(*ei, G));
				NodeProps * targetVP = get(get(vertex_props, G), target(*ei, G));
				
				/*
				 if (opCode == Instruction::Store)
				 {
				 NodeProps *v = get(get(vertex_props, G), source(*ei,G));
				 if (!v)
				 {
				 cerr<<"Null V(2) in printToDot\n";
				 continue;
				 }
				 
				 O<<sourceV<<"[label=\""<<v->name;
				 if (printLineNum)
				 O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				 O<<"\",shape=Mdiamond]\n";
				 }
				 */
				
				O<< sourceV  << "->" << targetV;
				
				if (!opCode && printImplicit)
					O<< "[color=grey, style=dashed]";
				else if ( opCode && printInstType)
	      {
					if (opCode == ALIAS_OP)
						O<< "[label=\""<<"ALIAS"<<"\"]";
					else if (opCode == GEP_BASE_OP )
						O<< "[label=\""<<"P--BASE"<<"\", color=powderblue]";
					else if (opCode == GEP_OFFSET_OP )
						O<< "[label=\""<<"P__OFFSET"<<"\", color=powderblue]";
					else if (opCode == RESOLVED_OUTPUT_OP)
						O<< "[label=\""<<"R_OUTPUT"<<"\", color=green]";
					else if (opCode == RESOLVED_EXTERN_OP )
						O<< "[label=\""<<"R_EXTERN"<<"\", color=pink]";
					else if (opCode == RESOLVED_MALLOC_OP )
						O<< "[label=\""<<"R_MALLOC"<<"\", color=red]";
					else if (opCode == Instruction::Call )
					{
						int paramNum = MAX_PARAMS + 1;
						//cerr<<"Call from "<<sourceVP->name<<" to "<<targetVP->name<<endl;
						
						set<FuncCall *>::iterator fc_i = sourceVP->funcCalls.begin();
						
						for (; fc_i != sourceVP->funcCalls.end(); fc_i++)
						{
							FuncCall * fc = *fc_i;
							
							if (fc->funcName == targetVP->name)
							{
								paramNum = fc->paramNumber;
								break;
							}
							
							
							//cerr<<"     PN is "<<fc->paramNumber<<" for func "<<fc->funcName<<endl;
						}
						
						
						
						O<< "[color=red, label=\""<<Instruction::getOpcodeName(opCode)<<" "<<paramNum<<"\"]";
					}
					else
						O<< "[label=\""<<Instruction::getOpcodeName(opCode)<<"\"]";
	      }
				O<<" ;"<<endl;	
			}
		}
  }
  O<<"}";
	
}



/* Prints blame relationships to dot file
 printImplicit - also prints implicit relationship
 printInstType - also prints the type for each instruction 
 printLineNum - also prints linenum the instruction generated from
 */
void FunctionBFC::printToDotPretty(ostream &O, bool printImplicit, bool printInstType, 
															 bool printLineNum,int * opSet, int opSetSize)
{  
  property_map<MyGraphType, edge_iore_t>::type edge_type
	= get(edge_iore, G);
	
 printLineNum = false;	
	
  O<<"digraph G {"<<endl;
  graph_traits<MyGraphType>::vertex_iterator i, v_end;
  for(tie(i,v_end) = vertices(G); i != v_end; ++i) 
	{
	#ifdef DEBUG_OUTPUT		
		blame_info<<"Getting v for "<<*i<<endl;
#endif
		
		NodeProps * v = get(get(vertex_props, G),*i);
		
		if (!v)
		{
#ifdef DEBUG_ERROR				
			blame_info<<"V is NULL"<<endl;
			cerr<<"Null V in printToDot\n";
			#endif
			continue;
		}
		
		int v_index = v->number;
		int in_d = in_degree(v_index, G);
		int out_d = out_degree(v_index, G);
		
		

		
		if (in_d == 0 && out_d == 0)
		{

			bool shouldIContinue = true;
			
			if (v->isLocalVar)
				shouldIContinue = false;
		
			//if (!(v->isLocalVar))
				//continue;
			
			
			//if (isa<Instruction>(v->llvm_inst))
		//	{
			//	Instruction * pi = cast<Instruction>(v->llvm_inst);	
				//if (pi->getOpcode() == Instruction::Call)
					//shouldIContinue = false;
			//}	
			
			if (v->name.find("--") != string::npos)
				shouldIContinue = false;
			
			if (shouldIContinue)
				continue;
		}
		
		if (v->llvm_inst != NULL)
		{	
			if (isa<Instruction>(v->llvm_inst))
			{
				Instruction * pi = cast<Instruction>(v->llvm_inst);	
								
				const llvm::Type * t = pi->getType();
				unsigned typeVal = t->getTypeID();
				
				if (pi->getOpcode() == Instruction::Alloca)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					if (v->isLocalVar == true)
						O<<"\",shape=rectangle, style=filled, fillcolor=pink]\n";
					else if (v->isFormalArg)
						O<<"\",shape=invtriangle, style=filled, fillcolor=green]\n";
					else
						O<<"\",shape=Mdiamond, style=filled, fillcolor=darkseagreen]\n";
				}	
				//else if (pi->getOpcode() == Instruction::GetElementPtr)
				else if (v->pointsTo != NULL)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\",shape=Mdiamond, style=filled, fillcolor=yellowgreen]\n";
				}
				else if(pi->getOpcode() == Instruction::Call || pi->getOpcode() == Instruction::Invoke)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\",shape=tripleoctagon, style=filled, fillcolor=red]\n";		
				}
				else if (typeVal == Type::PointerTyID && v->name.find("Constant") == string::npos)
				{
					O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\",shape=Mdiamond, style=filled, fillcolor=yellow]\n";
				}
				else
				{
					string constNum;
                    size_t plusPos = v->name.find("+");
					if (plusPos != string::npos)
					{
						unsigned plusPos2 = v->name.find("+",plusPos+1);
						constNum = v->name.substr(plusPos+1, plusPos2-plusPos-1);
						
						//constNum.append(0,v->name.substr(plusPos, plusPos2-plusPos));
					}
				
					if (v->name.find("Constant") == string::npos)
						O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
					else
						O<<get(get(vertex_index, G),*i)<<"[label=\""<<"C: "<<constNum;
					if (printLineNum)
						O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
					O<<"\"];"<<"\n";
					
				}
			}
			else
			{
				//cerr<<"Not an instruction "<<v->name<<"\n";
				O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
				if (printLineNum)
					O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				O<<"\",shape=Mdiamond, style=filled, fillcolor=purple]\n";	
				
			}
		}
		else
		{
			O<<get(get(vertex_index, G),*i)<<"[label=\""<<v->name;
			if (printLineNum)
				O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
			O<<"\",shape=Mdiamond, style=filled, fillcolor=purple]\n";
		}
	}
	
  //  a -> b [label="hello", style=dashed];
  graph_traits<MyGraphType>::edge_iterator ei, edge_end;
  for(tie(ei, edge_end) = edges(G); ei != edge_end; ++ei) {
		
    int opCode = get(get(edge_iore, G),*ei);
		
    if (opSetSize) {
	  for (int a = 0; a < opSetSize; a++) {
		if (opSet[a] == opCode)
	    {
		  O<< get(get(vertex_index, G), source(*ei, G)) << "->" << get(get(vertex_index, G), target(*ei, G));
					
		  if ( opCode && printInstType) {
			if (opCode != ALIAS_OP)
				O<< "[label=\""<<Instruction::getOpcodeName(get(get(edge_iore, G),*ei))<<"\"]";
			else
			    O<< "[label=\""<<"ALIAS"<<"\"]";
					
          }
		  O<<" ;"<<endl;	
	    }
	  }
	}
		
    else
		{
			if ( opCode || printImplicit )	
			{
				int sourceV = get(get(vertex_index, G), source(*ei, G));
				int targetV = get(get(vertex_index, G), target(*ei, G));
				
				
				NodeProps * sourceVP = get(get(vertex_props, G), source(*ei, G));
				NodeProps * targetVP = get(get(vertex_props, G), target(*ei, G));
				
				/*
				 if (opCode == Instruction::Store)
				 {
				 NodeProps *v = get(get(vertex_props, G), source(*ei,G));
				 if (!v)
				 {
				 cerr<<"Null V(2) in printToDot\n";
				 continue;
				 }
				 
				 O<<sourceV<<"[label=\""<<v->name;
				 if (printLineNum)
				 O<<":("<<v->line_num<<":"<<v->lineNumOrder<<")";
				 O<<"\",shape=Mdiamond]\n";
				 }
				 */
				
				O<< sourceV  << "->" << targetV;
				
				if (!opCode && printImplicit)
					O<< "[color=grey, style=dashed]";
				else if ( opCode && printInstType)
	      {
					if (opCode == ALIAS_OP)
						O<< "[label=\""<<"ALIAS"<<"\"]";
					else if (opCode == GEP_BASE_OP )
						O<< "[label=\""<<"BASE"<<"\", color=powderblue]";
					else if (opCode == GEP_OFFSET_OP )
						O<< "[label=\""<<"OFFSET"<<"\", color=powderblue]";
					else if (opCode == GEP_S_FIELD_VAR_OFF_OP)
						O<< "[label=\""<<"GEP--FIELD--OFFSET"<<"\", color=powderblue]";
					else if (opCode >= GEP_S_FIELD_OFFSET_OP)
						O<< "[label=\""<<"FIELD"<<"\", color=powderblue]";
					else if (opCode == RESOLVED_OUTPUT_OP)
						O<< "[label=\""<<"R_OUTPUT"<<"\", color=green]";
					else if (opCode == RESOLVED_EXTERN_OP )
						O<< "[label=\""<<"R_EXTERN"<<"\", color=pink]";
					else if (opCode == RESOLVED_MALLOC_OP )
						O<< "[label=\""<<"R_MALLOC"<<"\", color=red]";
					else if (opCode == Instruction::Call )
					{
						int paramNum = MAX_PARAMS + 1;
						//cerr<<"Call from "<<sourceVP->name<<" to "<<targetVP->name<<endl;
						
						set<FuncCall *>::iterator fc_i = sourceVP->funcCalls.begin();
						
						for (; fc_i != sourceVP->funcCalls.end(); fc_i++)
						{
							FuncCall * fc = *fc_i;
							
							if (fc->funcName == targetVP->name)
							{
								paramNum = fc->paramNumber;
								break;
							}
							
							
							//cerr<<"     PN is "<<fc->paramNumber<<" for func "<<fc->funcName<<endl;
						}
						
						
						
						O<< "[color=red, label=\""<<Instruction::getOpcodeName(opCode)<<" "<<paramNum<<"\"]";
					}
					else
					{
						O<< "[label=\""<<Instruction::getOpcodeName(opCode)<<"\"]";
						//O<< "[label=\""<<Instruction::getOpcodeName(opCode)<<" "<<opCode<<"\"]";

					}
	      }
				O<<" ;"<<endl;	
			}
		}
  }
  O<<"}";
	
}

/* 
 This function is for debugging purposes and prints out for each function
 - Name
 - Return Type 
 - Parameters
 - Name
 - Return Type
 */
void FunctionBFC::printFunctionDetails(ostream & O)
{
  O<<"Function "<<func->getName().data()<<" has return type of "<<returnTypeName(func->getReturnType(), string(" "))<<"\n";
  O<<"Parameters: \n";
	
  for(Function::arg_iterator af_i = func->arg_begin(); af_i != func->arg_end(); af_i++)
	{
		Argument *v = dyn_cast<Argument>(af_i);
		
		if (v->hasName())
			O<<v->getName().str()<<" of type " << returnTypeName(v->getType(), string(" "))<< "\n";
	}
	
}
