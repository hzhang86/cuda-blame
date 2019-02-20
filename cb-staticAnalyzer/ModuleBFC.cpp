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

//#include "FunctionBFC.h"
#include "ModuleBFC.h"

#include <iostream>
#include <fstream>

using namespace std;

// to be used in pidArrayResolve
extern ofstream struct_file;


void ModuleBFC::exportOneStruct(ostream &O, StructBFC *sb)
{
    O<<"BEGIN STRUCT"<<endl;
    
    O<<"BEGIN S_NAME "<<endl;
    O<<sb->structName<<endl;
    O<<"END S_NAME "<<endl;
    
    O<<"BEGIN M_PATH"<<endl;
    O<<sb->modulePathName<<endl;
    O<<"END M_PATH"<<endl;
    
    O<<"BEGIN M_NAME"<<endl;
    O<<sb->moduleName<<endl;
    O<<"END M_NAME"<<endl;
    
    O<<"BEGIN LINENUM"<<endl;
    O<<sb->lineNum<<endl;
    O<<"END LINENUM"<<endl;
    
    O<<"BEGIN FIELDS"<<endl;
    vector<StructField *>::iterator vec_sf_i;
    for (vec_sf_i = sb->fields.begin(); vec_sf_i != sb->fields.end(); vec_sf_i++) {
        O<<"BEGIN FIELD"<<endl;
        StructField * sf = (*vec_sf_i);
        if (sf == NULL) {
            O<<"END FIELD"<<endl;
            continue;
        }
        O<<"BEGIN F_NUM"<<endl;
        O<<sf->fieldNum<<endl;
        O<<"END F_NUM"<<endl;
        
        O<<"BEGIN F_NAME"<<endl;
        O<<sf->fieldName<<endl;
        O<<"END F_NAME"<<endl;
        
        O<<"BEGIN F_TYPE"<<endl;
        O<<sf->typeName<<endl;
        O<<"END F_TYPE"<<endl;
        
        O<<"END FIELD"<<endl;
    }
    
    O<<"END FIELDS"<<endl;
    
    //end of this new added struct
    O<<"END STRUCT"<<endl;
}


void ModuleBFC::exportStructs(ostream &O)
{
	StructBFCHash::iterator sbh_i;
	
	for (sbh_i = structs.begin(); sbh_i != structs.end(); sbh_i++) {
		StructBFC * sb = (*sbh_i).second;
		O<<"BEGIN STRUCT"<<endl;
		
		O<<"BEGIN S_NAME "<<endl;
		O<<sb->structName<<endl;
		O<<"END S_NAME "<<endl;
		
		O<<"BEGIN M_PATH"<<endl;
		O<<sb->modulePathName<<endl;
		O<<"END M_PATH"<<endl;
		
		O<<"BEGIN M_NAME"<<endl;
		O<<sb->moduleName<<endl;
		O<<"END M_NAME"<<endl;
		
		O<<"BEGIN LINENUM"<<endl;
		O<<sb->lineNum<<endl;
		O<<"END LINENUM"<<endl;
		
		O<<"BEGIN FIELDS"<<endl;
		vector<StructField *>::iterator vec_sf_i;
		for (vec_sf_i = sb->fields.begin(); vec_sf_i != sb->fields.end(); vec_sf_i++) {
			O<<"BEGIN FIELD"<<endl;
			StructField * sf = (*vec_sf_i);
			if (sf == NULL) {
				O<<"END FIELD"<<endl;
				continue;
			}
			O<<"BEGIN F_NUM"<<endl;
			O<<sf->fieldNum<<endl;
			O<<"END F_NUM"<<endl;
			
			O<<"BEGIN F_NAME"<<endl;
			O<<sf->fieldName<<endl;
			O<<"END F_NAME"<<endl;
			
			O<<"BEGIN F_TYPE"<<endl;
			O<<sf->typeName<<endl;
			O<<"END F_TYPE"<<endl;
			
			O<<"END FIELD"<<endl;
		}
		
		O<<"END FIELDS"<<endl;
		
		O<<"END STRUCT"<<endl;
	}
	
    //we move the following MARK to the end of runOnModule since we may addin more
    //when firstPassing each user functions (pidArrays)
	//O<<"END STRUCTS"<<endl;
}


string returnTypeName(const llvm::Type * t, string prefix)
{
	if (t == NULL)
		return prefix += string("NULL");
	
    unsigned typeVal = t->getTypeID();
    if (typeVal == Type::VoidTyID)
        return prefix += string("Void");
    else if (typeVal == Type::FloatTyID)
        return prefix += string("Float");
    else if (typeVal == Type::DoubleTyID)
        return prefix += string("Double");
    else if (typeVal == Type::X86_FP80TyID)
        return prefix += string("80 bit FP");
    else if (typeVal == Type::FP128TyID)
        return prefix += string("128 bit FP");
    else if (typeVal == Type::PPC_FP128TyID)
        return prefix += string("2-64 bit FP");
    else if (typeVal == Type::LabelTyID)
        return prefix += string("Label");
    else if (typeVal == Type::IntegerTyID)
        return prefix += string("Int");
    else if (typeVal == Type::FunctionTyID)
        return prefix += string("Function");
    else if (typeVal == Type::StructTyID)
        return prefix += string("Struct");
    else if (typeVal == Type::ArrayTyID)
        return prefix += string("Array");
    else if (typeVal == Type::PointerTyID)
        return prefix += returnTypeName(cast<PointerType>(t)->getElementType(),
			string("*"));
    else if (typeVal == Type::MetadataTyID)
        return prefix += string("Metadata");
    else if (typeVal == Type::VectorTyID)
        return prefix += string("Vector");
    else
        return prefix += string("UNKNOWN");
}


void ModuleBFC::printStructs()
{
	StructBFCHash::iterator sbh_i;
	
	for (sbh_i = structs.begin(); sbh_i != structs.end(); sbh_i++)	{
		cout<<endl<<endl;
		StructBFC *sb = (*sbh_i).second;
		cout<<"Struct "<<sb->structName<<endl;
		cout<<"Module Path "<<sb->modulePathName<<endl;
		cout<<"Module Name "<<sb->moduleName<<endl;
		cout<<"Line Number "<<sb->lineNum<<endl;
		
		vector<StructField *>::iterator vec_sf_i;
		for (vec_sf_i = sb->fields.begin(); vec_sf_i != sb->fields.end(); vec_sf_i++) {
			StructField * sf = (*vec_sf_i);
			if (sf == NULL)
				continue;
			cout<<"   Field # "<<sf->fieldNum<<" ";
			cout<<", Name "<<sf->fieldName<<" ";
			cout<<" Type "<<sf->typeName<<" ";
			//cout<<", Type "<<returnTypeName(sf->llvmType, string(" "));
			cout<<endl;
		}
	}
}


StructBFC *ModuleBFC::structLookUp(string &sName)
{	
  if (structs.find(sName) != structs.end())
	  return structs[sName];

  return NULL;
}


void ModuleBFC::addStructBFC(StructBFC * sb)
{
	//cout<<"Entering addStructBFC"<<endl;
	structs[sb->structName] = sb;
    exportOneStruct(struct_file, sb); //added 03/31/17
}


StructBFC* ModuleBFC::findOrCreatePidArray(string pidArrayName, int numElems, const llvm::Type *sbPointT)
{
    StructBFC *retSB = NULL;
    retSB = structLookUp(pidArrayName);
    if (retSB != NULL)
      return retSB;
    
    else {
      retSB = new StructBFC();
      retSB->structName = pidArrayName;
      //we leave context information blank since we dont know & we dont need them
      //That includes: lineNum, moduleName, and modulePathName
      // This should be true
      if (sbPointT->isArrayTy()) {
        const llvm::Type *pidType = cast<ArrayType>(sbPointT)->getElementType();
        for (int i=0; i<numElems; i++) {
          StructField *sf = new StructField(i); //fieldNum
          char tempBuf[20];
          sprintf(tempBuf, "pid_x%d", i);

          sf->fieldName = string(tempBuf); //fieldName
          sf->llvmType = pidType; //field->llvmType
          sf->typeName = returnTypeName(sf->llvmType, ""); // field->typeName
          sf->parentStruct = retSB;
          // add this field to the struct
          retSB->fields.push_back(sf);
        }
      }
      else {
        cerr<<"Error, non pid array apears in findOrCreatePidArray:"<<
            pidArrayName<<endl;
        delete retSB; 
        return NULL;
      }
    }
    
    addStructBFC(retSB);
    return retSB;
}

    


void StructBFC::setModuleNameAndPath(llvm::DIScope *contextInfo)
{
	if (contextInfo == NULL)
		return;

	this->moduleName = contextInfo->getFilename().str();
    this->modulePathName = contextInfo->getDirectory().str();
}


void ModuleBFC::parseDITypes(DIType* dt) //deal with derived type and composite type
{
#ifdef DEBUG_P    //stands for debug_print
    cout<<"Entering parseDITypes for "<<dt->getName().str()<<endl;
#endif	
    StructBFC *sb;
    string dtName = dt->getName().str();
    if (isa<DICompositeType>(dt)) { //DICompositeType:public DIDerivedType
#ifdef DEBUG_P
      cout<<"Composite Type primary for: "<<dt->getName().str()<<endl;
#endif		
      sb = new StructBFC();
      if (parseCompositeType(dt, sb, true))
        addStructBFC(sb);
      else
        delete sb;
    }	
        
    else if (isa<DIDerivedType>(dt)) {
      DIDerivedType *ddt = cast<DIDerivedType>(dt);
      if (ddt->getTag() != dwarf::DW_TAG_member) { //or ddt->Tag
#ifdef DEBUG_P
        cout<<"Derived Type: "<<dt->getName().str()<<endl;
#endif		
        sb = new StructBFC();
        if (parseDerivedType(dt, sb, NULL, false))
          addStructBFC(sb);
        else
          delete sb;
      }     
    }	
}



/*
!6 = metadata !{
   i32,      ;; Tag (see below)
   metadata, ;; Reference to context
   metadata, ;; Name (may be "" for anonymous types)
   metadata, ;; Reference to file where defined (may be NULL)
   i32,      ;; Line number where defined (may be 0)
   i64,      ;; Size in bits
   i64,      ;; Alignment in bits
   i64,      ;; Offset in bits
   i32,      ;; Flags
   metadata, ;; Reference to type derived from
   metadata, ;; Reference to array of member descriptors
   i32       ;; Runtime languages
}*/
bool ModuleBFC::parseCompositeType(DIType *dt, StructBFC *sb, bool isPrimary)
{
    if (sb == NULL)
		return false;
    else if (structLookUp(sb->structName)) //this compositeType has been processed
        return false; 
	
	if (dt == NULL) 
		return false;
    else if (!isa<DICompositeType>(dt)) {//verify that dt is a composite type
#ifdef DEBUG_P
      cout<<"parseCompositeType fail: "<<dt->getName().str()<<" is not a compositeType"<<endl;
#endif
      return false; 
    }
    else if (dt->getFlagString(dt->getFlags()).equals(StringRef("DIFlagFwdDecl"))) {//verify that dt is a composite type
#ifdef DEBUG_P
      cout<<"parseCompositeType fail: "<<dt->getName().str()<<" is a foward declaration"<<endl;
#endif
      return false; 
    }
    
    else {
      // There are no other typedefs that would alias to this
	  if (isPrimary) {
		sb->lineNum = dt->getLine();
		sb->structName = dt->getName().str();//dt->getStringField(2).str();
        sb->setModuleNameAndPath(dt);
	  }
      //DICompositeType dct = dyn_cast<DICompositeType>(dt);
      DICompositeType *dct = cast<DICompositeType>(dt);
      DINodeArray fields = dct->getElements(); // !3 = !{!1,!2}
                                                  //return the mdnode for members
      int numFields = fields.size();
      if (numFields == 0) {
#ifdef DEBUG_P
          cout<<"parseCompositeType fail: "<<dt->getName().str()<<" has 0 fields"<<endl;
#endif
          return false; 
      }
      
      //else omit the field info
      for (int i = 0; i < numFields; i++) {
        DINode *field = fields[i];      
        if (!isa<DIType>(field)) {
#ifdef DEBUG_P
          cout<<"parseCompositeType fail: "<<dt->getName().str()<<" field#"<<i<<" isn't a DIType"<<endl;
#endif
          return false; 
        }       

        DIType *dtField = cast<DIType>(field);
        StructField *sf = new StructField(i);
        bool success;
        success = parseDerivedType(dtField, NULL, sf, true);
        if (success) {
          sb->fields.push_back(sf);
          sf->parentStruct = sb;
        }
        else {
#ifdef DEBUG_P
          cout<<"parseCompositeType fail: "<<dt->getName().str()<<" call parseDerivedType on field#"<<i<<" returns false"<<endl;
#endif
          delete sf;
          return false;
        }
      } //for loop
#ifdef DEBUG_P
      cout<<"parseCompositeType SUCCEED @ "<<dt->getName().str()<<endl;
#endif
      return true;
	}
#ifdef DEBUG_P
    cout<<"Why are we here??? parseCompositeType "<<dt->getName().str()<<endl;
#endif
	return false;
}





/*
!5 = metadata !{
  i32,      ;; Tag (see below)
  metadata, ;; Reference to context
  metadata, ;; Name (may be "" for anonymous types) //
  metadata, ;; Reference to file where defined (may be NULL)
  i32,      ;; Line number where defined (may be 0)
  i64,      ;; Size in bits
  i64,      ;; Alignment in bits
  i64,      ;; Offset in bits
  i32,      ;; Flags to encode attributes, e.g. private
  metadata, ;; Reference to type derived from
  metadata, ;; (optional) Name of the Objective C property associated with
            ;; Objective-C an ivar, or the type of which this
            ;; pointer-to-member is pointing to members of.
  metadata, ;; (optional) Name of the Objective C property getter selector.
  metadata, ;; (optional) Name of the Objective C property setter selector.
  i32       ;; (optional) Objective C property attributes.
}
 */
//void ModuleBFC::parseDerivedType(GlobalVariable *gv)
bool ModuleBFC::parseDerivedType(DIType *dt, StructBFC *sb, StructField *sf, bool isField)
{    
	if (dt == NULL) // both sf and sb can be NULL here
	  return false;
    else if (!isa<DIDerivedType>(dt)) {//verify that dt is a Derived type
#ifdef DEBUG_P
      cout<<"parseDerivedType fail: "<<dt->getName().str()<<" isn't DIDerivedType"<<endl;
#endif
      return false; 
    }
    else {
      DIDerivedType *ddt = cast<DIDerivedType>(dt);
	  DIType *derivedFromType = ddt->getBaseType().resolve(); 
		
      if (isField == false) {
        if (isa<DICompositeType>(derivedFromType)) { //only care about 
                                                  //direct derived composite
          DICompositeType *dfct = cast<DICompositeType>(derivedFromType); //derivedFromCompositeType -> dfct
          //TODO:replace the below func with easier method//
          //DIScope dfct_scope = dfct->getContext();
          //sb->setModuleNameAndPath(&dfct_scope);
          return parseCompositeType(dfct, sb, true);//not see a case for isPrimary=false yet
        }
        else {//we don't deal with other derived type for now, like typedef...
#ifdef DEBUG_P
          cout<<"We do not deal with derived types other than whose basetype is a struct right now"<<endl;
#endif
          return false;
        }
      }
      else {
        sf->fieldName = dt->getName().str();
        sf->typeName = derivedFromType->getName().str();//getStringField(2).str();
#ifdef DEBUG_P
        cout<<"parseDerivedType (as field) SUCCEED @name: "<<sf->fieldName \
            <<" fieldType: "<<sf->typeName<<endl;
#endif
        return true;
      }
    }
}



