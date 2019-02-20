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

#ifndef _BFC_H
#define _BFC_H

#include "llvm/Analysis/Passes.h"
#include "llvm/ADT/Statistic.h"
//#include "llvm/Assembly/Writer.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

namespace {

	class BFC : public ModulePass {
	
	private:
		DebugInfoFinder Finder;
		
	public:
		static char ID; // Pass identification, replacement for typeid
		BFC() : ModulePass(ID) {}
		
        //virtual bool runOnModule(Module & M);
        bool runOnModule(Module & M) override;
		//virtual void getAnalysisUsage(AnalysisUsage &AU) const {
		//	AU.setPreservesAll();
		//}
		void getAnalysisUsage(AnalysisUsage &AU) const override {
			AU.setPreservesAll();
		}
		//virtual void print(raw_ostream &O, const Module *M) const {}
		void print(raw_ostream &O, const Module *M) const override {}
		
	};
}

char BFC::ID = 0;
static RegisterPass<BFC> X("bfc", "Calculates Blame for cuda kernels at LLVM level", false, true);
#endif
