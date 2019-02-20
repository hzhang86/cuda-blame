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

package blame;

import java.util.Vector;
import java.util.HashMap;

public class StackFrame {

	int frameNum;
	String moduleName;
	String modulePath;
	int lineNumber;
	
	Instance parentInst;
	BlameFunction bf;

	
	HashMap<String, ExitSuper> rawVariables;
	HashMap<String, ExitSuper> variables;
	Vector<VariableInstance> varInstances;
	
	static boolean DO_OVERRIDE_PATH = false;
	//static String OVERRIDE_PATH = "/Users/nickrutar/UnixStuff/BLAME/TEST-PROGRAMS/HPL/OLD/hpl-openmpi-2-llvm/";
	static String OVERRIDE_PATH = "/Users/nickrutar/UnixStuff/BLAME/TEST-PROGRAMS/HPL/";

	
	
	StackFrame(int frameNum, String moduleName, String pathName,int lineNumber, BlameFunction bf)
	{
		this.frameNum = frameNum;
		this.moduleName = moduleName;
		this.modulePath = pathName;
		this.lineNumber = lineNumber;
		this.bf = bf;
		
		varInstances = new Vector<VariableInstance>();
		
		rawVariables = new HashMap<String, ExitSuper>();
		variables = new HashMap<String, ExitSuper>();

		
		if (DO_OVERRIDE_PATH)
		{
			if (Global.testProgram.equals(new String("HPL")))
			{
				String tail = modulePath.substring(modulePath.indexOf("hpl-2.0"));
				String trimTail = tail.substring(0, tail.indexOf("LLVM")-1);
				modulePath = OVERRIDE_PATH + trimTail;
			}
			else if (Global.testProgram.equals(new String("SMALL")))
			{
				modulePath = "/Users/nickrutar/UnixStuff/BLAME/SMALLEXAMPLE";
			}
		}
	}
	
	/*
	public void transferFields(ExitSuper callee, ExitSuper caller, VariableInstance vi)
	{
		System.out.println(callee.getHierName() + " has " + callee.getFields().size() + " fields.");
		if (callee.getFields().size() == 0)
			return;
		
		Iterator it = callee.getFields().values().iterator();
		while (it.hasNext())
		{
			ExitSuper es = (ExitSuper) it.next();
			System.out.println("Looking at field " + es.getName());
			
			ExitSuper gocReturn = caller.getOrCreateField(es, vi);
			if (es != gocReturn)
				transferFields(es, gocReturn, vi);
		}
		
	}
	*/
	
	
	/*
	public void resolveFrameES()
	{
		System.out.println("In frame number " + frameNum + " for instance " + parentInst.getInstanceNum());
		
		Iterator<ExitSuper> it = rawVariables.values().iterator();
		while (it.hasNext())
		{
			ExitSuper es = (ExitSuper) it.next();
			
			System.out.println("In resolveFrameES, looking at " + es.getHierName());
			// We have the field, now we need to find its parent
			if (es.isField())
			{
				//System.out.println("Parent for " + es.getHierName() + " is " + es.getParentName());
				ExitSuper parentField = rawVariables.get(es.getParentName());
				if (parentField != null)
				{
					parentField.addField(es);
					System.out.println("In resolveFrameES, adding field " + es.getHierName() + " for " + parentField.getHierName());
				}
				else
					System.out.println("In resolveFrameES, couldn't find " + es.getParentName());
				
			}		
		}
		
		StackFrame calleeStack = parentInst.getStackFrame(frameNum - 1 );
		if (calleeStack == null)
			return;
		
		it = rawVariables.values().iterator();
		while (it.hasNext())
		{
			ExitSuper es = (ExitSuper) it.next();
			System.out.println("ES - " + es.getHierName());
			
			if (es.getStructType() == null)
				continue;
				
			Iterator<ExitSuper> calleeit = calleeStack.getRawVariables().values().iterator();
			while (calleeit.hasNext())
			{
				ExitSuper calleeEs = (ExitSuper) calleeit.next();
				System.out.println("CES - " + es.getHierName());
				
				if (calleeEs.getStructType() == null)
					continue;
				
				System.out.println("ES - " + es.getHierName() + "ST - " + es.getStructType());
				System.out.println("CES - " + calleeEs.getHierName() + "CST - " + es.getStructType());
				
				if (es.structType.compareTo(calleeEs.structType) == 0)
				{
					
					
					System.out.println("Match between " + es.getHierName() + " and " + calleeEs.getHierName());
					
					Iterator vi_it =  calleeStack.getVarInstances().iterator();
					while (vi_it.hasNext())
					{
						VariableInstance vi = (VariableInstance)vi_it.next();
						if (vi.getVar().getHierName().compareTo(calleeEs.getHierName()) == 0)
								//&& vi.getCallerParam() == es.getCalleeParam() )
						{
							System.out.println("Match(2) between " + es.getHierName() + " and " + calleeEs.getHierName());
							transferFields(calleeEs, es, vi);
						}
						
					}
					
					
					
				}
			}
		}
	}

	*/
	
	public Instance getParentInst() {
		return parentInst;
	}


	public void setParentInst(Instance parentInst) {
		this.parentInst = parentInst;
	}

/*
	public HashMap<String, ExitSuper> getRawVariables() {
		return rawVariables;
	}


	public void setRawVariables(HashMap<String, ExitSuper> rawVariables) {
		this.rawVariables = rawVariables;
	}
	*/


	public HashMap<String, ExitSuper> getVariables() {
		return variables;
	}


	public void setVariables(HashMap<String, ExitSuper> variables) {
		this.variables = variables;
	}

	
	
	public void addRawEV(ExitSuper ev)
	{
		String varName = ev.getHierName();
		
		System.out.println("Adding raw EV for " + varName);
		
		if (varName.indexOf('-') > 0)
			varName = varName.substring(0, varName.indexOf('-'));
		
		System.out.println("New var name is " + varName );
		
		if (!rawVariables.containsKey(varName))
			rawVariables.put(varName, ev);
	}
	
	public void addVarInstance(VariableInstance vi)
	{
		varInstances.add(vi);
	}
	
	public int getFrameNum() {
		return frameNum;
	}

	public void setFrameNum(int frameNum) {
		this.frameNum = frameNum;
	}

	public String getModuleName() {
		return moduleName;
	}

	public void setModuleName(String moduleName) {
		this.moduleName = moduleName;
	}

	public String getModulePath() {
		return modulePath;
	}

	public void setModulePath(String modulePath) {
		this.modulePath = modulePath;
	}

	public int getLineNumber() {
		return lineNumber;
	}

	public void setLineNumber(int lineNumber) {
		this.lineNumber = lineNumber;
	}

	public Vector<VariableInstance> getVarInstances() {
		return varInstances;
	}

	public void setVarInstances(Vector<VariableInstance> varInstances) {
		this.varInstances = varInstances;
	}


	
	
	
	void addBlameFunction(BlameFunction bf)
	{
		this.bf = bf;
	}
	
	public void print()
	{
		System.out.println(frameNum + "---" + modulePath + "/" + moduleName + ":" + lineNumber);
	}
	
	public String toString()
	{
		//String s = new String(frameNum + "---" + modulePath + "/" + moduleName + ":" + lineNumber);
		String s = new String(frameNum + "---" + moduleName + ":" + lineNumber);
		return s;
	}

	public BlameFunction getBf() {
		return bf;
	}

	public void setBf(BlameFunction bf) {
		this.bf = bf;
	}
}
