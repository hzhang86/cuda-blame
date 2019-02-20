package blame;

import java.io.BufferedReader;
import java.util.HashMap;
import java.util.Iterator;
import java.io.IOException;

public class BlameFunction implements Comparable {

	//private boolean isBlamePoint, isGraphParsed, isUDBlamePoint;
	
	public static final int NO_BLAME_POINT = 0;
	public static final int EXPLICIT_BLAME_POINT = 1;
	public static final int IMPLICIT_BLAME_POINT = 2;
	
	private short blamePointType;
	private String moduleName;
	private String name;
	private int beginNum;
	private int endNum;
	
	private HashMap<String, ExitVariable> exitVariables;
	private HashMap<String, ExitProgram> exitPrograms;
	private HashMap<String, ExitOutput> exitOutputs;
	
	private HashMap<String, ExitVariable> exitVarFields;
	private HashMap<String, ExitProgram>  exitProgFields;

	
	 public int compareTo(Object obj) 
	 {
		 BlameFunction comparor = (BlameFunction) obj;
		 double cMBV = comparor.mostBlamedVariable();
		 double mbv = mostBlamedVariable();
		 
		 if (mbv == cMBV)
			 return 0;
		 else if (mbv > cMBV)
			 return 1;
		 else 
			 return -1;
	}
	 
	public double mostBlamedVariable()
	{
		double maxBlame = 0.0;
		
		Iterator<ExitVariable> it_ev = getExitVariables().values().iterator();
		while (it_ev.hasNext())
		{
			ExitVariable ev = it_ev.next();
			double d = ev.percentageBlame();
			if (d > maxBlame)
				maxBlame = d;
		}
		
		Iterator<ExitVariable> it_evf = getExitVarFields().values().iterator();
		while (it_evf.hasNext())
		{
			ExitVariable ev = it_evf.next();
			double d = ev.percentageBlame();
			if (d > maxBlame)
				maxBlame = d;
			
		}
		
		
		Iterator<ExitProgram> it_lv = getExitPrograms().values().iterator();
		while (it_lv.hasNext())
		{
			ExitProgram ev = it_lv.next();
			double d = ev.percentageBlame();
			if (d > maxBlame)
				maxBlame = d;
			
		}
		
		
		Iterator<ExitProgram> it_lvf =getExitProgFields().values().iterator();
		while (it_lvf.hasNext())
		{
			ExitProgram ev = it_lvf.next();
			double d = ev.percentageBlame();
			if (d > maxBlame)
				maxBlame = d;
			
		}
		
		
		
		return maxBlame;
	}
	
	
	BlameFunction(String n, String mN, short bPT)
	{
		name = n;
		moduleName = mN;
		blamePointType = bPT;
		
		exitVariables = new HashMap<String, ExitVariable>(); 
		exitPrograms = new HashMap<String, ExitProgram>();
		exitOutputs = new HashMap<String, ExitOutput>();
		
		exitVarFields = new HashMap<String, ExitVariable>();
		exitProgFields = new HashMap<String, ExitProgram>();
		
	}
	
	BlameFunction(String n, String mN, int bNum, int eNum, int isBlamePoint, int isGraphParsed, boolean isUDBP)
	{
		name = n;
		moduleName = mN;
		
		beginNum = bNum;
		endNum = eNum;
		
		exitVariables = new HashMap<String, ExitVariable>(); 
		exitPrograms = new HashMap<String, ExitProgram>();
		exitOutputs = new HashMap<String, ExitOutput>();
		
		exitVarFields = new HashMap<String, ExitVariable>();
		exitProgFields = new HashMap<String, ExitProgram>();
	}
	
	 void parseBlamedFunction(BufferedReader bufReader)
	 { 
		 String line;
		 try 
		 {
			 // BEGIN F_B_LINE_NUM
			 bufReader.readLine();
			 
			 String strFBLN = bufReader.readLine();
			 beginNum = Integer.valueOf(strFBLN).intValue();
			 
			 // END F_B_LINE_NUM
			 bufReader.readLine();
			//-----
			 // BEGIN F_E_LINE_NUM
			 bufReader.readLine();
			 
			 String strEBLN = bufReader.readLine();
			 endNum = Integer.valueOf(strEBLN).intValue();
			 
			 // END F_E_LINE_NUM
			 bufReader.readLine();
		    //-----
			 //F_BPOINT
			 bufReader.readLine();  bufReader.readLine();  bufReader.readLine();
			 
			 boolean proceed = true;
			 while (proceed)
			 {    
				 line = bufReader.readLine();
				// System.out.println("L1 - " + line);
				 if (line.indexOf("BEGIN VAR") >= 0)
				 {
					 //BEGIN V_NAME
					 bufReader.readLine();
					 // Get Variable Name
					 String varName = bufReader.readLine();
					 // END V_NAME
					 bufReader.readLine();
					//---
					 //BEGIN V_TYPE
					 bufReader.readLine();
					 // Get Variable E Type
					 String strEType = bufReader.readLine();
					 int eStatus = Integer.valueOf(strEType).intValue();
					 // END V_TYPE
					 bufReader.readLine();
					 
					 // BEGIN N_TYPE
					 bufReader.readLine();
					 // Get Variable N Type
					 String strNType = bufReader.readLine();
					 String nTokens[] = strNType.split("\\s");
					 int isLocalVar   = Integer.valueOf(nTokens[ExitSuper.LOCAL_VAR]);
					 int isLocalField = Integer.valueOf(nTokens[ExitSuper.LOCAL_VAR_FIELD]);
					 int isExitVField = Integer.valueOf(nTokens[ExitSuper.EXIT_VAR_FIELD]);
					 
					 // END N_TYPE
					 bufReader.readLine();
					 
					 int totalCheck = eStatus + isLocalVar + isLocalField + isExitVField;
					 
					 //System.out.println("TotalCheck Val: " + totalCheck);
					 
					 // Nothing we are interested in
					 if (totalCheck == 0)
					 {
						while ((line = bufReader.readLine()) != null)
						{
							//System.out.println("Variable not interesting(1)");
							//System.err.println("Variable not interesting(1)");
							if (line.indexOf("END VAR") >= 0)
								break;
						} 
					 }
					 // Some aspect of this variable is something we want
					 else
					 {
						 ExitSuper es = new ExitSuper();
						 
						 if (eStatus == ExitSuper.EXIT_VAR_GLOBAL)
						 {
							 es = this.getOrCreateEV(varName);
						 }
						 else if (eStatus == ExitSuper.EXIT_VAR_RETURN)
						 {
							es = this.getOrCreateEV(varName); 
						 }
						 else if (eStatus >= ExitSuper.EXIT_VAR_PARAM)
						 {
							 es = this.getOrCreateEV(varName);
							 es.setParentBF(this);
							 //es.transferVI();
						 }
						 else if (eStatus == ExitSuper.EXIT_OUTP)
						 {
							 es = this.getOrCreateEO(varName);
						 }
						 else if (isLocalVar > 0)
						 {
							 es = this.getOrCreateEP(varName);
						 }
						 else if (isExitVField > 0)
						 {
							 es = this.getOrCreateEVField(varName);
						 }
						 else if (isLocalField > 0)
						 {
							es = this.getOrCreateEPField(varName);
						 }
						 else
						 {
							 System.err.print(totalCheck + " " + eStatus + " " + isLocalVar);
							 System.err.print(" " + isLocalField + " " + isExitVField + " ");
							 System.err.println("WHY ARE WE HERE?!?");
							 
							 System.out.print(totalCheck + " " + eStatus + " " + isLocalVar);
							 System.out.print(" " + isLocalField + " " + isExitVField + " ");
							 System.out.println("WHY ARE WE HERE?!?");
							 //System.exit(-1);
							while ((line = bufReader.readLine()) != null)
							{
								if (line.indexOf("END VAR") >= 0)
									break;
							} 
							es = null;
						 }
						 
						 if (es != null)
						 {
							 
						 System.out.println("Filling in variable " + es.getName());
						 
						 es.setParentBF(this);
						 
						 
						 // BEGIN DECLARED_LINE 
						 bufReader.readLine();
						 String strDeclaredLine = bufReader.readLine();
						 System.out.println("Declared Line Str - " + strDeclaredLine);
						 int dL = Integer.valueOf(strDeclaredLine).intValue();
						 es.setDeclaredLine(dL);
						 // END DECLARED_LINE
						 bufReader.readLine();
						 
						 
						while ((line = bufReader.readLine()) != null)
						{
							if (line.indexOf("BEGIN FIELDS") >= 0)
								break;
						} 
						
						while ((line = bufReader.readLine()) != null)
						{
							if (line.indexOf("END FIELDS ") >= 0)
								break;
							else
							{
								System.out.println("Adding field " + line + " for " + es.getName());
								es.addField(line, this);
							}
						}
						
						
						// BEGIN GENTYPE
						bufReader.readLine();
						String genType = bufReader.readLine();
						es.setGenType(genType);
						// END GENTYPE
						bufReader.readLine();
						
						
						// BEGIN STRUCTTYPE
						bufReader.readLine();
						String structType = bufReader.readLine();
						es.setStructType(structType);
						// END STRUCTTYPE
						bufReader.readLine();
						
						while ((line = bufReader.readLine()) != null)
						{
							if (line.indexOf("BEGIN LINENUMS") >= 0)
								break;
						} 
						
						while ((line = bufReader.readLine()) != null)
						{
							if (line.indexOf("END LINENUMS ") >= 0)
								break;
							else
							{
								//System.out.println("Should be a line - " + line);
								es.addLine(Integer.valueOf(line).intValue());
								//es.addField(line, this);
							}
						}
						
						// END VAR
						bufReader.readLine();
						}
						
					 }
				 }
				 else
				 {
					 proceed = false;
				 }
			 } // end while proceed
			 
		 } // end try 
		 catch(IOException ie)
		 {
			 ie.printStackTrace();
		 }	 
	 }
	
	 
	public void addEP(String key, ExitProgram ep)
	{
		exitProgFields.put(key, ep);
	}
	
	public void addEV(String key, ExitVariable ev)
	{
		exitVarFields.put(key, ev);
	}

	
	public void adjustBeginLine(int evLineNum)
	{
		if (evLineNum == 0)
			return;
			
		if (beginNum > evLineNum)
			beginNum = evLineNum;
	}
	
	public ExitVariable getOrCreateEV(String name)
	{
		ExitVariable ev = exitVariables.get(name);
		if (ev == null)
		{
			ev = new ExitVariable(name);
			exitVariables.put(name, ev);
			ev.setParentBF(this);
		}
		return ev;
	}
	
	public ExitVariable getOrCreateEVField(String name)
	{
		ExitVariable ev = exitVarFields.get(name);
		if (ev == null)
		{
			//System.out.println("In gocEVF for " + name);
			ev = new ExitVariable(name);
			exitVarFields.put(name, ev);
			ev.setParentBF(this);
		}
		return ev;
	}
	
	public ExitProgram getOrCreateEPField(String name)
	{
		ExitProgram ev = exitProgFields.get(name);
		if (ev == null)
		{
			ev = new ExitProgram(name);
			exitProgFields.put(name, ev);
			ev.setParentBF(this);
		}
		return ev;
	}
	
	public ExitProgram getOrCreateEP(String name)
	{
		ExitProgram ep = exitPrograms.get(name);
		if (ep == null)
		{
			ep = new ExitProgram(name);
			exitPrograms.put(name, ep);
			ep.setParentBF(this);
		}
		return ep;
	}
	
	public ExitOutput getOrCreateEO(String name)
	{
		ExitOutput eo = exitOutputs.get(name);
		if (eo == null)
		{
			eo = new ExitOutput(name);
			exitOutputs.put(name, eo);
			eo.setParentBF(this);
		}
		return eo;
	}

	public HashMap<String, ExitProgram> getExitPrograms() {
		return exitPrograms;
	}

	public void setExitPrograms(HashMap<String, ExitProgram> exitPrograms) {
		this.exitPrograms = exitPrograms;
	}

	public HashMap<String, ExitVariable> getExitVariables() {
		return exitVariables;
	}

	public void setExitVariables(HashMap<String, ExitVariable> exitVariables) {
		this.exitVariables = exitVariables;
	}
	
	public void printVariables()
	{
		System.out.println("Variables");
		Iterator<ExitVariable> it = exitVariables.values().iterator();
		while (it.hasNext())
		{
			ExitVariable ev = (ExitVariable) it.next();
			System.out.println(ev.getName());
			ev.printInstances();
		}
		
		Iterator<ExitProgram> it2 = exitPrograms.values().iterator();
		while (it2.hasNext())
		{
			ExitProgram ep = (ExitProgram) it2.next();
			System.out.println(ep.getName());
			ep.printInstances();
		}
		
	}
	

	public short getBlamePointType() {
		return blamePointType;
	}

	public void setBlamePointType(short blamePointType) {
		this.blamePointType = blamePointType;
	}
	
	public HashMap<String, ExitOutput> getExitOutputs() {
		return exitOutputs;
	}

	public void setExitOutputs(HashMap<String, ExitOutput> exitOutputs) {
		this.exitOutputs = exitOutputs;
	}

	public HashMap<String, ExitVariable> getExitVarFields() {
		return exitVarFields;
	}

	public void setExitVarFields(HashMap<String, ExitVariable> exitVarFields) {
		this.exitVarFields = exitVarFields;
	}

	public HashMap<String, ExitProgram> getExitProgFields() {
		return exitProgFields;
	}

	public void setExitProgFields(HashMap<String, ExitProgram> exitProgFields) {
		this.exitProgFields = exitProgFields;
	}

	
	public String toString()
	{
		String prefixName = new String("");	
		return moduleName+":  "+prefixName+name;
	}
	
	public String getModuleName() {
		return moduleName;
	}

	public void setModuleName(String moduleName) {
		this.moduleName = moduleName;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public int getBeginNum() {
		return beginNum;
	}

	public void setBeginNum(int beginNum) {
		this.beginNum = beginNum;
	}

	public int getEndNum() {
		return endNum;
	}

	public void setEndNum(int endNum) {
		this.endNum = endNum;
	}
	
}
