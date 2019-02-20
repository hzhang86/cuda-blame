package blame;
//
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.text.*;


public class ExitSuper implements Comparable<ExitSuper> {
	
	public int compareTo(ExitSuper o) {
        return (int) o.numInstances() - (int) this.numInstances();
    }
	
	
	public static final int EXIT_VAR_FIELD = 6;
	public static final int LOCAL_VAR = 9;
	public static final int LOCAL_VAR_FIELD = 12;
	
	public static final int NO_EXIT       =   0;
	public static final int EXIT_PROG       = 1;
	public static final int EXIT_OUTP       = 2;
	public static final int EXIT_VAR_GLOBAL = 3;
	public static final int EXIT_VAR_RETURN = 4;
	public static final int EXIT_VAR_PARAM  = EXIT_VAR_RETURN;

	/* ESSENTIAL TO GUI */
	
	// Different Name representations
	protected String name;     // display name
	protected String hierName;  // for the case of fields, the name given by LLVM
	protected String fullHierName;  // full "path" of variable including enclosing ancestor structures
	
	
	protected String evTypeName;
	public String getEvTypeName() {
		return evTypeName;
	}

	public void setEvTypeName(String evTypeName) {
		this.evTypeName = evTypeName;
	}


	protected String typeName;
	protected int declaredLine;
	protected int lineNum;
	
	// Aggregate Blame for GUI
	//protected double[] blameByNode;
	protected HashMap<String, Double> blameByNode;
	


	protected double aggregateBlame;
	
	protected String genType;
	protected Set<Integer> lineNums;
	protected HashMap<String, ExitSuper> fields; 
	
	public boolean isGlobal;
	protected double rawWeight;
	protected double skidRawWeight;
	
	public double getSkidRawWeight() {
		return skidRawWeight;
	}

	public void setSkidRawWeight(double skidRawWeight) {
		this.skidRawWeight = skidRawWeight;
	}


	protected int calleeParam;
	
	protected boolean isField;
	
	protected String eStatus;
	protected String structType;
	
	protected Instance lastInst;
	
	
	/* METADATA INFORMATION */
	protected HashMap<String, NodeInstance> nodeInstances;
	protected BlameFunction parentBF;
	protected Set<Instance> instances;
	
	
	double numReadsInstances; 
	double weightedReadInstances;
	double skidReadInstances;
	
	
	
	ExitSuper()
	{
		//blameByNode = new double[Global.totalNodes];
		blameByNode = new HashMap<String, Double>();
		aggregateBlame = 0.0;
		
		nodeInstances = new HashMap<String, NodeInstance>();
		isField = false;
		//lineNumsByFunc = new HashMap<String, HashSet<Integer>>();
		
		isGlobal = false;
		hierName = null;
		eStatus = null;
		
		numReadsInstances = 0.0; 
		weightedReadInstances = 0.0;
		skidReadInstances = 0.0;
		
		lastInst = null;
		
		evTypeName = new String();
		
		lineNums = new HashSet<Integer>();
		fields = new HashMap<String, ExitSuper>();
		instances = new HashSet<Instance>();
	}
	
	public Instance getLastInst() {
		return lastInst;
	}

	public void setLastInst(Instance lastInst) {
		this.lastInst = lastInst;
	}

	ExitSuper(String n)
	{
		//blameByNode = new double[Global.totalNodes];
		name = n;
		
		blameByNode = new HashMap<String, Double>();
		aggregateBlame = 0.0;
		
		nodeInstances = new HashMap<String, NodeInstance>();
		isField = false;
		//lineNumsByFunc = new HashMap<String, HashSet<Integer>>();
		
		isGlobal = false;
		hierName = null;
		eStatus = null;
		
		lineNums = new HashSet<Integer>();
		fields = new HashMap<String, ExitSuper>();
		instances = new HashSet<Instance>();
			
	}
	

	public String getFullHierName() {
		return fullHierName;
	}

	public void setFullHierName(String fullHierName) {
		this.fullHierName = fullHierName;
	}
	

	public String getTypeName() {
		return typeName;
	}





	public void setTypeName(String typeName) {
		this.typeName = typeName;
	}





	public HashMap<String, Double> getBlameByNode() {
		return blameByNode;
	}





	public void setBlameByNode(HashMap<String, Double> blameByNode) {
		this.blameByNode = blameByNode;
	}





	public double getAggregateBlame() {
		return aggregateBlame;
	}





	public void setAggregateBlame(double aggregateBlame) {
		this.aggregateBlame = aggregateBlame;
	}





	public Set<Instance> getInstances() {
		return instances;
	}





	public void setInstances(Set<Instance> instances) {
		this.instances = instances;
	}





	public double getRawWeight() {
		return rawWeight;
	}

	public void setRawWeight(double rawWeight) {
		this.rawWeight = rawWeight;
	}

	
	public String getGenType() {
		return genType;
	}

	public void setGenType(String genType) {
		
		String editGT = genType.substring(genType.indexOf('*')+1);
		 
		this.genType = editGT;
	}

	public boolean isGlobal() {
		return isGlobal;
	}

	public void setGlobal(boolean isGlobal) {
		this.isGlobal = isGlobal;
	}

	public int getCalleeParam() {
		return calleeParam;
	}

	public void setCalleeParam(int calleeParam) {
		this.calleeParam = calleeParam;
	}
	
	public BlameFunction getParentBF() {
		return parentBF;
	}

	public void setParentBF(BlameFunction parentBF) {
		this.parentBF = parentBF;
	}
	
	

	public String getStructType() {
		return structType;
	}

	public void setStructType(String structType) {
		
		if (structType.indexOf("NULL") >= 0)
			this.structType = null;
		else
		{
			this.structType = structType;
			//System.out.println("Setting struct type for " + hierName + " to " + this.structType);
		}
	}

	public void addField(ExitSuper es)
	{
		if (!fields.containsKey(es.getName()))
		{
			//System.out.println("Adding field " + es.getName() + " to " + getName());
			fields.put(es.getName(), es);
		}		
	}
	
	public void transferVI()
	{
		
	}
	
	
	
	public void fillInFields(BlameStruct bs)
	{
		Iterator<BlameField> it = bs.getFields().iterator();
		
		while (it.hasNext())
		{
			BlameField bf = (BlameField) it.next();
			if (fields.get(bf.getName()) == null)
			{
				ExitSuper es = new ExitSuper();
				es.setName(bf.getName());
				
				//System.out.println("Adding field(2) " + bf.getName() + " to " + this.getName());
				fields.put(bf.getName(), es);
			}	
		}
	}
		
	public void addField(String s, BlameFunction bf)
	{
			
	}
	
	public HashMap<String, ExitSuper> getFields() {
		return fields;
	}

	public void setFields(HashMap<String, ExitSuper> fields) {
		this.fields = fields;
	}

	
	
	public int getDeclaredLine() {
		return declaredLine;
	}

	public void setDeclaredLine(int declaredLine) {
		this.declaredLine = declaredLine;
		this.lineNum = declaredLine;
	}

	public Set<Integer> getLineNums() {
		return lineNums;
	}

	public void setLineNums(Set<Integer> lineNums) {
		this.lineNums = lineNums;
	}

	
	
	public void addLine(int ln)
	{
		lineNums.add(new Integer(ln));
	}
	
	public String geteStatus() {
		return eStatus;
	}


	public void seteStatus(String eStatus) {
		this.eStatus = eStatus;
	}


	public String getHierName() {
		return hierName;
	}


	public void setHierName(String origName) {
		this.hierName = origName;
	}

	
	
	
	public boolean isField() {
		return isField;
	}


	public void setField(boolean isField) {
		this.isField = isField;
	}

	
	
	
	public boolean checkLineNum(int ln)
	{
		//HashSet<Integer> lineNums = lineNumsByFunc.get(funcName);
		
		if (lineNums == null)
			return false;
		
		if (lineNums.contains(new Integer(ln)))
			return true;
		else
			return false;
	}
	
	
	public void calcNumInstances()
	{
///////////////////////////////////////////////////////////////////////////////
		System.out.println("I'm in calcNumInstances()");
//////////////////////////////////////////////////////////////////////////////	
		Iterator<NodeInstance> it = nodeInstances.values().iterator();
		int num = 0;
		while (it.hasNext())
		{
			NodeInstance ni = (NodeInstance) it.next();
			String nodeName = ni.getNodeName();
			double val = (double) ni.numInstances();// HashMap<Integer, VariableInstance > varInstances.size()
///////////////////////////////////////////////////////////////			
			System.out.println("For Variable " + name + " and node " + nodeName + " the blame is " + val);
			
			this.blameByNode.put(nodeName, val);
			
			num += ni.numInstances();
		}
		
		for (ExitSuper es : this.getFields().values() )
			es.calcNumInstances();
		
		System.out.println("Aggregate blame is " + num);
		aggregateBlame = (double) num;
	}
	
	public void calcNumReadInstances()
	{
		System.out.println("FOR VARIABLE " + this.getName() + " Calculating Values");
		
		System.out.println("CALCULATING NUM READ INSTANCES");
	    this.numReadInstances();
	    
	    System.out.println("CALCULATING NUM SKID READ INSTANCES");
	    this.skidReadInstances();
	    
	    System.out.println("CALCULATING NUM WEIGHTED READ INSTANCES");
	    this.weightedReadInstances();
	    
		System.out.println("FOR VARIABLE " + this.getName() + " Done Calculating Values");

	    
	    System.out.println();
	    System.out.println();
	    System.out.println();
	    
	    this.aggregateBlame = this.numReadsInstances;
	}
	
	
	
	
	public double numInstances()
	{
		return this.aggregateBlame;
	}
	
	
	/*
	public int numInstances()
	{
		Iterator<NodeInstance> it = nodeInstances.values().iterator();
		int num = 0;
		while (it.hasNext())
		{
			NodeInstance ni = (NodeInstance) it.next();
			num += ni.numInstances();
		}
		return num;
	}
	*/
	
	
	void printInstances()
	{
	}
	
	void addInstance(Instance currInst, StackFrame currFrame)
	{
///////////////////////////////////////////////////////////////////
		System.out.println("I'm in addInstance");
//////////////////////////////////////////////////////////////////
		if (Global.typeOfTest == 1)
		{
			instances.add(currInst);
///////////////////////////////////////////////////////////////////
                        System.out.println("I'm in if1");
///////////////////////////////////////////////////////////////////
		}
		
		String nodeName = currInst.getNodeName();
		NodeInstance nI = nodeInstances.get(nodeName);
		if (nI == null)
		{
			nI = new NodeInstance(name, nodeName); // name is the variable
			nodeInstances.put(nodeName, nI);
///////////////////////////////////////////////////////////////////
                        System.out.println("I'm in if2");
///////////////////////////////////////////////////////////////////
		}
		
		VariableInstance vi = new VariableInstance(this, currInst);
		
		if (currFrame != null)
		{
			currFrame.addVarInstance(vi);
///////////////////////////////////////////////////////////////////
                        System.out.println("I'm in if3");
///////////////////////////////////////////////////////////////////
		}
		
		//nI.addInstance(lineNumber, currInst, vi );
		
		if (this.getEvTypeName().contains("-") == false)
		{
			nI.addInstance(vi);
///////////////////////////////////////////////////////////////////
			System.out.println("I'm in if4");
///////////////////////////////////////////////////////////////////
		}
		else
		{
			nI.addInstanceSkid(vi);
///////////////////////////////////////////////////////////////////
                        System.out.println("I'm in else");
///////////////////////////////////////////////////////////////////
		}
		vi.setNodeName(nodeName);
		vi.setWeight(getRawWeight());
		vi.setSkidWeight(getSkidRawWeight());
	
		
	}
	
	
	void addInstanceTrunc(Instance currInst)
	{
//////////////////////////////////////////////////////////////////////////////
//		System.out.println("In function addInstanceTrunc");
		// get the number of vars to be blamed for this instance
		int numBlamees = currInst.getNumBlamees();
		double fractionAggregateBlame = 1.0/numBlamees;

		Double value = this.blameByNode.get(currInst.getNodeName());
		// increment the blame value for a variable in a certain node
		if (value == null)
		{
			this.blameByNode.put(currInst.getNodeName(), fractionAggregateBlame);
		}
		else
		{
			//value++;
			value = value + fractionAggregateBlame;
			this.blameByNode.put(currInst.getNodeName(), value);
		}		
		//aggregateBlame++;
		aggregateBlame = aggregateBlame + fractionAggregateBlame;
/////////////////////////////////////////////////////////////////////////////////////
//		System.out.println("New value for aggregate blame is " + aggregateBlame + " for variable " + name);
/////////////////////////////////////////////////////////////////////////////////////
	}
	
	protected double percentageBlame()
	{
		double nI = numInstances();
		double tI = (double) Global.totalInstances;
		double dValue = (nI/tI)*100.0;
		
		return dValue;
	}
	
	public String getType() {
		return typeName;
	}

	public void setType() 
	{		
		String truncName = name.substring(name.lastIndexOf('.')+1);

		
		if (truncName.contains("ksp"))
		{
			typeName = "*_p_KSP";
		}
		else if (truncName.contains("snes"))
		{
			typeName = "*_p_SNES";
		}
		else if (this.genType == null)
		{
			typeName = new String("null");
		}	
		else if(this.genType.contains("Struct"))
		{
			typeName = new String();
			int numStars = genType.indexOf("Struct");
			
			for (int a = 0; a < numStars; a++)
				typeName += '*';
			
			typeName += this.structType;
		}
		else
		{
			typeName = this.genType;
		}	
	}

	double numReadInstances()
	{
		double nri = 0.0;
		
		Iterator<Instance> it = instances.iterator();
		while (it.hasNext())
		{
			Instance inst = (Instance) it.next();
			nri += inst.calcNumVariables(this);
		}
		
		numReadsInstances = nri;

		return nri;
	}
	
	double weightedReadInstances()
	{
		double wri = 0.0;
		
	    
	   Iterator<NodeInstance> it = nodeInstances.values().iterator();
		while (it.hasNext())
		{
			NodeInstance ni = (NodeInstance) it.next();
			//instances += ni.getNName();
			
			Iterator<VariableInstance> vi_it = ni.getVarInstances().values().iterator();
			while (vi_it.hasNext())
			{
				VariableInstance vi = (VariableInstance) vi_it.next();
				
				wri += (vi.getWeight() * vi.getInst().calcNumVariables(this));
				
				//instances += vi.getInst().getInstanceNum();
				//instances += " ";
			}
		}		
		
		weightedReadInstances = wri;
		
		return wri;		
	}
	
	
	double skidReadInstances()
	{
		double sri = 0.0;
		
		   Iterator<NodeInstance> it = nodeInstances.values().iterator();
			while (it.hasNext())
			{
				NodeInstance ni = (NodeInstance) it.next();
				
				Iterator<VariableInstance> vi_it = ni.getVarInstances().values().iterator();
				while (vi_it.hasNext())
				{
					VariableInstance vi = (VariableInstance) vi_it.next();
					
					sri += (vi.getSkidWeight() * vi.getInst().calcSkidNumVariables(this));
					System.out.println("Current val of sri is " + sri );
				}
				
				vi_it = ni.getSkidVarInstances().values().iterator();
				while (vi_it.hasNext())
				{
					VariableInstance vi = (VariableInstance) vi_it.next();
					
					sri += (vi.getSkidWeight() * vi.getInst().calcSkidNumVariables(this));
					System.out.println("Current val of sri(2) is " + sri);
				}
				
			}	
			
			
		skidReadInstances = sri;
				
		return sri;		
			
	}
	
	
	public String toString()
	{
		
		 double d = percentageBlame();
	     DecimalFormat df = new DecimalFormat("#.##");
	     
	     /*
	     String instances = new String();
	     
	     Iterator<NodeInstance> it = nodeInstances.values().iterator();
		while (it.hasNext())
		{
			NodeInstance ni = (NodeInstance) it.next();
			//instances += ni.getNName();
			
			Iterator<VariableInstance> vi_it = ni.getVarInstances().values().iterator();
			while (vi_it.hasNext())
			{
				VariableInstance vi = (VariableInstance) vi_it.next();
				instances += vi.getInst().getInstanceNum();
				instances += " ";
			}
		}
		*/
	     
	     
		String truncName = name.substring(name.lastIndexOf('.')+1);
		
		String funcName;
		
		if (parentBF != null)
			funcName = parentBF.getName();
		else
			funcName = "U_FUNC";

		
		if (Global.typeOfTest == 0)
		{
			return truncName + " [ " + typeName + " ]  "  + funcName + " " +  df.format(d) + "%";
		}
		else if (Global.typeOfTest == 1)
		{
			System.out.println("For Variable " + this.getName() + " " + this.getFullHierName());
			//double nri = numReadInstances();
			double nri = this.numReadsInstances;
			System.out.println("Number of Read Instances is " + nri);
			double tI = (double) Global.totalInstances;
			System.out.println("Number of Global Instances is " + tI);
			double dValue = (nri/tI)*100.0;
			System.out.println("DValue is " + dValue);
			
			//double wri = weightedReadInstances();
			double wri = this.weightedReadInstances;
			System.out.println("WRI is " + wri);
			double dwValue = (wri/tI)*100.0;
			System.out.println("DWValue is " + dwValue);
			
			
			//double sri = this.skidReadInstances();
			double sri = this.skidReadInstances;
			
			double sdwValue = (sri/tI)*100.0;
			System.out.println("SWValue is " + dwValue);
			
			return truncName + " [ " + typeName + " ]  "  + " " + df.format(nri) + " " + 
				df.format(dValue) + "%  " + df.format(wri) + " " + df.format(dwValue) + "%" + " " + 
				df.format(sri) + " " + df.format(sdwValue) + "%";

		}
		else
		{
			return null;
		}
	}

	public String toStringFull()
	{
		
		double d = percentageBlame();
	    DecimalFormat df = new DecimalFormat("#.##");
	     
	    String instances = new String();
	     
	    Iterator<NodeInstance> it = nodeInstances.values().iterator();
		while (it.hasNext())
		{
			NodeInstance ni = (NodeInstance) it.next();
			//instances += ni.getNName();
			
			Iterator<VariableInstance> vi_it = ni.getVarInstances().values().iterator();
			while (vi_it.hasNext())
			{
				VariableInstance vi = (VariableInstance) vi_it.next();
				instances += vi.getInst().getInstanceNum();
				instances += " ";
			}
		}
	     
	
		
		return name + " ["+this.genType+" ("+this.structType+") ]  " + df.format(d) + "%";
		
	}
	

	public String printDescription()
	{
		String s = new String(name + "\n");
		
		 double d = percentageBlame();
	     DecimalFormat df = new DecimalFormat("#.##");

		
		s += "Responsible for " + df.format(d) + "%\n";
		s += "Defined on Line " + lineNum + "\n";
		
		
		return s;
	}
	
	public String getName() {
		return name;
	}


	public void setName(String name) {
		this.name = name;
	}


	public HashMap<String, NodeInstance> getNodeInstances() {
		return nodeInstances;
	}


	public void setNodeInstances(HashMap<String, NodeInstance> nodeInstances) {
		this.nodeInstances = nodeInstances;
	}

	public int getLineNum() {
		return lineNum;
	}

	public void setLineNum(int lineNum) {
		this.lineNum = lineNum;
	}
	
}











/*
void addInstance(int lineNumber, Instance currInst, String[] lines, 
		Vector<VariableInstance> evI, String nodeName, String funcName, 
		StackFrame currFrame, boolean isExitVar)
{
	// TODO: Make this more efficient
	
	HashSet<Integer> lineNums = lineNumsByFunc.get(funcName);
	if (lineNums == null)
	{
		lineNums = new HashSet<Integer>();
		
		for (int i = 0; i < lines.length; i++)
		{
			Integer lineno = Integer.valueOf(lines[i]);
			lineNums.add(lineno);
		}
		
		lineNumsByFunc.put(funcName, lineNums);
	}
	
	NodeInstance nI = nodeInstances.get(nodeName);
	if (nI == null)
	{
		nI = new NodeInstance(name, nodeName);
		nodeInstances.put(nodeName, nI);
	}
	VariableInstance currVI = nI.addInstance(lineNumber, currInst, lines, evI);
	
	//HERE,  we need to check the Node Instances on the stack
	// frame directly above us and delete if necessary
	Vector<StackFrame> stackFrames = currInst.getStackFrames();
	Iterator it = stackFrames.iterator();
	
	StackFrame priorFrame = null;
	boolean foundIt = false;
	while (it.hasNext() && !foundIt)
	{
		StackFrame sf = (StackFrame)it.next();
		//System.out.println(sf);
		if (sf.equals(currFrame))
		{
			//System.out.println("Found it");
			foundIt = true;
		}
		else
			priorFrame = sf;
	}
	
	if (foundIt == false)
	{
		System.err.println("Where was it!");
		return;
	}
	else if (priorFrame == null)
	{
		return;
	}
		
	BlameFunction priorBF = priorFrame.getBf();
	it = priorBF.getExitPrograms().values().iterator();
	while (it.hasNext())
	{
		ExitSuper priorES = (ExitSuper) it.next();
		HashMap<String, NodeInstance> priorNIs = priorES.getNodeInstances();
		NodeInstance priorNI = priorNIs.get(nodeName);
		
		// This should never be the case but who knows
		if (priorNI == null)
			continue;
		
		Iterator it2 = priorNI.getVarInstances().iterator();
		while (it2.hasNext())
		{
			VariableInstance priorVI = (VariableInstance) it2.next();
			if (priorVI.equals(currVI) && !isExitVar)
			{
				VariableInstance tempVI;
				NodeInstance tempNI;
				if (priorFrame.frameNum < currFrame.frameNum)
				{
					tempVI = priorVI;
					priorVI = currVI;
					currVI = tempVI;
					
					tempNI = priorNI;
					priorNI = nI;
					nI = tempNI;
					
				}
				
				
				
				
				//System.out.println("WE have a match!");
				//System.out.println(priorVI);
				//System.out.println(currVI);
				//priorVI.print();
				//currVI.print();
				//System.out.println("End Match");
				priorNI.getVarInstances().remove(priorVI);
				return;
			}
		}
	}
	
}
*/

/*
public ExitSuper getOrCreateField(ExitSuper es, VariableInstance vi)
{
	if (fields.containsKey(es.getName()))
	{
		ExitSuper existing = fields.get(es.getName());
		System.out.println("Existing field " + existing.getName());
		
		HashMap<String,NodeInstance> nihash = existing.getNodeInstances();
			
		String otherNode = vi.getNodeName();
		NodeInstance existingNI = nihash.get(otherNode);
		existingNI.addInstance(vi);
		
		return existing;
	}
	else
	{
		System.out.println("Adding field " + es.getName() + " to " + hierName);
		ExitSuper newField = new ExitSuper(es);
		
		String otherNode = vi.getNodeName();
		System.out.println("Node name is " + otherNode);
		NodeInstance existingNI = newField.getNodeInstances().get(otherNode);
		
		if (existingNI == null)
		{
			System.out.println("existingNI null for " + newField.getHierName());
		}
		
		existingNI.addInstance(vi);
		
		fields.put(newField.getName(), newField);
		//newField.getFields().put(es.getName(), es);
		return newField;
	}
}
*/


/*
public ExitSuper(ExitSuper es)
{
	blameByNode = new double[Global.totalNodes];
	aggregateBlame = 0.0;
	
	System.out.println("Creating new ES from ES " + es.getHierName());
	name = es.getName();
	hierName = es.getHierName();
	
	
	if (es.getGenType() != null)
		genType = es.getGenType();
	
	declaredLine = es.getDeclaredLine();
	lineNum = es.getLineNum();

	nodeInstances = new HashMap<String, NodeInstance>();
	fields = new HashMap<String, ExitSuper>();
			
	isField = es.isField();
	eStatus = es.geteStatus();

	lineNums = es.getLineNums();	
	
	instances = new HashSet<Instance>();
	
	// We want to go through the nodes for the matching value
	//  and append it to the existing values
	Iterator<NodeInstance> ni_it = es.getNodeInstances().values().iterator();
	while(ni_it.hasNext())
	{
		NodeInstance ni = (NodeInstance) ni_it.next();
		
		System.out.println("Node name(2) is " + ni.getNName());
		NodeInstance blankNI = new NodeInstance(es.getHierName(), ni.getNName());
		
		
		System.out.println("Putting new NI " + ni.getNName() + " for " + hierName);
		nodeInstances.put(ni.getNName(), blankNI);
	} 
	
}
*/


	
	
