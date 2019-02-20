package profiler;

import java.text.DecimalFormat;
import java.util.*;


public class ProfilerFunction {

	String contextName;
	int numPeriods;
	
	
	public int getNumPeriods() {
		return numPeriods;
	}

	public void setNumPeriods(int numPeriods) {
		this.numPeriods = numPeriods;
	}

	Vector<ProfilerFunction> funcDescendants;
	
	
	public void addDescendant(ProfilerFunction pf)
	{
		funcDescendants.add(pf);
	}
	
	public Vector<ProfilerFunction> getFuncDescendants() {
		return funcDescendants;
	}


	public void setFuncDescendants(Vector<ProfilerFunction> funcDescendants) {
		this.funcDescendants = funcDescendants;
	}


	public ProfilerFunction getFuncParent() {
		return funcParent;
	}


	public void setFuncParent(ProfilerFunction funcParent) {
		this.funcParent = funcParent;
	}

	// Since this is the full context we have only one parent
	ProfilerFunction funcParent;
	
	
	public String getContextName() {
		return contextName;
	}


	public void setContextName(String contextName) {
		this.contextName = contextName;
	}


	public String getName() {
		return name;
	}


	public void setName(String name) {
		this.name = name;
	}


	public HashMap<String, ProfileInstance> getInstances() {
		return instances;
	}


	public void setInstances(HashMap<String, ProfileInstance> instances) {
		this.instances = instances;
	}

	String name;
	HashMap<String, ProfileInstance> instances;
	
	ProfilerFunction(String name)
	{
		this.contextName = name;
		this.name = name;
		
		funcDescendants = new Vector<ProfilerFunction>();
		
		numPeriods = 0;
		for (int a = 0; a < name.length(); a++)
		{
			if (name.charAt(a) == '.')
				numPeriods++;
		}
		
		if (numPeriods > 0)
		{
			int lastPeriod = name.lastIndexOf('.');
			this.name = name.substring(lastPeriod + 1);
		}
		
		instances = new HashMap<String, ProfileInstance>();
		
	}
	
	
	 public int compareTo(Object obj) 
	 {
		 ProfilerFunction comparor = (ProfilerFunction) obj;
		 double cMBV = comparor.getInstances().size();
		 double mbv = instances.size();
		 
		 if (mbv == cMBV)
			 return 0;
		 else if (mbv > cMBV)
			 return 1;
		 else 
			 return -1;
	}
	 
	 protected double percentageTime()
		{
			double nI = (double) instances.size();
			double tI = (double) Global.totalInstances;
			double dValue = (nI/tI)*100.0;
			
			return dValue;
		}
		
		
	 
	
	public String toString()
	{
		 double d = percentageTime();
	     DecimalFormat df = new DecimalFormat("#.##");
	     String padding = new String();
	     
	     for (int a = 0; a < numPeriods; a++)
	     {
	    	 padding += "  ";
	     }
	     
	     return padding + name + " " + df.format(d) + "%";
	}
	 
	void addInstance(ProfileInstance pi)
	{
		instances.put(pi.getIdentifier(), pi);
	}
	
}
