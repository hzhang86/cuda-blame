package blame;

import java.util.Vector;
import java.util.Iterator;
import java.util.HashSet;
import java.util.Set;

public class Instance {

	private int instanceNum;
	private String nodeName;
	// keep the number of variables to be blamed on for this instance
	private int numBlamees;

	private	Vector<StackFrame> stackFrames;

	private Vector<ExitSuper> variables;
	private Vector<ExitSuper> noSkidVariables;
	
	public Vector<ExitSuper> getNoSkidVariables() {
		return noSkidVariables;
	}


	public void setNoSkidVariables(Vector<ExitSuper> noSkidVariables) {
		this.noSkidVariables = noSkidVariables;
	}


	public Vector<ExitSuper> getVariables() {
		return variables;
	}

	// getter of numBlamees
	public int getNumBlamees(){
		return numBlamees;
	}

	// increment numBlamees
	public void incNumBlamees(){
		this.numBlamees++;
	}

	public String getNodeName() {
		return nodeName;
	}
	
	public void addVariable(ExitSuper es)
	{
		variables.add(es);
	}

	public void addNoSkidVariable(ExitSuper es)
	{
		noSkidVariables.add(es);
	}
	
	
	public void setNodeName(String nodeName) {
		this.nodeName = nodeName;
	}


	public StackFrame getStackFrame(int frameNum)
	{
		Iterator<StackFrame> it = stackFrames.iterator();
		while (it.hasNext())
		{
			StackFrame sf =  (StackFrame) it.next();
			if (sf.getFrameNum() == frameNum)
				return sf;
		}
		
		return null;
	}
	
	
	Instance(int iN, String nodeName)
	{
		this.nodeName = nodeName;
		instanceNum = iN;
		stackFrames = new Vector<StackFrame>();
		variables = new Vector<ExitSuper>();
		noSkidVariables = new Vector<ExitSuper>();
	}
	
	
	
	public double calcSkidNumVariables(ExitSuper match)
	{
		System.out.println("In calcSkidNumVariables for " + match.getName());
		double numVariables = 0.0;
		double matchedVariables = 0.0;
		boolean fieldInMatch = false;

		Set<String> otherEVs = new HashSet<String>();
		
		Iterator<ExitSuper> it = variables.iterator();
		while (it.hasNext())
		{
			ExitSuper es = (ExitSuper) it.next();
			
			numVariables++;
			
			if (es.getName().contains(match.getName()))
			{
				matchedVariables++;
				if (es.getName().contains("."))
					fieldInMatch = true;
			}
			else if (es.getName().contains("."))
			{
				String truncName = es.getName().substring(0, es.getName().indexOf('.'));
				
				
				if ( otherEVs.contains(truncName))
				{
					
				}
				else
				{
					otherEVs.add(truncName);
					numVariables--;
				}
			}
		}
		
		if (fieldInMatch)
		{
			numVariables--;
			matchedVariables--;
		}
		
		
		System.out.print("Variable - " + match.getName() + " ");
		System.out.print("Instance " + this.getInstanceNum() + " " + numVariables);
		System.out.println(" " + matchedVariables + " " + fieldInMatch + " " + matchedVariables/numVariables);
	
		//return matchedVariables/numVariables;

		return matchedVariables;
	}
	
		
	
	public double calcNumVariables(ExitSuper match)
	{
		double numVariables = 0.0;
		double matchedVariables = 0.0;
		boolean fieldInMatch = false;

		Set<String> otherEVs = new HashSet<String>();
		
		
		//Iterator<ExitSuper> it = variables.iterator();
		//while (it.hasNext())
		for (ExitSuper es : this.noSkidVariables)
		{
			//ExitSuper es = (ExitSuper) it.next();
			
			System.out.println("Looking at " + es.getName());
			System.out.println("EVTypeName - " + es.getEvTypeName());
			
			// this comes into play with the skid factor
			//if (es.getEvTypeName().indexOf('-') > 0)
			//	continue;
			
			numVariables++;
			
			System.out.println("Made it here for " + es.getName());

			
			if (es.getName().contains(match.getName()))
			{
				matchedVariables++;
				System.out.println("Current val of matchedVariables is " + matchedVariables);
				if (es.getName().contains("."))
					fieldInMatch = true;
			}
			else if (es.getName().contains("."))
			{
				String truncName = es.getName().substring(0, es.getName().indexOf('.'));
				
				
				if ( otherEVs.contains(truncName))
				{
					
				}
				else
				{
					otherEVs.add(truncName);
					numVariables--;
				}
			}
		}
		
		if (fieldInMatch)
		{
			numVariables--;
			matchedVariables--;
		}
		
		
		System.out.print("Variable - " + match.getName() + " ");
		System.out.print("Instance " + this.getInstanceNum() + " " + numVariables);
		System.out.println(" " + matchedVariables + " " + fieldInMatch + " " + matchedVariables/numVariables);
	
		return matchedVariables/numVariables;

	}

	public Vector<StackFrame> getStackFrames() {
		return stackFrames;
	}

	public void setStackFrames(Vector<StackFrame> stackFrames) {
		this.stackFrames = stackFrames;
	}
	
	public void addStackFrame(StackFrame sf)
	{
		stackFrames.add(sf);
	}
	
	public void print()
	{
		System.out.println("Instance #" + instanceNum);
		System.out.println("We have " + stackFrames.size() + "stack Frames");
		Iterator<StackFrame> itr = stackFrames.iterator();
		
		while (itr.hasNext())
		{
			((StackFrame)itr.next()).print();	
		}
	}
	
	public String toString()
	{
		return "Instance " + instanceNum;
	}

	public int getInstanceNum() {
		return instanceNum;
	}

	public void setInstanceNum(int instanceNum) {
		this.instanceNum = instanceNum;
	}
	
}
