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

import java.util.*;

// EVs are global variables, return values, pointer parameters
public class ExitVariable extends ExitSuper {

	/*
	public double compareTo(ExitVariable o) {
        return o.numInstances() - this.numInstances();
    }*/
	
	ExitVariable(String n)
	{
		super();
		name = n;
		hierName = n;
		isGlobal = false;	
	}	
	
	public String printDescription()
	{
		String s = new String("Exit Variable - " + super.printDescription());
		return s;
	}
	
	
	
	public void addField(String s, BlameFunction bf)
	{
		HashMap<String, ExitVariable> eVFields = bf.getExitVarFields();
		ExitSuper es = eVFields.get(s);
		if (es != null)
		{
			//int lastDot = es.getHierName().lastIndexOf('.');
			//lastDot++;
			
			//String truncHierName = es.getHierName().substring(lastDot);
			fields.put(es.getName(), es);
			//es.setName(truncHierName);
		}
		else
		{
			System.out.println("EV addField NULL for " + s);
		}
	}
}








/*
public String toString()
{
	
//	return name + " " + numInstances();
	double nI = (double) numInstances();
	double tI = (double) Global.totalInstances;
	double dValue = (nI/tI)*100.0;
	
     DecimalFormat df = new DecimalFormat("#.##");

	//int value = (int) dValue;
	return "<html><font color=\"blue\">" + name + " ( " + genType + " ) " + df.format(dValue) + "%" + "</font></html>";
}
*/






/*
public void transferVI()
{
	System.out.println("In TransferVI for " + hierName);
	BlameFunction bf = this.getParentBF();
	Iterator<ExitVariable> it = bf.getExitVariables().values().iterator();
	
	while (it.hasNext())
	{
		ExitVariable es = (ExitVariable) it.next();
		System.out.println("Looking(1) at " + es.getHierName());
		if (es.getHierName().indexOf('-') <= 0 )
			continue;
		
		System.out.println("Looking(2) at " + es.getHierName());
		if (es.getHierName().indexOf(hierName) < 0)
			continue;
		
		System.out.println("Transferring VI from " + es.getHierName() + " to " + hierName);
		
		// We want to go through the nodes for the matching value
		//  and append it to the existing values
	
		//HashMap<String,NodeInstance> nihash = es.getNodeInstances();
		
		Iterator<NodeInstance> ni_it = es.getNodeInstances().values().iterator();
		while(ni_it.hasNext())
		{
			
			// Existing
			NodeInstance ni = (NodeInstance) ni_it.next();
			
			// Probably not there to start
			NodeInstance matchingNI = nodeInstances.get(ni.getNName());
		
			if (matchingNI == null)
			{
				matchingNI = new NodeInstance(this.getHierName(), ni.getNName());
				nodeInstances.put(matchingNI.getNName(), matchingNI);

			}
		
			matchingNI.getVarInstances().addAll(ni.getVarInstances());


		}	 
	}
	
}
*/

