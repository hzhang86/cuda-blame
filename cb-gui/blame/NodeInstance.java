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

import java.util.HashMap;
import java.util.Iterator;
import java.text.*;

public class NodeInstance {

  String varName;
  String nodeName;
  
  int numTotalInstances;
  
  //private Vector<VariableInstance> varInstances;
  private HashMap<Integer, VariableInstance > varInstances; //unique instNum in each node
  private HashMap<Integer, VariableInstance > skidVarInstances;
  
  
  NodeInstance(String vN, String nN)
  {
    varName = vN;
    //System.out.println("Setting Node Name(2) to " + nN);
    nodeName = nN;
    varInstances = new HashMap<Integer, VariableInstance>();
    skidVarInstances = new HashMap<Integer, VariableInstance>();
  }
  
  
  public HashMap<Integer, VariableInstance> getSkidVarInstances() {
    return skidVarInstances;
  }

  public void setSkidVarInstances(
      HashMap<Integer, VariableInstance> skidVarInstances) {
    this.skidVarInstances = skidVarInstances;
  }

  
  
  public double numInstances()
  {
    double total = 0.0;
    Iterator<VariableInstance> vi_it = varInstances.values().iterator();
    while (vi_it.hasNext())
    {
      VariableInstance vi = (VariableInstance) vi_it.next();
      total += vi.getInst().getWeight();
    }
    
    return total;
  }
  
  public String toString()
  {
    if (!Global.GPU_TEST) {
      double cpuTime = Global.SAMPLE_INTERVAL * numInstances(); 
      DecimalFormat df = new DecimalFormat("#.###");
      return "Node:" + nodeName + ", Total CPU time(s): " + df.format(cpuTime);
    }
    else {
      double gpuNumSamples = numInstances()/1000;
      DecimalFormat df = new DecimalFormat("#.###");
      return "Node: " + nodeName + ", #PC_SAMPLE: " + df.format(gpuNumSamples) + "k"; //unit is a thousand
    }
  }
  
  void addInstance(VariableInstance vi)
  {
        //String instID = vi.getInst().getNodeName() + "-" + vi.getInst().getInstanceNum();
    varInstances.put(vi.getInst().getInstanceNum(), vi);
    //varInstances.add(vi);
  }
  
  void addInstanceSkid(VariableInstance vi)
  {
    skidVarInstances.put(vi.getInst().getInstanceNum(),vi);
  }
  
  
  public String getNName()
  {
    return nodeName;
  }

  public String getNodeName() {
    return nodeName + " " + varInstances.size();
  }

  public void setNodeName(String nodeName) {
    System.out.println("Setting node name to " + nodeName);
    this.nodeName = nodeName;
  }

  public HashMap<Integer, VariableInstance> getVarInstances() {
    return varInstances;
  }

  public void setVarInstances(HashMap<Integer, VariableInstance> varInstances) {
    this.varInstances = varInstances;
  }

  public String getVarName() {
    return varName;
  }

  public void setVarName(String varName) {
    this.varName = varName;
  }

  public int getNumTotalInstances() {
    return numTotalInstances;
  }

  public void setNumTotalInstances(int numTotalInstances) {
    this.numTotalInstances = numTotalInstances;
  }
  
}













/*
VariableInstance addInstance(int lineNumber, Instance currInst, String[] lines, Vector<VariableInstance> evI)
{
  boolean isMatch = false;
  
  Vector<Integer> lineNumbers = new Vector<Integer>();
  
  for (int i = 0; i < lines.length; i++)
  {
    Integer lineno = Integer.valueOf(lines[i]);
    lineNumbers.add(lineno);
    
    if (lineNumber == lineno.intValue())
    {
      isMatch = true;
    }
  }
  
  if (isMatch == true)
  {
    //System.out.println("Match for EV " + name + " for linenum " + lineNumber);
    
    VariableInstance vi = new VariableInstance(lineNumbers, currInst, varName);
    
    // This is specific to the stack frame
    evI.add(vi);
    
    // This is specific to the exit variable across all instances
    varInstances.add(vi);
    
    return vi;
  }
  else
    return null;
}
*/



