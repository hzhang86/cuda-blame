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
import java.util.Iterator;
import java.util.HashSet;
import java.util.Set;

public class Instance {

  private int instanceNum;
  private String nodeName;
  private double share;
  private int occurance;
  private double weight; //weight = share * occurance
  
  private  Vector<StackFrame> stackFrames;
  //08/20/17: make it public for easy access in BlameDataTrunc
  public HashSet<String> nodeGVHash;

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


  public String getNodeName() {
    return nodeName;
  }

  public Double getWeight() {
    return weight;
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
    nodeGVHash = new HashSet<String>();//added by Hui 08/20/17
    variables = new Vector<ExitSuper>();
    noSkidVariables = new Vector<ExitSuper>();
  }
  
  
  Instance(int iN, String nodeName, double sha, int occur)
  {
    this.nodeName = nodeName;
    instanceNum = iN;
    share = sha;
    occurance = occur;
    weight = sha * occur;

    stackFrames = new Vector<StackFrame>();
    nodeGVHash = new HashSet<String>();//added by Hui 08/20/17
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
      //  continue;
      
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
