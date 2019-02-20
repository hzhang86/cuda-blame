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
  
  
  public double getTotalBlamedInstances() 
  {
    double total = 0.0;
    Iterator<ProfileInstance> pi_it = instances.values().iterator();
    while (pi_it.hasNext())
    {
      ProfileInstance pi = (ProfileInstance) pi_it.next();
      total += pi.getWeight();
    }

    return total;
  }

  
  public int compareTo(Object obj) 
  {
    ProfilerFunction comparor = (ProfilerFunction) obj;
    double cMBV = comparor.getTotalBlamedInstances();
    double mbv = getTotalBlamedInstances();
     
    if (mbv == cMBV)
      return 0;
    else if (mbv > cMBV)
      return 1;
    else 
      return -1;
  }
   
  protected double percentageTime()
  {
    double nI = getTotalBlamedInstances();
    double tI = (double) Global.totalInstances;
    double dValue = (nI/tI)*100.0;
    
    return dValue;
  }
    
    
   
  
  public String toString()
  {
     double d = percentageTime();
     DecimalFormat df = new DecimalFormat("#.##");
     String padding = new String();
       
     //TOCHECK: don't think we need padding here
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
