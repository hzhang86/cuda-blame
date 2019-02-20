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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Vector;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.io.IOException;
import java.util.TreeMap;
import java.util.Map;
import java.util.SortedMap;

public class BlameContainer {
	
	
	private Vector<ParentData> nodes;
	//private Vector<String> impBlamePoints;
	
	//private HashMap<String, SourceFile> allSourceFiles;
	private HashMap<String, BlameFunction> blamedFunctions;
	private HashMap<String, BlameFunction> allFunctions;
	private HashMap<String, BlameStruct> allStructs;
	
	private Vector<ExitSuper> allLocalVariables;
    //changed by Hui 02/16/16 global variables should be name distinguishable 
    //we want to aggregete all gv nodes with the same name from different funcs 
    //into a single node, therefore HashMap shall be used
    private HashMap<String, ExitSuper> allGlobalVariablesHash;
    //we still use allGlobalVariables later as it's easier than HashMap
	private Vector<ExitSuper> allGlobalVariables;
	private HashSet<String> allTypes;
    //04/07/17: keep all valid names from usr_names file
	private HashSet<String> allUsrVarNames;
	private HashSet<String> allUsrFuncNames;
	
	

	
	public HashSet<String> getAllTypes() {
		return allTypes;
	}

	
	public HashSet<String> getAllUsrVarNames() {
		return allUsrVarNames;
	}


	public HashSet<String> getAllUsrFuncNames() {
		return allUsrFuncNames;
	}


	public void addType(String typeName)
	{
		allTypes.add(typeName);
	}
	

	public void setAllTypes(HashSet<String> allTypes) {
		this.allTypes = allTypes;
	}


	public void setAllLocalVariables(Vector<ExitSuper> allLocalVariables) {
		this.allLocalVariables = allLocalVariables;
	}


	public HashMap<String, BlameFunction> getAllFunctions() {
		return allFunctions;
	}
	
	
	private boolean startsWithNum(String name)
	{
		if (name.length() == 0)
			return true;
		
		char firstChar = name.charAt(0);
		
		if (Character.isDigit(firstChar))
			return true;
		
		
		return false;
	}
	
	public Vector<ExitSuper> getAllLocalVariables() {
		
		// we calculate this on demand
		if (allLocalVariables.size() > 0)
			return allLocalVariables;
		
		Iterator<BlameFunction> it = getAllFunctions().values().iterator();
		while (it.hasNext())
		{
			BlameFunction bf = (BlameFunction) it.next();
			//private HashMap<String, ExitProgram> exitPrograms;
			Iterator<ExitProgram> it2 = bf.getExitPrograms().values().iterator();
			
			while (it2.hasNext())
			{
				ExitProgram ep = (ExitProgram) it2.next();
			//////////////////////////////////////////////////////////////////////////	
				System.out.println("Examining localVar " + ep.getName());
			/////////////////////////////////////////////////////////////////////////	
				String truncName = ep.getName().substring(ep.getName().lastIndexOf('.')+1);

				
				if (truncName.contains("ierr"))//|| truncName.startsWith("_"))
				{
					//////////////////////////////////////////////////////////
					System.out.println("Not Adding localVar " + truncName);
					//////////////////////////////////////////////////////////
				}
				else if (startsWithNum(truncName))
				{
					//////////////////////////////////////////////////////////
					System.out.println("Not Adding var " + truncName);
					///////////////////////////////////////////////////////////
				}
				else
				{
                    /////////////added by Hui///////////////////

                    //we want to keep call_tmps that came from main
                    /*if(ep.getParentBF().getName().compareTo("main.main") == 0){ 
                        if(ep.getName().compareTo("call_tmp") ==0)
                            ep.setName("foo");
                        else if(ep.getName().compareTo("ret4") ==0)
                            ep.setName("Number");
                        else 
                            if(ep.getName().contains("_tmp") ||
                                ep.getName().compareTo("ret3")==0)
                                continue;
                    }
                    else {
                        if(ep.getName().contains("_tmp")) 
                            continue;
                    }*/
                   
                    /* 
                    if(ep.getName().contains("_tmp") || ep.getName().contains("tmp_")
                            || ep.getName().contains("ret_") || ep.getName().contains("sum_")) //we don't want to see temp vars just for now 03/22/16
                        continue;
                    */
                    ////////////////////////////////////////////
				    System.out.println("Adding LocalVar " + ep.getName());
					allLocalVariables.add(ep);
				}
			}
		}
    	
		return allLocalVariables;
		
	}
	
    //changed by Hui 02/16/16: compute allGVsHash first, then feed values to allGVs
	public Vector<ExitSuper> getAllGlobalVariables() {
		
		// we calculate this on demand
		if (allGlobalVariables.size() > 0)
			return allGlobalVariables;
		
        //first we fill in the allGlobalVariablesHash
    	Iterator<BlameFunction> it = getAllFunctions().values().iterator();
		while (it.hasNext())
		{
			BlameFunction bf = (BlameFunction) it.next();
			//private HashMap<String, ExitProgram> exitPrograms;
			Iterator<ExitVariable> it2 = bf.getExitVariables().values().iterator();
			
			while (it2.hasNext())
			{
				ExitVariable ev = (ExitVariable) it2.next();
                String evName = ev.getName();
				
				System.out.println("Examining globalVar " + evName + " " + ev.getHierName());
				
				String truncName = evName.substring(evName.lastIndexOf('.')+1);
				
				if (ev.isGlobal) 
				{
					System.out.println("Adding " + evName + " to allGlobalVariablesHash.");
                    //added by Hui 02/16/16
                    ExitSuper gv = allGlobalVariablesHash.get(evName);
                    if (gv == null) 
                        allGlobalVariablesHash.put(evName, ev);
                    else 
                    {
                        //first update the aggregateBlame
                        double preAggBlame = gv.getAggregateBlame();
                        preAggBlame = preAggBlame + ev.getAggregateBlame();
                        gv.setAggregateBlame(preAggBlame);
                        //second update the parentBF if it's not main
                        BlameFunction pbf = gv.getParentBF();
                        if (pbf == null) 
                        {
                            System.out.println("Weird, egv("+evName+") didn't have a parentBF.");
                            pbf = getAllFunctions().get("main");
                            gv.setParentBF(pbf);
                        }
                        // 02/07/18: we now have bf for main,
                        // all GVs should belong to it
                        else if(pbf != null && pbf.getName().compareTo("main") !=0) 
                        {
                            pbf = getAllFunctions().get("main");
                            gv.setParentBF(pbf);
                        }

                        //we also need to merge all nodeInstances and varInstances of each nI 08/19/17
                        for (Map.Entry<String, NodeInstance> nI_e: ev.nodeInstances.entrySet()) 
                        {
                            String nodeName = nI_e.getKey();
                            NodeInstance nI = nI_e.getValue();
                            
                            //get previous nI node from gv
                            NodeInstance pre_nI = gv.nodeInstances.get(nodeName);
                            if (pre_nI == null)
                                gv.nodeInstances.put(nodeName, nI);
                            else 
                            {
                                //put all vIs from the nI to the previous nI
                                for (Map.Entry<Integer, VariableInstance> vI_e: nI.getVarInstances().entrySet())
                                {
                                    if (pre_nI.getVarInstances().containsKey(vI_e.getKey()) && evName.compareTo("elemToNode")==0)
                                        System.out.println("It had key: "+vI_e.getKey());
                                    pre_nI.getVarInstances().put(vI_e.getKey(), vI_e.getValue());
                                }
                            } 
                        }//for all ev's nodeInstances
                    }//allGlobalVariableHash had this ev
				} //isGlobal
			}
		}

        //Now we can put all GVs in the HashMap to Vector
        for (ExitSuper esh: allGlobalVariablesHash.values()) 
        {
			allGlobalVariables.add(esh);
        }
		
		if (allGlobalVariables.size() == 0)
		{
			ExitSuper es = new ExitSuper("NULL_GLOBAL");
			allGlobalVariables.add(es);
		}
    	
		return allGlobalVariables;
		
	}
	
	
	
	public Map<String, BlameFunction> getSortedFunctions(boolean standardSort) {
		
		if (standardSort)
		{
			//TreeMap<String, BlameFunction> sortedMap = new TreeMap<String, BlameFunction>(allFunctions);
			SortedMap<String, BlameFunction> sortedMap = new TreeMap<String, BlameFunction>(allFunctions);
			return sortedMap;
		}
		else
		{
			SortedMap<String, BlameFunction> sortedMap = MapValueSort.getValueSortedMap(allFunctions);
			return sortedMap;
		}
	}

	public void setAllFunctions(HashMap<String, BlameFunction> allFunctions) {
		this.allFunctions = allFunctions;
	}

	public HashMap<String, BlameStruct> getAllStructs() {
		return allStructs;
	}

	public void setAllStructs(HashMap<String, BlameStruct> allStructs) {
		this.allStructs = allStructs;
	}


	BlameContainer()
	{
		//impBlamePoints = new Vector<String>();
		nodes = new Vector<ParentData>();
		allFunctions = new HashMap<String, BlameFunction>();
		blamedFunctions = new HashMap<String, BlameFunction>();
		allStructs = new HashMap<String, BlameStruct>();
		
		allLocalVariables = new Vector<ExitSuper>();
        allGlobalVariablesHash = new HashMap<String, ExitSuper>();//added by Hui 02/16/16
		allGlobalVariables = new Vector<ExitSuper>();
		allTypes = new HashSet<String>();
        //added by Hui 04/07/17
        allUsrVarNames = new HashSet<String>();
        allUsrFuncNames = new HashSet<String>();
	}

	BlameFunction getOrCreateBlameFunction(String fName, String mName, short blamePointType)
	{
		//String fullName = new String(mName+":"+fName);
		//BlameFunction bf = allFunctions.get(fullName);
		BlameFunction bf = allFunctions.get(fName);
		if (bf == null)
		{
			bf = new BlameFunction(fName, mName, blamePointType);
			//System.out.println("Adding function " + fName);
			//allFunctions.put(fullName, bf);
			allFunctions.put(fName, bf);
		}
		return bf;
	}
	
	
	
	void parseStructs(String structFileName)
	{
		File f = new File(structFileName);
		
		try 
		{	
			BufferedReader bufReader = new BufferedReader(new FileReader(f));
			
			String line;
			
			// BEGIN STRUCT (or END STRUCTS)
			line = bufReader.readLine();
			
			while (line.indexOf("BEGIN STRUCT") >= 0)
			{
				BlameStruct bs = new BlameStruct();
				bs.parseStruct(bufReader);
				
				//System.out.println("Adding struct " + bs.getName());
				allStructs.put(bs.getName(), bs);
				
				line = bufReader.readLine();
			}		
		}
		catch(IOException ioe)
		{
			System.err.println("IO Error!");
			ioe.printStackTrace();
			System.exit(-1);
		}
	}
	
	
	void fillInStructs()
	{
		Iterator<BlameFunction> func_it = allFunctions.values().iterator();
		while (func_it.hasNext())
		{
			BlameFunction bf = (BlameFunction) func_it.next();
			
			// Go through all the exit programs
			Iterator<ExitProgram> ep_it = bf.getExitPrograms().values().iterator();
			
			// Go through all the exit variables
			Iterator<ExitVariable> ev_it = bf.getExitVariables().values().iterator();
			
			// Go through all the exit programs
			Iterator<ExitProgram> epf_it = bf.getExitProgFields().values().iterator();
			
			// Go through all the exit variables
			Iterator<ExitVariable> evf_it = bf.getExitVarFields().values().iterator();			
		
			
			while (ep_it.hasNext())
			{
				ExitProgram ep = (ExitProgram) ep_it.next();
				if (ep.getStructType() != null)
				{
					//System.out.println("EP - " + ep.getName() + " of type " + ep.getStructType());
					BlameStruct bs = allStructs.get(ep.getStructType());
					ep.fillInFields(bs);
				}
			}
			
			
			while (ev_it.hasNext())
			{
				ExitVariable ev = (ExitVariable) ev_it.next();
				if (ev.getStructType() != null)
				{
					//System.out.println("EV - " + ev.getName() + " of type " + ev.getStructType());
					BlameStruct bs = allStructs.get(ev.getStructType());
					ev.fillInFields(bs);
				}
			}
			
			while (epf_it.hasNext())
			{
				ExitProgram ep = (ExitProgram) epf_it.next();
				if (ep.getStructType() != null)
				{
					//System.out.println("EP - " + ep.getName() + " of type " + ep.getStructType());
					BlameStruct bs = allStructs.get(ep.getStructType());
					ep.fillInFields(bs);
				}
			}
			
			
			while (evf_it.hasNext())
			{
				ExitVariable ev = (ExitVariable) evf_it.next();
				if (ev.getStructType() != null)
				{
					//System.out.println("EV - " + ev.getName() + " of type " + ev.getStructType());
					BlameStruct bs = allStructs.get(ev.getStructType());
					ev.fillInFields(bs);
				}
			}
			
			
		}
	}

	
	void augmentStaticInfo(String staticInfoFileName)
	{
		File f = new File(staticInfoFileName);
		
		try 
		{	
			BufferedReader bufReader = new BufferedReader(new FileReader(f));
			String line = null;

			//System.out.println("Parsing file " + staticInfoFileName);
			
			while((line = bufReader.readLine()) != null)
			{
				if (line.indexOf("BEGIN FUNC") >= 0) 
				{
					//BEGIN_F_NAME
					bufReader.readLine();
					// Actual Nmae
					String fName = bufReader.readLine();
					// END F_NAME
					bufReader.readLine();
					
					// BEGIN M_NAME_PATH, <M_NAME_PATH> END M_NAME_PATH
					bufReader.readLine(); bufReader.readLine(); bufReader.readLine();
					
					// BEGIN M_NAME
					bufReader.readLine();
					// Module Name
					String mName = bufReader.readLine();
					// END M_NAME
					bufReader.readLine();
					
					String fullName = new String(mName+":"+fName);
					BlameFunction bf = allFunctions.get(fullName);
					
					// this means the function isn't in any of the traces, we don't care about it
					if (bf == null)
					{
						//System.out.println("Not interested in function " + funcName);
						while((line = bufReader.readLine()) != null)
						{
							if (line.indexOf("END FUNC") >= 0)
								break;
						}
					}
					else
					{
						//System.out.println("Interested in " + fullName);	
						bf.parseBlamedFunction(bufReader);
						
					}
				}
			}	
		}
		catch(IOException e)
		{
			System.err.println("exception happened - here's what I know: ");
			e.printStackTrace();
			System.exit(-1);
		}	
	}
	
	BlameFunction getOrCreateBlameFunction(String fName, String mName, int bN, int eN, int bP, int gP, boolean isUDBP)
	{
		//String fullName = new String(mName+":"+fName);
		//BlameFunction bf = allFunctions.get(fullName);
		BlameFunction bf = allFunctions.get(fName);
		if (bf == null)
		{
			bf = new BlameFunction(fName, mName, bN, eN, bP, gP, isUDBP);
			//allFunctions.put(fullName, bf);
			allFunctions.put(fName, bf);
		}
		return bf;
	}
	
	
	void printBlamed()
	{
		//Iterator it = blamedFunctions.values().iterator();
		Iterator<BlameFunction> it = allFunctions.values().iterator();
		while (it.hasNext())
		{
			BlameFunction bf = (BlameFunction) it.next();
			//System.out.println("Blame Function -- " + bf);
			bf.printVariables();
		}
	}
	
	//added by Hui 04/09/17 keep all valid var/func names from usr_names
    public void getUsrNames(String name_file)
    {   
      File nf = new File(name_file);

      try {
        BufferedReader bufReader = new BufferedReader(new FileReader(nf));
        String line = null;

        while ((line = bufReader.readLine()) != null) {
          if (line.indexOf("BEGIN VARIABLE NAMES") >= 0) {
            while ((line = bufReader.readLine()) != null) {
              if (line.indexOf("END VARIABLE NAMES") >=0) 
                break;
              
              String varName = new String(line);
              allUsrVarNames.add(varName);
            } //end of var while
          }

          else if (line.indexOf("BEGIN FUNCTION NAMES") >=0) {
            while ((line = bufReader.readLine()) != null) {
              if (line.indexOf("END FUNCTION NAMES") >=0) 
                break;
              
              String funcName = new String(line);
              allUsrFuncNames.add(funcName);
            } //end of func while
          }
        } //end of out-most while
      } //end of try

      catch(IOException ie) {
        ie.printStackTrace();
      }
    }


	void addNode(ParentData bd)
	{
		nodes.add(bd);
	}
	
	public Vector<ParentData> getNodes() {
		return nodes;
	}


	public void setNodes(Vector<ParentData> nodes) {
		this.nodes = nodes;
	}

	public void addBlamedFunction(String key, BlameFunction value)
	{
		blamedFunctions.put(key, value);
	}

	public BlameFunction getBlamedFunction(String key)
	{
		return blamedFunctions.get(key);
	}
	
	public HashMap<String, BlameFunction> getBlamedFunctions() {
		return blamedFunctions;
	}


	public void setBlamedFunctions(HashMap<String, BlameFunction> blamedFunctions) {
		this.blamedFunctions = blamedFunctions;
	}


}
