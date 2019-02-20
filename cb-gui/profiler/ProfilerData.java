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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.StringTokenizer;
//import java.util.Vector;

//import blame.BlameFunction;



public class ProfilerData {
  
  private HashMap<String, ProfilerFunction> allFunctions;
    //04/07/17: keep all valid names from usr_names file
  private HashSet<String> allUsrVarNames;
  private HashSet<String> allUsrFuncNames;

  public ProfilerData()
  {
    allFunctions = new HashMap<String, ProfilerFunction>();
        //added by Hui 04/07/17
        allUsrVarNames = new HashSet<String>();
        allUsrFuncNames = new HashSet<String>();
  }
  
  
  public HashMap<String, ProfilerFunction> getAllFunctions() {
    return allFunctions;
  }

  public HashSet<String> getAllUsrVarNames() {
    return allUsrVarNames;
  }


  public HashSet<String> getAllUsrFuncNames() {
    return allUsrFuncNames;
  }

  public Map<String, ProfilerFunction> getSortedFunctions(boolean standardSort) {
    
    if (standardSort)
    {
      SortedMap<String, ProfilerFunction> sortedMap = new TreeMap<String, ProfilerFunction>(allFunctions);
      return sortedMap;
    }
    else
    {
      SortedMap<String, ProfilerFunction> sortedMap = MapValueSort.getValueSortedMap(allFunctions);
      return sortedMap;
    }
  }
  
  

  public void setAllFunctions(HashMap<String, ProfilerFunction> allFunctions) {
    this.allFunctions = allFunctions;
  }

  public double parseNode(String outputFile)
  {
    String nodeName = outputFile.substring(outputFile.indexOf("PARSED_") + 7);
    return parseOutputFile(outputFile, nodeName);
  }
  
  //allFunctions has names like main.foo.bar, main.foo, main....
  public void addInstanceToEntireCallPath(String funcName, ProfileInstance pi, ProfilerFunction oldFunc)
  {
    ProfilerFunction pf = allFunctions.get(funcName);
        
    if (pf == null)
    {
      pf = new ProfilerFunction(funcName);
      allFunctions.put(funcName, pf);
      //System.out.println("Adding " + funcName);
    }
    
    // VERY IMPORTANT
    pf.addInstance(pi);

    if (oldFunc != null)
    {
      //System.out.println("Adding descendant " + oldFunc.getName() + " to " + pf.getName());
      pf.addDescendant(oldFunc);
      oldFunc.setFuncParent(pf);
    }
    
    int lastPeriod = funcName.lastIndexOf('.');
    if (lastPeriod == -1)
    {
      return;
    }
    
    String truncString = funcName.substring(0, lastPeriod);
    addInstanceToEntireCallPath(truncString, pi, pf);
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


  public String removeGenFrames(String callPathName)
    {
        
        HashSet<String> funcNames = getAllUsrFuncNames();
        String cp = new String(""); //Initialize an empty string to return
        StringBuilder sb = new StringBuilder(cp); //String.concat() doesn't work! Use StringBuilder
        // callPathName is the original call path, including gen frames
        if (callPathName.indexOf(".") >0) {
            StringTokenizer st = new StringTokenizer(callPathName, ".");
            String[] esTokens = new String[st.countTokens()];
            int counter = 0;
            // store tokens into esTokens array
            while (st.hasMoreTokens()) {
              esTokens[counter] = st.nextToken();
                counter++;
            }
            // feed usr frames into cp
            for (int i=0; i<counter; i++) {
                if (funcNames.contains(esTokens[i])) {
                    sb.append(esTokens[i]);
                    sb.append(".");
                }
            }

            cp = sb.toString();
            // chop the last '.' symbol
            if (cp.endsWith("."))
                cp = cp.substring(0, cp.length()-1);
        }

        // only one frame in this instance, could be Empty if it's gen frame
        else {
            if (funcNames.contains(callPathName))
                cp = callPathName;
        }
        

        return cp;
    }


  public String parseFrame(BufferedReader bufReader, ProfileInstance currInst, String line)
  {
    //String pathTokens[] = line.split("\\s"); //Q: why double back slash
    String pathTokens[] = line.split("\\t"); //Q: why double back slash
    //String strFrameNum = pathTokens[1];
    //int frameNum = Integer.valueOf(strFrameNum).intValue();
    String funcName = pathTokens[2];
    /////////////added by HUI 08/23/15////////////////
    //String cleanFuncName = funcName.replaceAll("chpl_user_main","main");
    //funcName = funcName.replace("_chpl","");
    // remove all generated frames inside the call path 05/08/17
    // commented out on 07/31/17 to enable display of chpl_user_main and chpl_gen_main, 
    // not sure whether we shall cleanup non-user frames since sometimes that's valuable info
    //funcName = removeGenFrames(funcName);
        
    if (!funcName.isEmpty()) {
    // we only add this instance if the trimed funcName isn't empty (means we have at least one usr frame in this call path)
      addInstanceToEntireCallPath(funcName, currInst, null);
    }
    
    else
      System.out.println("Found one instance without any usr frames in the call path!");
    /*
    ProfilerFunction pf = allFunctions.get(funcName);
    //System.out.println("cashew");
    
    
    
    if (pf == null)
    {
      pf = new ProfilerFunction(funcName);
      allFunctions.put(funcName, pf);
      System.out.println("Adding " + funcName);
    }
    */
    
    
    try {
      String evLine = bufReader.readLine();    
      while (evLine.indexOf("FRAME#") < 0)
      {
        if (evLine.indexOf("$$INSTANCE") >= 0)
        {
          return evLine;
        }
                  
        // we only need to parse the first FRAME so pop out all other lines in this instance
        evLine = bufReader.readLine();
      }
      return evLine; //until next frame
    }
    catch (IOException e) {
      System.err.println("exception happened - here's what I know: ");
      e.printStackTrace();
      System.exit(-1);
    }
    
    return new String();
  }
  
  public double parseOutputFile(String outputFile, String nodeName)
  {
    File f = new File(outputFile);
    double instanceCount = 0;
    try {  
      //System.out.println("Read in file " + outputFile);
      BufferedReader bufReader = new BufferedReader(new FileReader(f));
      String line = null;
      
      
      while ((line = bufReader.readLine()) != null) 
      {        
        ProfileInstance currInst;
        
        if (line.indexOf("--INSTANCE") >= 0) 
        {
          String tokens[] = line.split("\\s");
          String tmp = tokens[1];
          int instNum = Integer.parseInt(tmp);
          tmp = tokens[2];
          double share = Double.parseDouble(tmp);
          tmp = tokens[3];
          int occurance = Integer.parseInt(tmp);
          currInst = new ProfileInstance(instNum, nodeName, share, occurance);
          instanceCount += currInst.getWeight();
            
          line = bufReader.readLine();
          
          boolean atLeastOneFrame = false;
          
          // We are dealing with a FRAME
          if (line.indexOf("FRAME") >= 0)
          {
            atLeastOneFrame = true;
                        //just to get the func name for this frame
            line = parseFrame(bufReader, currInst, line);
          }
          // we only need to parse the first FRAME since it has complete call path
          while (line.indexOf("$$INSTANCE") < 0)
          {
            line = bufReader.readLine();
            //line = parseFrame(bufReader, currInst, line);
          }
          
          if (atLeastOneFrame == false)
          {
            String funcName = "UNRESOLVED";
            ProfilerFunction pf = allFunctions.get(funcName);
            
            if (pf == null)
            {
              pf = new ProfilerFunction(funcName);
              allFunctions.put(funcName, pf);
              //System.out.println("Adding " + funcName);
            }
            
            pf.addInstance(currInst);
          }
        }
      }
    }
    catch (IOException e) {
      System.err.println("exception happened - here's what I know: ");
      e.printStackTrace();
      System.exit(-1);
    }
    return instanceCount;
  }
}
