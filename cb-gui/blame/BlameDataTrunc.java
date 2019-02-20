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
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.HashSet;


public class BlameDataTrunc extends BlameData {

  //private int instanceCount;
  //DEBUG FLAGS: just like #ifdef in C++
    private static final boolean DEBUG_PARSE_FRAME = false;
  
  
  public BlameDataTrunc(String outputFile, BlameContainer bc)
  {
    super(outputFile,bc);
  }
  
  public int numInstances()
  {
    //System.out.println("Num Instances in BlameDataTrunc for " + nodeName + " is " + instanceCount);
    return instanceCount;
  }
  
  public void print()
  {
    System.out.println("We have " + numInstances() + "instances");
  }
  

    public boolean nameFromUsrCode(String rawName)
    {
        HashSet<String> varNames = bc.getAllUsrVarNames();
        boolean isUsrVar = false;
        // rawName is the name to a field
        if (rawName.indexOf(".") >0) {
            StringTokenizer st = new StringTokenizer(rawName, ".");
            String[] esTokens = new String[st.countTokens()];

            int counter = 0;
            while (st.hasMoreTokens()) {
              esTokens[counter] = st.nextToken();
                counter++;
            }

            for (int i=0; i<counter; i++) {
                if (varNames.contains(esTokens[i])) {
                    isUsrVar = true;
                    break;
                }
            }
        }
        // not a field
        else {
            if (varNames.contains(rawName))
                isUsrVar = true;
        }

        return isUsrVar;
    }

  
  public String parseFrame(BufferedReader bufReader, Instance currInst, String line)
  {
    //String pathTokens[] = line.split("\\s"); //NOT sure why we need double backslash for a space
    String pathTokens[] = line.split("\t"); //NOT sure why we need double backslash for a space
    //String strFrameNum = pathTokens[1];
    //int frameNum = Integer.valueOf(strFrameNum).intValue();
    String funcName = pathTokens[2];
    /////////////added by HUI 08/23/15////////////////
    //String cleanFuncName = funcName.replaceAll("chpl_user_main","main");
    //funcName = funcName.replace("_chpl","");
    //we merge all on_fn* and coforall_fn* frames into their callers, so no this functions appear in GUI
    //funcName = funcName.replaceAll("");Not good for now since different loc with same name would merge
     //////////////////////////////////////////////////
    String moduleName = pathTokens[3];
    String pathName = pathTokens[4];
    //String strLineNum  = pathTokens[5];
    //int lineNumber = Integer.valueOf(strLineNum).intValue();
    String strBPType   = pathTokens[6]; //BlamePoint
    short bptType = Short.valueOf(strBPType).shortValue();
    String strFBeginLine = pathTokens[7];
    int beginLine = Integer.valueOf(strFBeginLine).intValue();
    String strFEndLine = pathTokens[8];
    int endLine = Integer.valueOf(strFEndLine).intValue();
        
    BlameFunction bf = bc.getOrCreateBlameFunction(funcName, moduleName, bptType);
    
    bf.setBeginNum(beginLine);
    bf.setEndNum(endLine);
    
    if (bptType > BlameFunction.NO_BLAME_POINT)
      bc.addBlamedFunction(funcName,bf);
    
    //SourceFiles sourceF = bc.getOrCreateSourceFile();
    if (SourceContainer.sourceFiles.get(moduleName) == null)
    {
      SourceFile sourceF = new SourceFile();
      sourceF.setFileName(moduleName);
      sourceF.setFilePath(pathName);
      
      // TODO: 6/10/10 Make this on demand
      //sourceF.setLinesOfCode();
      SourceContainer.sourceFiles.put(moduleName, sourceF);
    }
    
    //StackFrame sf = new StackFrame(frameNum, moduleName, pathName,lineNumber, bf);
    //currInst.addStackFrame(sf);
    //sf.setParentInst(currInst);
    
    try {
        
      String evLine = bufReader.readLine();
          
      while (evLine.indexOf("FRAME#") < 0)
      {
        if (evLine.indexOf("$$INSTANCE") >= 0)
        {
          return evLine;
        }
                
        if (evLine.indexOf("***No") >= 0)
        {
          evLine = bufReader.readLine();
          continue;
        }
          
        
        /*
        // Combo of Exit Variable Type and Exit Variable designators
        
        // Exit Variable Types
        // EV (Exit Variable)
        // EO (Exit Output)
        // EF (Exit Field)
        // EDF (Exit Derived Field)
        // LF (Local Field)
        // LDF (Local Derived Field)
        // VFL (Ignored Field Local)
        // VL  (Ignored Local)
        
        // Exit Variable Designators 
        //1(11) - EV lineNumbers
        //2(12) - EV declaredLine
        //3(13) - EV tempLine
        //4(14) - EV findBlamedExits
        //5(15) - BP lineNumbers
        //6(16) - BP declaredLine
        //7(17) - BP tempLine
        //8(18) - BP findBlamedExits

        //21(31) - LV lineNumbers
        //22(32) - LV declaredLine
        //23(33) - LV tempLine
        //24(34) - LV findBlamedExits

        //40(50) - SE EV seLineNumbers
        //41(51) - SE BP seLineNumbers
        //42(52) - SE LV seLineNumbers

        //61-64 - DQ EV

        //81(91) - SE Alias
        //82(92) - SE Relation

        //90 - Derived
         */
        
        
        StringTokenizer st = new StringTokenizer(evLine);
        
        String[] esTokens = new String[st.countTokens()];
        
        int counter = 0;
        while (st.hasMoreTokens())
        {
          esTokens[counter] = st.nextToken();
          counter++;
        }
        
        if (counter < 4)
        {
          System.err.println("Not enough info for line "+evLine);
          System.exit(1);
        }
        
        // EV Type concatenated with EV designator
        String strEVType = esTokens[0];
        
        // Full Name
        String strVarName = esTokens[1];
      
        // Type Name
        String strTypeName = esTokens[2];

        // Struct Name (may be NULL)
        String structName = esTokens[3];
        //bc.addType(strTypeName, structName);
            
        ExitSuper es = new ExitSuper();
        
        //use UsrNames to filter out most variables that didn't come from user code
        //02/07/18: currently we don't have user name log file for cuda code
        if (true/*nameFromUsrCode(strVarName)*/) {
          if (strEVType.indexOf("EV") >= 0)
          {
            ExitVariable ev = bf.getOrCreateEV(strVarName);
            es = ev;
          }
          ///added by Hui 01/08/16, to distinguish global vars from params/rets
          else if (strEVType.indexOf("EGV") >=0)
          {
           //we shouldn't blame the same GV for the same instance twice even though they appear in two different frames
            if (!currInst.nodeGVHash.contains(strVarName)) {
              ExitVariable ev = bf.getOrCreateEV(strVarName);
              ev.isGlobal = true;
        es = ev;
        currInst.nodeGVHash.add(strVarName);
        System.out.println("Add in EGV "+strVarName+" for inst "+currInst.getInstanceNum()+" on node "+currInst.getNodeName());
      }
      else {//to forall, it means we only created EV for coforall_fn, but not for on_fn and further
        evLine = bufReader.readLine();
        continue;
      }
    }
      ////////////////////////////////////////////////////////////////
      else if (strEVType.indexOf("EO") >= 0)
      {
    ExitOutput eo = bf.getOrCreateEO(strVarName);
    es = eo;
      }
      else if (strEVType.indexOf("U") >= 0)
      {
    ExitProgram ep = bf.getOrCreateEP(strVarName);
    es = ep;
      }
      else if (strEVType.indexOf("EF") >= 0 || strEVType.indexOf("EDF") >= 0)
      {
    ExitVariable evf = bf.getOrCreateEVField(strVarName);
    ExitVariable fieldParent;
    es = evf;
    
    //while (strVarName.indexOf('.') != strVarName.lastIndexOf('.'))
    while (strVarName.indexOf('.') > 0) //changed by Hui 01/20/16: mimic LF to get hiearchical view 
    {
        String newStrVarName = strVarName.substring(0,strVarName.lastIndexOf('.'));
        
        if (strVarName.indexOf('.') == strVarName.lastIndexOf('.'))
        {
      fieldParent = bf.getOrCreateEV(newStrVarName);
      //fieldParent.isGlobal = true; //added by Hui 01/26/16, in order to keep it in AllGlobalVariabales
        }
        else
        {
      fieldParent = bf.getOrCreateEVField(newStrVarName);
        }
        
        //added by Hui 01/25/16 for test
        if (DEBUG_PARSE_FRAME) {
      System.out.println("Add field "+evf.getName()+" to parent "+fieldParent.getName());
        }

        fieldParent.addField(evf);  
        
        //TOCHECK: WHETHER BELOW SHOULD BE ADDED !!!
        //if (fieldParent.getLastInst() != currInst)
      //fieldParent.addInstanceTrunc(currInst);

        strVarName = newStrVarName;
        evf = fieldParent; //added by Hui 01/20/16: mimic what LF did to show hiearchical view
    }
      }
      else if (strEVType.indexOf("LF") >= 0 || strEVType.indexOf("LDF") >= 0)
      {
    ExitProgram epf = bf.getOrCreateEPField(strVarName);
    ExitProgram fieldParent;
    es = epf;
    
    while (strVarName.indexOf('.') > 0)
    {
        String newStrVarName = strVarName.substring(0,strVarName.lastIndexOf('.'));
        
        if (newStrVarName.indexOf('.') < 0)
        {
      fieldParent = bf.getOrCreateEP(newStrVarName);
        }
        else
        {
      fieldParent = bf.getOrCreateEPField(newStrVarName);
        }
        
        fieldParent.addField(epf);  
        
        //if (fieldParent.getLastInst() != currInst)
      //fieldParent.addInstanceTrunc(currInst);
        
        
        strVarName = newStrVarName;
        epf = fieldParent;
    }
      }
      
      else if (strEVType.indexOf("VL") >= 0)
      {
    ExitProgram ep = bf.getOrCreateEP(strVarName);
    es = ep;
      }        
      else if (strEVType.indexOf("VFL") >= 0)//|| strEVType.indexOf("VL") >= 0)
      {
    
    ExitProgram epf = bf.getOrCreateEPField(strVarName);
    ExitProgram fieldParent;
    es = epf;
    
    while (strVarName.indexOf('.') > 0) {
      String newStrVarName = strVarName.substring(0,strVarName.lastIndexOf('.'));
      if (newStrVarName.indexOf('.') < 0)
      {
        fieldParent = bf.getOrCreateEP(newStrVarName);
      }
      else
      {
        fieldParent = bf.getOrCreateEPField(newStrVarName);
      }
        
      fieldParent.addField(epf);  
        
      //if (fieldParent.getLastInst() != currInst)
      //fieldParent.addInstanceTrunc(currInst);
              
      strVarName = newStrVarName;
      epf = fieldParent;
    }
  }
      else
      {
        evLine = bufReader.readLine();
        continue;
      }

      /////////////////////HUI//////////////////////////////////
      if (DEBUG_PARSE_FRAME) {
    System.out.println("Setting GT " + strTypeName + " ST " + structName + " for " + es.getName());
      }
      es.setGenType(strTypeName);
      es.setStructType(structName);
      es.setType();
      bc.addType(es.getType());
      
      if (es.getLastInst() != currInst)
        es.addInstanceTrunc(currInst);
      
      // We set this to make sure we don't have any repeats
      es.setLastInst(currInst);

      evLine = bufReader.readLine();
  } //end of nameFromUsrCode
  // If name not from usr code, we ignore this var completely
  else
  {
      evLine = bufReader.readLine();
      continue;
  }
      } //end of while #FRAME
      
      return evLine;
    }
    catch (IOException e) {
      System.err.println("exception happened - here's what I know: ");
      e.printStackTrace();
      System.exit(-1);
    }
    
    return new String();
  }
  

  public void parseOutputFile(String outputFile, String nodeName)
  {        
    File f = new File(outputFile);
    
    try {  
      //System.out.println("Read in file " + outputFile);
      BufferedReader bufReader = new BufferedReader(new FileReader(f));
      String line = null;
      
      //int instanceCount = 0; //was initialized in BlameData
      double instanceCountTemp = 0.0;
      while ((line = bufReader.readLine()) != null) 
      {        
        Instance currInst;
          
        if (line.indexOf("--INSTANCE") >= 0) 
        {
          String tokens[] = line.split("\\s");
          String tmp = tokens[1];
          int instNum = Integer.parseInt(tmp);
          tmp = tokens[2];
          double share = Double.parseDouble(tmp);
          tmp = tokens[3];
          int occurance = Integer.parseInt(tmp);
          currInst = new Instance(instNum, nodeName, share, occurance);
          instanceCountTemp += currInst.getWeight();
          int sfCount = 0;
          
          line = bufReader.readLine();
          
          // We are dealing with a FRAME
          while (line.indexOf("$$INSTANCE") < 0)
          {
            line = parseFrame(bufReader, currInst, line);
            sfCount++;
          }
          
          if (sfCount == 0)
          {  
            short bpType = BlameFunction.EXPLICIT_BLAME_POINT;
            BlameFunction bf = bc.getOrCreateBlameFunction("UNRESOLVED", "NO_MODULE", bpType);
            
            bc.addBlamedFunction("UNRESOLVED",bf);
            
            ExitProgram ep = bf.getOrCreateEP("INCOMPLETE_STACK_WALK");
            Double value = ep.blameByNode.get(currInst.getNodeName());
            if (value == null)
            {
              ep.blameByNode.put(currInst.getNodeName(), 
                                 currInst.getWeight());
            }
            else
            {
              value += currInst.getWeight();
              ep.blameByNode.put(currInst.getNodeName(), value);
            }    
            
           //ep.addInstance(currInst, null, bf);  
           //this.instanceCount--; //added by Hui 03/08/16: we should NOT count in the invalid (polling) samples
          }
        }
      }
      this.instanceCount = (int)instanceCountTemp; 
    }
    catch (IOException e) {
      System.err.println("exception happened - here's what I know: ");
      e.printStackTrace();
      System.exit(-1);
    }
  }
  
}
