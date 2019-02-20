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
//

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;
import java.util.Iterator;
import java.util.StringTokenizer;

public class BlameData extends ParentData {
  

  BlameData()
  {
    bc = null;
    nodeName = null;
  }
  
  BlameData(String outputFile, BlameContainer bc)
  {
    this.bc = bc;
    instanceCount = 0;  // = instanceNum
    
    nodeName = outputFile.substring(outputFile.indexOf("PARSED_") + 7);
    parseOutputFile(outputFile, nodeName);
    
  }
  
  
  
  public String parseFrame(BufferedReader bufReader, Instance currInst, String line)
  {
        ///////////////////////////////
        System.out.println("parseFrame in BlameData Called !");
    String pathTokens[] = line.split("\\s"); // return an array of strings that are separated by "\\s"(white space) in line
    String strFrameNum = pathTokens[1];
    int frameNum = Integer.valueOf(strFrameNum).intValue();   //=4
    String funcName = pathTokens[2];        //=main
    String moduleName = pathTokens[3];        //=helloBlame3.c
    String pathName = pathTokens[4];  //=/fs/mashie/hzhang86/myBlame/Blame_Distro/Projects/hellBlame3/
    String strLineNum  = pathTokens[5];         //=89
    int lineNumber = Integer.valueOf(strLineNum).intValue();  
    String strBPType   = pathTokens[6];        //=1 bpType = isBlamePoint, 1 means explicit blame point
    short bptType = Short.valueOf(strBPType).shortValue();
    String strFBeginLine = pathTokens[7];        //=82 why not 80 ?
    int beginLine = Integer.valueOf(strFBeginLine).intValue();
    String strFEndLine = pathTokens[8];        //=108
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
    
    StackFrame sf = new StackFrame(frameNum, moduleName, pathName,lineNumber, bf);
    currInst.addStackFrame(sf);
    sf.setParentInst(currInst);
    
    try {
        String evLine = bufReader.readLine(); // first line of variables
          
      while (evLine.indexOf("FRAME#") < 0) // means there are variables to blame
      {
///////////////////////////////////////////////////////////////////////////////////////////////////
        System.out.println("parsing frame, will do addInstance in the end");
//////////////////////////////////////////////////////////////////////////////////////////////////
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
          System.err.println("Not enough info for line " + evLine);
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
        
        if (strEVType.indexOf("EV") >= 0)
        {
          ExitVariable ev = bf.getOrCreateEV(strVarName);
          es = ev;
        }
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
          
          while (strVarName.indexOf('.') != strVarName.lastIndexOf('.'))
          {
            String newStrVarName = strVarName.substring(0,strVarName.lastIndexOf('.'));
            
            if (strVarName.indexOf('.') == strVarName.lastIndexOf('.'))
            {
              fieldParent = bf.getOrCreateEV(newStrVarName);
            }
            else
            {
              fieldParent = bf.getOrCreateEVField(newStrVarName);
            }
            
            fieldParent.addField(evf);  
            fieldParent.addInstance(currInst, sf);

            strVarName = newStrVarName;
            
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
            fieldParent.addInstance(currInst, sf);
            
            strVarName = newStrVarName;
            epf = fieldParent;
          }
        }
        
        else if (strEVType.indexOf("VL") >= 0)
        {
          ExitProgram ep = bf.getOrCreateEP(strVarName);
          es = ep;
        }        
        else if (strEVType.indexOf("VFL") >= 0 )//|| strEVType.indexOf("VL") >= 0)
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
            fieldParent.addInstance(currInst, sf);
            
            strVarName = newStrVarName;
            epf = fieldParent;

          }
        }
        else
        {
            evLine = bufReader.readLine();
            continue;
        }
        
        
        //System.out.println("Setting GT " + strTypeName + " ST " + structName + " for " + es.getName());
        es.setGenType(strTypeName);
        es.setStructType(structName);
        es.setType();
        //es.setParentBF(bf);
        bc.addType(es.getType());
  
        es.addInstance(currInst, sf);
      
        
        evLine = bufReader.readLine();
      }
      
      
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
    instances = new Vector<Instance>();
        
    File f = new File(outputFile);
    
    try {  
      //System.out.println("Read in file " + outputFile);
      BufferedReader bufReader = new BufferedReader(new FileReader(f));
      String line = null;
      
      int instanceCount = 0;
      
      while ((line = bufReader.readLine()) != null) 
      {
        //System.out.println("LINE --- " + line);
        
        Instance currInst;
        
        
        if (line.indexOf("--INSTANCE") >= 0) 
        {
          currInst = new Instance(instanceCount, nodeName);
          instanceCount++;
          instances.add(currInst);  
          
          line = bufReader.readLine();
          
          // We are dealing with a FRAME  
          while (line.indexOf("$$INSTANCE") < 0)  //means if the line is not"$$$INSTANCE ...", then keep parsing
          {
            line = parseFrame(bufReader, currInst, line);
          }
          
          if (currInst.getStackFrames().size() == 0)
          {
            
            short bpType = BlameFunction.EXPLICIT_BLAME_POINT;
            BlameFunction bf = bc.getOrCreateBlameFunction(new String("UNRESOLVED"), 
                new String("NO_MODULE"), bpType);
            
            bc.addBlamedFunction("UNRESOLVED",bf);
            
            ExitProgram ep = bf.getOrCreateEP(new String("INCOMPLETE_STACK_WALK"));
            ep.addInstance(currInst, null);


                      
          }
        }
      }
        
    }
    catch (IOException e) {
      System.err.println("exception happened - here's what I know: ");
      e.printStackTrace();
      System.exit(-1);
    }
  }
  
  
  public void print()
  {
    System.out.println("We have " + instances.size() + "instances");
    
    Iterator<Instance> itr = instances.iterator();
    while (itr.hasNext()) 
    {
      ((Instance)itr.next()).print();
    }
  }
  
  public int numInstances()
  {
    return instances.size();
  }
  
  
  public String toString()
  {
    return nodeName;
  }
  
  
}
