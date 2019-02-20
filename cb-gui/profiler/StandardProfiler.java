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
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Iterator;


public class StandardProfiler {



  public static void listFunctionsByName(ProfilerData bc) throws IOException
  {
    FileOutputStream foStream;
    PrintStream pStream;
    
    try 
    {
      foStream = new FileOutputStream("PROFILEfuncsOutByName.txt");
      pStream = new PrintStream(foStream);
      
      
      Iterator<ProfilerFunction> it = bc.getSortedFunctions(true).values().iterator();
      while (it.hasNext())
      {
        ProfilerFunction bf = (ProfilerFunction) it.next();  
        pStream.println(bf);
        
      }
    }
    catch(Exception e)
    {
      System.err.println("Could not load file!");
    }
  }
  
  
  
  public static void listFunctions(ProfilerData bc) throws IOException
  {
    
    FileOutputStream foStream;
    PrintStream pStream;
    
    try 
    {
      foStream = new FileOutputStream("PROFILEfuncsOut.txt");
      pStream = new PrintStream(foStream);
            
      Iterator<ProfilerFunction> it = bc.getSortedFunctions(false).values().iterator();
      while (it.hasNext())
      {
        ProfilerFunction bf = (ProfilerFunction) it.next();  
        pStream.println(bf);
          
        
      }
    }
    catch(Exception e)
    {
      System.err.println("Could not load file!");
    }
  }
  
  
  
  
  
  

  public static void main(String[] args) throws IOException
  {

    int numNodes = 0;
    ProfilerData bd = new ProfilerData();

    File f = new File(args[0]);
    try {
      BufferedReader bufReader = new BufferedReader(new FileReader(f));
      String line = null;

      String typeOfTest = bufReader.readLine(); // typeOfTest = 0
      Global.typeOfTest = Integer.valueOf(typeOfTest).intValue(); 
      
      Global.testProgram = new String(bufReader.readLine());
      System.out.println("Name of test program is " + Global.testProgram);

      String strNumNodes = bufReader.readLine();
      numNodes = Integer.valueOf(strNumNodes).intValue();
      
      System.out.println("Number of nodes is " + numNodes);
      Global.totalInstByNode = new int[numNodes];

      for (int a = 0; a < numNodes; a++)
      {
        line = bufReader.readLine();
        double numInstances = bd.parseNode(line); 
        Global.totalInstByNode[a] = (int)numInstances;
        Global.totalInstances += (int)numInstances;
      }
    }
    catch (IOException e) {
      System.err.println("exception happened - here's what I know: ");
      e.printStackTrace();
      System.exit(-1);
    }  
    
    
    listFunctions(bd);
    listFunctionsByName(bd);
  }

}
