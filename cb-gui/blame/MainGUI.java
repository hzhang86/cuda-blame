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

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.util.HashMap;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.Collections;
import java.util.Iterator;
import java.util.HashSet;
import java.util.Vector;


import javax.swing.JFrame;
import javax.swing.JTabbedPane;
import javax.swing.UIManager;
import javax.swing.JComponent;


import javax.swing.JLabel;
import javax.swing.JPanel;
import java.awt.GridLayout;


public class MainGUI {

  protected static JComponent makeTextPanel(String text) {
        JPanel panel = new JPanel(false);
        JLabel filler = new JLabel(text);
        filler.setHorizontalAlignment(JLabel.CENTER);
        panel.setLayout(new GridLayout(1, 1));
        panel.add(filler);
        return panel;
    }

  
  
  public static void launchGUI(BlameContainer bc, profiler.ProfilerData pd) throws IOException 
  {
    // Window for showing all nodes (and the instances) that
    // occurred during a blame run, this will then change
    // what is in the BlameSourceWindow and the BlameExitVarWindow
    
    JTabbedPane tabbedPane = new JTabbedPane();
  
        // Three different windows, for Hybrid, Data-centric, Code-centric respectively
    BlameTreeBlamePoints bt = new BlameTreeBlamePoints(bc);
    BlameTreeDataCentric btdc = new BlameTreeDataCentric(bc);
    ProfileTree pt = new ProfileTree(pd);
    
    
    //JComponent panel2 = makeTextPanel("Panel #2");
    //tabbedPane.addTab("Full Data Centric", panel2);
    
    tabbedPane.addTab("Full Data Centric", btdc);
    tabbedPane.addTab("Blame Points (Data/Code Hybrid)", bt);
    //JComponent panel3 = makeTextPanel("Panel #3");
    //tabbedPane.addTab("Full Code Centric", panel3);
    
    tabbedPane.addTab("Full Code Centric", pt);
    
    BlameNodeWindow   bnw = new BlameNodeWindow(bc);
    VariableMetaData vmd = new VariableMetaData();
    
    final boolean useSystemLookAndFeel = false;

    if (useSystemLookAndFeel) {
        try {
            UIManager.setLookAndFeel(
                UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            System.err.println("Couldn't use system look and feel.");
        }
    }

    //Create and set up the window.
    JFrame frame;
    if (Global.exclusive_blame)
        frame = new JFrame("Exclusive Blame");
    else 
        frame = new JFrame("Inclusive Blame"); //The usual GUI  
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    

    //Add content to the window.
    //frame.add(new BlameTree(bc));
    frame.add(tabbedPane);
    
    //Display the window.
    frame.pack();
    frame.setVisible(true);

//bt.createAndShowGUI();
    
//    Window for showing the source files blame touches
    BlameSourceWindow bsw = new BlameSourceWindow(bc, bnw, vmd, bt, btdc);
    bsw.setVisible(true);  
    
    
    /*
    VariableFilter vf = new VariableFilter();
    vf.setVisible(true);
    
      JFrame vfframe = new JFrame("CheckBoxDemo");
        vfframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //Create and set up the content pane.
        JComponent newContentPane = new VariableFilter();
        newContentPane.setOpaque(true); //content panes must be opaque
        vfframe.setContentPane(newContentPane);

        //Display the window.
        vfframe.pack();
        vfframe.setVisible(true);
        */
    
    
    
    // Make it so if you close the source window, the program ends
    bsw.addWindowListener(new WindowAdapter() {
        public void windowClosing( WindowEvent e) {System.exit( 0 );}
      }
    );
    
    // Make it so if you close the node window, the program ends
    bnw.addWindowListener(new WindowAdapter() {
        public void windowClosing(WindowEvent e) {System.exit( 0 );}
      } //WindowAdaptor here acts as an inner class and 
    );    //also an abstract class that partially implements
        //the interface WindowListener(the abstract function 
        //in WindowListener: windowClosing)
  }
  
  
  public static void listFunctionsByName(BlameContainer bc) throws IOException
  {
    
    FileOutputStream foStream;
    PrintStream pStream;
    
    try 
    {
      foStream = new FileOutputStream("funcsOutByName2.txt");
      pStream = new PrintStream(foStream);
      
      //Vector<BlameFunction> coll = (Vector) bc.getAllFunctions().values();
      
      Iterator<BlameFunction> it = bc.getSortedFunctions(true).values().iterator();
      while (it.hasNext())
      {
        BlameFunction bf = (BlameFunction) it.next();  
        pStream.println(bf.getName());
        
        
        pStream.println("  Exit Variables:");
        Iterator<ExitVariable> it_ev = bf.getExitVariables().values().iterator();
        while (it_ev.hasNext())
        {
          ExitVariable ev = it_ev.next();
          pStream.println("  " + ev);
          
        }
        pStream.println();
        
        pStream.println("  Exit Variable Fields:");
        Iterator<ExitVariable> it_evf = bf.getExitVarFields().values().iterator();
        while (it_evf.hasNext())
        {
          ExitVariable ev = it_evf.next();
          pStream.println("  " + ev);
          
        }
        pStream.println();    
        
        
        pStream.println("  Local Variables:");
        Iterator<ExitProgram> it_lv = bf.getExitPrograms().values().iterator();
        while (it_lv.hasNext())
        {
          ExitProgram ev = it_lv.next();
          pStream.println("  " + ev);
          
        }
        pStream.println();              
        
        
        pStream.println("  Local Variable Fields:");
        Iterator<ExitProgram> it_lvf = bf.getExitProgFields().values().iterator();
        while (it_lvf.hasNext())
        {
          ExitProgram ev = it_lvf.next();
          pStream.println("  " + ev);
          
        }
        pStream.println();              
        
      }

      
      
      
    }
    catch(Exception e)
    {
      System.err.println("Could not load file!");
    }
  }
  
  
  
  public static void listFunctions(BlameContainer bc) throws IOException
  {
    
    FileOutputStream foStream;
    PrintStream pStream;
    
    try 
    {
      foStream = new FileOutputStream("funcsOut2.txt");
      pStream = new PrintStream(foStream);
      
      //Vector<BlameFunction> coll = (Vector) bc.getAllFunctions().values();
      
      Iterator<BlameFunction> it = bc.getSortedFunctions(false).values().iterator();
      while (it.hasNext())
      {
        BlameFunction bf = (BlameFunction) it.next();  
        pStream.println(bf.getName());
        
        
        pStream.println("  Exit Variables:");
        Iterator<ExitVariable> it_ev = bf.getExitVariables().values().iterator();
        while (it_ev.hasNext())
        {
          ExitVariable ev = it_ev.next();
          pStream.println("  " + ev);
          
        }
        pStream.println();
        
        pStream.println("  Exit Variable Fields:");
        Iterator<ExitVariable> it_evf = bf.getExitVarFields().values().iterator();
        while (it_evf.hasNext())
        {
          ExitVariable ev = it_evf.next();
          pStream.println("  " + ev);
          
        }
        pStream.println();    
        
        
        pStream.println("  Local Variables:");
        Iterator<ExitProgram> it_lv = bf.getExitPrograms().values().iterator();
        while (it_lv.hasNext())
        {
          ExitProgram ev = it_lv.next();
          pStream.println("  " + ev);
          
        }
        pStream.println();              
        
        
        pStream.println("  Local Variable Fields:");
        Iterator<ExitProgram> it_lvf = bf.getExitProgFields().values().iterator();
        while (it_lvf.hasNext())
        {
          ExitProgram ev = it_lvf.next();
          pStream.println("  " + ev);
          
        }
        pStream.println();              
        
      }
      
    }
    catch(Exception e)
    {
      System.err.println("Could not load file!");
    }
  }
  
  

  public static void listFunctionsByName(profiler.ProfilerData bc) throws IOException
  {
    FileOutputStream foStream;
    PrintStream pStream;
    
    try 
    {
      foStream = new FileOutputStream("PROFILEfuncsOutByName2.txt");
      pStream = new PrintStream(foStream);
      
      
      Iterator<profiler.ProfilerFunction> it = bc.getSortedFunctions(true).values().iterator();
      while (it.hasNext())
      {
        profiler.ProfilerFunction bf = (profiler.ProfilerFunction) it.next();  
        pStream.println(bf);
        
      }
    }
    catch(Exception e)
    {
      System.err.println("Could not load file!");
    }
  }
  
  
  
  public static void listFunctions(profiler.ProfilerData bc) throws IOException
  {
    
    FileOutputStream foStream;
    PrintStream pStream;
    
    try 
    {
      foStream = new FileOutputStream("PROFILEfuncsOut2.txt");
      pStream = new PrintStream(foStream);
            
      Iterator<profiler.ProfilerFunction> it = bc.getSortedFunctions(false).values().iterator();
      while (it.hasNext())
      {
        profiler.ProfilerFunction bf = (profiler.ProfilerFunction) it.next();  
        pStream.println(bf);
          
        
      }
    }
    catch(Exception e)
    {
      System.err.println("Could not load file!");
    }
  }
  
  
  public static void outputVarsRecursive(ExitSuper es, PrintStream pStream, String prefix) throws IOException
  {
    // comments use PROFILE_Vars_Out.txt in helloblame3 as example
    String fullVal = es.toString(); //j [ Int ]  main 27.48%
    fullVal += " " + es.getParentBF().getName(); // main
    fullVal += " " + es.getParentBF().getModuleName(); // helloBlame3.c
    
    pStream.print(prefix);
    pStream.println(fullVal);
    
    if (es.getFields().size() > 0)
    {
      prefix += fullVal.subSequence(0, fullVal.indexOf(' '));
      prefix += "->";

      //System.out.println(prefix);
      for (ExitSuper field : es.getFields().values())
      {
        outputVarsRecursive(field, pStream, prefix);
      }
    }
  }
  
  public static boolean hardCodedFilter(String s, String[] hlVars)
  {
    //System.out.print(s);
    
    if (s.startsWith("ksp "))
    {
      hlVars[0] = s;
      return true;
    }
    else if  (s.startsWith("ksp->pc "))
    {
      hlVars[1] = s;
      return true;
    }
    else if (s.startsWith("ksp->gmres(data) "))
    {
      hlVars[5] = s;
      return true;
     }
    else if (s.startsWith("ksp->work "))
    {
      hlVars[4] = s;
      return true;
     }
    else if (s.startsWith("Amat"))
    {
      hlVars[7] = s;
      return true;
    }
    else if (s.startsWith("ksp->pc->jac(data) "))
    {
      hlVars[8] = s;
      return true;
    }
    else if (s.startsWith("ksp->pc->osm(data) "))
    {
      hlVars[9] = s;
      return true;
    }
    else if (s.startsWith("ksp->pc->pmat "))
    {
      hlVars[10] = s;
      System.out.println(s + " --- true ");
      return true;
    }
    else if (s.contains("ksp->pc") && s.contains("vec_sol "))
    {
      hlVars[6] = s;
      //System.out.println(s + " --- true(2)");
      return true;
    }
    
    //System.out.println(" ----------- false");
    return false;
  }
  
  
  public static boolean hardCodedFilter2(String s, String[] hlVars)
  {
    //System.out.print(s);
    
    if (s.startsWith("simulation "))
    {
      hlVars[0] = s;
      return true;
    }
    else if  (s.startsWith("simulation->flow_stepper "))
    {
      hlVars[1] = s;
      return true;
    }
    else if (s.startsWith("simulation->realization "))
    {
      hlVars[2] = s;
      return true;
     }
    else if (s.startsWith("simulation->tran_stepper "))
    {
      hlVars[3] = s;
      return true;
     }
    else if (s.startsWith("simulation->flow_stepper->solver "))
    {
      hlVars[4] = s;
      return true;
    }
    else if (s.startsWith("simulation->flow_stepper->solver->snes "))
    {
      hlVars[5] = s;
      return true;
    }
    else if (s.startsWith("simulation->flow_stepper->solver->snes->ksp "))
    {
      hlVars[6] = s;
      return true;
    }
    else if (s.startsWith("simulation->flow_stepper->solver->snes->ksp->pc "))
    {
      hlVars[7] = s;
      return true;
    }
    else if (s.startsWith("simulation->flow_stepper->solver->snes->ksp->work "))
    {
      hlVars[8] = s;
      return true;
    }
    else if (s.startsWith("simulation->flow_stepper->solver->snes->ksp->pc->osm(data) "))
    {
      hlVars[9] = s;
      //System.out.println(s + " --- true(2)");
      return true;
    }
    else if (s.startsWith("simulation->tran_stepper->solver "))
    {
      hlVars[10] = s;
      return true;
    }
    else if (s.startsWith("simulation->tran_stepper->solver->snes "))
    {
      hlVars[11] = s;
      return true;
    }
    else if (s.startsWith("simulation->tran_stepper->solver->snes->ksp "))
    {
      hlVars[12] = s;
      return true;
    }
    else if (s.startsWith("simulation->tran_stepper->solver->snes->ksp->work "))
    {
      hlVars[13] = s;
      return true;
    }
    else if (s.startsWith("simulation->tran_stepper->solver->snes->ksp->pc "))
    {
      hlVars[14] = s;
      return true;
    }
    else if (s.startsWith("simulation->tran_stepper->solver->snes->ksp->pc->osm(data) "))
    {
      hlVars[15] = s;
      //System.out.println(s + " --- true(2)");
      return true;
    }
    
    //System.out.println(" ----------- false");
    return false;
  }
  
  
  
  public static void outputHLVarsRecursive(ExitSuper es, String[] hlVars, String prefix) throws IOException
  {
    String fullVal = es.toString();
    
    /*
    if (hardCodedFilter(prefix+fullVal,hlVars))
    {
      pStream.print(prefix);
      pStream.println(fullVal);
    }
    */
    hardCodedFilter2(prefix+fullVal,hlVars);
    
    
    if (es.getFields().size() > 0)
    {
      prefix += fullVal.subSequence(0, fullVal.indexOf(' '));
      prefix += "->";

      //System.out.println(prefix);
      for (ExitSuper field : es.getFields().values())
      {
        outputHLVarsRecursive(field, hlVars, prefix);
      }
    }
    
  }
  
  
  public static void populateRecursiveES(ExitSuper es, String prefix, Vector<ExitSuper> flatVec) 
  {
    String fullVal = es.toString();
    String name = es.getName();
    es.setFullHierName(prefix + name);
    
    //pStream.print(prefix);
    //pStream.println(fullVal);
    
    flatVec.add(es);
    
    if (es.getFields().size() > 0)
    {
      prefix += fullVal.subSequence(0, fullVal.indexOf(' '));
      prefix += "->";

      //System.out.println(prefix);
      for (ExitSuper field : es.getFields().values())
      {
        populateRecursiveES(field, prefix, flatVec);
      }
    }
    
    
  }
  
  
  public static void populateNewVector(Vector<ExitSuper> locals, Vector<ExitSuper> flatVec)
  {
    String prefix = new String();
    
    for (ExitSuper es: locals)
      populateRecursiveES(es, prefix, flatVec);
  }
  
  

  
  public static void outputVariables(BlameContainer bc) throws IOException
  {
    FileOutputStream foStream, foStream2, foStream3;
    PrintStream pStream, pStream2, pStream3;
    
    String[] hlVars = new String[20];
    
    try 
    {
      foStream = new FileOutputStream(Global.rootDir + "PROFILE_Vars_Out.txt");
      pStream = new PrintStream(foStream);
      
      foStream3 = new FileOutputStream(Global.rootDir + "PROFILE_Vars_Top100_Out.txt");
      pStream3 = new PrintStream(foStream3);
            
      foStream2 = new FileOutputStream(Global.rootDir + "PROFILE_Vars_HL_Out.txt");
      pStream2 = new PrintStream(foStream2);
      
      
      String prefix = new String();
      
      
      Vector<ExitSuper> flatViewVector = new Vector<ExitSuper>();
      
      populateNewVector(bc.getAllLocalVariables(), flatViewVector);
      
      Collections.sort(bc.getAllLocalVariables());
      Collections.sort(flatViewVector);
      
      for (ExitSuper es : bc.getAllLocalVariables())
        outputVarsRecursive(es, pStream, prefix);
      
      for (ExitSuper es: bc.getAllLocalVariables())
        outputHLVarsRecursive(es, hlVars, prefix);
      
      
      int counter = 0;
      for (ExitSuper es: flatViewVector)
      {
        if (counter > 100)
          break;
        
        counter++;
        pStream3.print(es.getFullHierName() + " ");
        pStream3.println(es);
      }
        
      for (int a = 0; a < 16; a++)
      {
        pStream2.println(hlVars[a]);
      }

      pStream.close();
      pStream2.close();
      
      foStream.close();
      foStream2.close();
      
    }
    catch (Exception e)
    {
      System.err.println("Could not load file!");
    }
    
  }
  
  
  
  
  
  public static void outputReadVariables(BlameContainer bc) throws IOException
  {
    FileOutputStream foStream, foStream3;
    PrintStream pStream, pStream3;
    
    String[] hlVars = new String[20];
    
    try 
    {
      foStream = new FileOutputStream(Global.rootDir + "PROFILE_Vars_Out.txt");
      pStream = new PrintStream(foStream);
      
      foStream3 = new FileOutputStream(Global.rootDir + "PROFILE_Vars_Top100_Out.txt");
      pStream3 = new PrintStream(foStream3);
            
      
      String prefix = new String();
      
      
      Vector<ExitSuper> flatViewVector = new Vector<ExitSuper>();
      
      populateNewVector(bc.getAllLocalVariables(), flatViewVector);
      
      Collections.sort(bc.getAllLocalVariables());
      Collections.sort(flatViewVector);
      
      for (ExitSuper es : bc.getAllLocalVariables())
        outputVarsRecursive(es, pStream, prefix);
      
      for (ExitSuper es: bc.getAllLocalVariables())
        outputHLVarsRecursive(es, hlVars, prefix);
      
      
      int counter = 0;
      for (ExitSuper es: flatViewVector)
      {
        if (counter > 100)
          break;
        
        counter++;
        pStream3.print(es.getFullHierName() + " ");
        pStream3.println(es);
      }
        
      pStream.close();    
      foStream.close();
      
    }
    catch (Exception e)
    {
      System.err.println("Could not load file!");
    }
  }
  
  
  public static void main(String args[]) throws IOException
  {
    
    // Creates new BlameContainer for all of the blame for the
    //   program across all nodes
    BlameContainer bc = new BlameContainer();  
    profiler.ProfilerData pd = new profiler.ProfilerData();


    //SourceContainer has a public static member: sourceFiles Q:why 'new' here?
    SourceContainer.sourceFiles = new HashMap<String, SourceFile>();
    
    int numNodes = 0;
    
    File f = new File(args[0]); // java.io.File f->"gui_config.txt"
  
    if (System.getenv("EXCLUSIVE_BLAME") != null)
      Global.exclusive_blame = true;
    else
      Global.exclusive_blame = false;
    
    try {
      BufferedReader bufReader = new BufferedReader(new FileReader(f));
      
      String line = null;
      
      String typeOfTest = bufReader.readLine(); // typeOfTest = 0
            //all members in Global class are static, so they can be accessed without defining a Global object
      Global.typeOfTest = Integer.valueOf(typeOfTest).intValue(); 
      Global.useMetaData = false; //TOCHECK: Originally was false
      
      Global.testProgram = new String(bufReader.readLine()); //Global.testProgram = HelloBlame3
      profiler.Global.testProgram = Global.testProgram;
      
      System.out.println("Name of test program is " + Global.testProgram);
      //The following functions not used for cuda 02/07/18
      // output all valid variable/function names in the user code
      //line = bufReader.readLine(); //.../usr
      //bc.getUsrNames(line);
      //May 08/17 store usr func names for code-centric  
      //pd.getUsrNames(line);
      //System.out.println("#of usr functions: " + pd.getAllUsrFuncNames().size());

      String strNumNodes = bufReader.readLine();
      numNodes = Integer.valueOf(strNumNodes).intValue();
      System.out.println("Number of nodes is " + numNodes);
      
      if (numNodes <= 0)
      {
        System.err.println("Not enough nodes!");
        return;
      }

      for (int a = 0; a < numNodes; a++)
      {
        if (Global.typeOfTest == 0)
        {
          line = bufReader.readLine(); //line->PARSED_pygmy
          
          if (Global.rootDir == null)
          {
            Global.rootDir = new String(line.substring(0, line.lastIndexOf('/')+1));
          }
          
          BlameData bd = null;
          
          if (Global.useMetaData)
            bd = new BlameData(line, bc);
          else
            bd = new BlameDataTrunc(line, bc);
          // the above two constructors do the same thing  
          pd.parseNode(line); 
          bc.addNode(bd); //bc declared in this main func
        }
        else if (Global.typeOfTest == 1)
        {
          Global.rootDir = new String();
          line = bufReader.readLine();
          ReadData rd = new ReadData(line, bc);
          bc.addNode(rd);
        }
      }
      
      if (Global.typeOfTest == 0)
      {
        //HashSet<String> types = bc.getAllTypes();
        //Iterator<String> it = types.iterator();
        
        //System.out.println("Types - ");
        //while (it.hasNext())
        //{
        //  System.out.println(it.next());
        //}
        
      //System.out.println("Size of LV" + bc.getAllLocalVariables().size());
      //System.out.println("Size of GV" + bc.getAllGlobalVariables().size());
        
/////////////////////////////////////////////////////////////////////////////////
       System.out.println("In typeOfTest=0, Global.useMetaData = "+Global.useMetaData);
/////////////////////////////////////////////////////////////////////////////////

        if (Global.useMetaData)
        {
          for (ExitSuper es : bc.getAllLocalVariables())
            es.calcNumInstances();
        
          for (ExitSuper es : bc.getAllGlobalVariables())
            es.calcNumInstances();
        }
        
        //for (ExitSuper es : bc.getAllLocalVariables())
          //System.out.println(es);
      }
      else if (Global.typeOfTest == 1)
      {
        //if (Global.useMetaData)
        //{
        for (ExitSuper es : bc.getAllLocalVariables())
          es.calcNumReadInstances();
        
        for (ExitSuper es : bc.getAllGlobalVariables())
          es.calcNumReadInstances();
        //}
      }
      
    }
    catch (IOException e) {
      System.err.println("exception happened - here's what I know: ");
      e.printStackTrace();
      System.exit(-1);
    }    
    
    Global.totalNodes = numNodes;
    Global.totalInstByNode = new int[numNodes];//array,one entry per node
    profiler.Global.totalInstByNode = new int[numNodes];
    
    Iterator<ParentData> it = bc.getNodes().iterator();
    int numN = 0;
    int total = 0;
    while (it.hasNext())
    {
      ParentData bd = (ParentData)it.next();

      // QUESTION: why numN not changed in every iteration ???
      Global.totalInstByNode[numN] = bd.numInstances();
      profiler.Global.totalInstByNode[numN] = bd.numInstances();

      System.out.println("Total instances for " + bd.nodeName + " is " + bd.numInstances());
      
      total += bd.numInstances();
      //////////////added by Hui//////////////////////
      numN++;
      ///////////////////////////////////////////////
    }
    Global.totalInstances = total;
    profiler.Global.totalInstances = total;
  
    if (Global.typeOfTest == 0)
    {
      
      outputVariables(bc);

      listFunctions(pd);
      listFunctionsByName(pd);
    
      listFunctions(bc);
      listFunctionsByName(bc);
    }
    
    if (Global.typeOfTest == 1)
    {
      outputVariables(bc);
    }
    
    if (args.length == 1)
      launchGUI(bc, pd);

    
  }

}

  




/*

int numStructFiles = 0;
String strNumSF = bufReader.readLine();
numStructFiles = Integer.valueOf(strNumSF).intValue();
System.out.println("Number of struct files is " + numStructFiles);

for (int a = 0; a < numStructFiles; a++)
{      
  String structFileName = bufReader.readLine();
  bc.parseStructs(structFileName);
}


int numBlameFiles = 0;

String strNumBF = bufReader.readLine();
numBlameFiles = Integer.valueOf(strNumBF).intValue();
System.out.println("Number of blame files is " + numBlameFiles);

for (int a = 0; a < numBlameFiles; a++)
{
  // Parse in the dot .blm file for line numbers
  line = bufReader.readLine();
  String staticInfoFileName = line;
  bc.augmentStaticInfo(staticInfoFileName);
}

// fills in those fields that don't have any blame associated with them
bc.fillInStructs();
*/
  
  
  

////// PROFILER

//int numNodes = 0;
//ProfilerData bd = new ProfilerData();


//File f = new File(args[0]);



//try {
  //BufferedReader bufReader = new BufferedReader(new FileReader(f));
  //String line = null;

  //Global.testProgram = new String(bufReader.readLine());
  //System.out.println("Name of test program is " + Global.testProgram);

  //String strNumNodes = bufReader.readLine();
  //numNodes = Integer.valueOf(strNumNodes).intValue();
  
  
  //System.out.println("Number of nodes is " + numNodes);
  //Global.totalInstByNode = new int[numNodes];


  //for (int a = 0; a < numNodes; a++)
  //{
    //line = bufReader.readLine();
    //int numInstances = bd.parseNode(line); 
    //Global.totalInstByNode[a] = numInstances;
    //Global.totalInstances += numInstances;
  //}
//}
//catch (IOException e) {
  //System.err.println("exception happened - here's what I know: ");
  //e.printStackTrace();
  //System.exit(-1);
//}  
  
  
///// END PROFILER

  
  
  
  
  
  
  
