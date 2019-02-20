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

			Global.testProgram = new String(bufReader.readLine());
			System.out.println("Name of test program is " + Global.testProgram);

			String strNumNodes = bufReader.readLine();
			numNodes = Integer.valueOf(strNumNodes).intValue();
			
			
			System.out.println("Number of nodes is " + numNodes);
			Global.totalInstByNode = new int[numNodes];


			for (int a = 0; a < numNodes; a++)
			{
				line = bufReader.readLine();
				int numInstances = bd.parseNode(line); 
				Global.totalInstByNode[a] = numInstances;
				Global.totalInstances += numInstances;
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
