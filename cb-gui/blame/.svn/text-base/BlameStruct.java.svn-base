package blame;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.Vector;

public class BlameStruct {
	
	String name;
	String modulePathName;
	String moduleName;
	int lineNum;
	Vector<BlameField> fields;

	public String getModulePathName() {
		return modulePathName;
	}
	public void setModulePathName(String modulePathName) {
		this.modulePathName = modulePathName;
	}
	public String getModuleName() {
		return moduleName;
	}
	public void setModuleName(String moduleName) {
		this.moduleName = moduleName;
	}
	public int getLineNum() {
		return lineNum;
	}
	public void setLineNum(int lineNum) {
		this.lineNum = lineNum;
	}
	public Vector<BlameField> getFields() {
		return fields;
	}
	public void setFields(Vector<BlameField> fields) {
		this.fields = fields;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	
	BlameStruct()
	{
		fields = new Vector<BlameField>();
		name = modulePathName = moduleName = null;
		lineNum = 0;
	}
	
	
	void parseStruct(BufferedReader bufReader)
	{
		String line;
		
		try 
		{	
			/////////
			//BEGIN
			bufReader.readLine();			
			name = bufReader.readLine();
			System.out.println("Looking at Struct " + name);
			
			// END
			bufReader.readLine();
				
			/////////
			//BEGIN
			bufReader.readLine();	
			modulePathName = bufReader.readLine();
			// END
			bufReader.readLine();
			
			/////////
			//BEGIN
			bufReader.readLine();	
			moduleName = bufReader.readLine();
			// END
			bufReader.readLine();
			
			/////////
			//BEGIN
			bufReader.readLine();	
			String strLineNum = bufReader.readLine();
			lineNum = Integer.valueOf(strLineNum).intValue();
			// END
			bufReader.readLine();
			
			// BEGIN FIELDS
			bufReader.readLine();
			
			// BEGIN FIELD (or END FIELDS)
			line = bufReader.readLine();
			
			while (line.indexOf("BEGIN FIELD") >= 0 )
			{
				// BEGIN F_NUM
				line = bufReader.readLine();
				if (line.indexOf("END FIELD") >= 0)
				{
					bufReader.readLine();
					continue;
				}
				
				// FIELD NUM

				line = bufReader.readLine();
				int fNum = Integer.valueOf(line).intValue();
				
				// END F_NUM
				line = bufReader.readLine();
				
				// BEGIN F_NAME
				line = bufReader.readLine();
				String fName = bufReader.readLine();
				// END F_NAME
				line = bufReader.readLine();
				
				BlameField bf = new BlameField(fName, fNum);
				fields.add(bf);
				
				// END FIELD
				line = bufReader.readLine();
				
				// BEGIN FIELD (or END FIELDS)
				line = bufReader.readLine();
			
			}
			
			// END STRUCT 
			bufReader.readLine();
			
		}
		catch(IOException ioe)
		{
			System.err.println("IO Error!");
			ioe.printStackTrace();
			System.exit(-1);
		}
	}
	
	

}
