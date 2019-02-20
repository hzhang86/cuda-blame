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


public class ExitOutput extends ExitSuper {

	/*
	public double compareTo(ExitOutput o) {
        return o.numInstances() - this.numInstances();
    }
	
	
	public double compareTo(ExitProgram o) {
        return o.numInstances() - this.numInstances();
    }
    */
	
	ExitOutput(String n)
	{
		super();
		name = n;
		/*
		//varInstances = new Vector<VariableInstance>();
		nodeInstances = new HashMap<String, NodeInstance>();
		isField = false;
		//lineNumsByFunc = new HashMap<String, HashSet<Integer>>();
		hierName = null;
		eStatus = null;
		structType = null;
		
		lineNums = new HashSet<Integer>();
		*/
	}
	
	public String printDescription()
	{
		String s = new String("Exit Output - " + super.printDescription());
		return s;
	}
	
}
