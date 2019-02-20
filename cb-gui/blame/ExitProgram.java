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

import java.util.*;
//EPs are local variables
public class ExitProgram extends ExitSuper  {

	// Is this a random local variable (found for all) or is this one from
	// an implicit/explicit blame point that has the blame from calls below
	// it in the call chain
	boolean isBlamePoint;
	
	public boolean isBlamePoint() {
		return isBlamePoint;
	}

	public void setBlamePoint(boolean isBlamePoint) {
		this.isBlamePoint = isBlamePoint;
	}

	/*
	public double compareTo(ExitProgram o) {
        return o.numInstances() - this.numInstances();
    }
    */
	
	ExitProgram(String n)
	{
		super();
		hierName = n;
		name = n;
		isBlamePoint = true;
	}
	
	public void getOrCreateField(String s)
	{
		
	}
	
	public void addField(String s, BlameFunction bf)
	{
		HashMap<String, ExitProgram> eVFields = bf.getExitProgFields();
		ExitSuper es = eVFields.get(s);
		if (es != null)
		{
			//int lastDot = es.getHierName().lastIndexOf('.');
			//lastDot++;
			
			//String truncHierName = es.getHierName().substring(lastDot);
			fields.put(es.getName(), es);
			//es.setName(truncHierName);
		}
		else
		{
			System.out.println("EP addField NULL for " + s);
		}
	}
	
	public String printDescription()
	{
		String s = new String("Non Exit Variable - " + super.printDescription());
		return s;
	}
}
