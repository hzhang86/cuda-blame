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

public class ProfileInstance {
  
  String nodeName;
  int count;
  double share;
  int occurance;
  double weight; //weight = share * occurance
  
  String identifier;
  
  public String getNodeName() {
    return nodeName;
  }

  public void setNodeName(String nodeName) {
    this.nodeName = nodeName;
  }

  public int getCount() {
    return count;
  }

  public Double getWeight() {
    return weight;
  }  
  
  public void setCount(int count) {
    this.count = count;
  }

  public String getIdentifier() {
    return identifier;
  }

  public void setIdentifier(String identifier) {
    this.identifier = identifier;
  }

  ProfileInstance(int count, String nodeName)
  {
    this.count = count;
    this.nodeName = nodeName;
    
    identifier = nodeName + count;
  }

  ProfileInstance(int count, String nodeName, double s, int o)
  {
    this.count = count;
    this.nodeName = nodeName;
    this.share = s;
    this.occurance = o;
    this.weight = s * o;
    
    identifier = nodeName + count;
  }
}
