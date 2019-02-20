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

public class MapValueSort {
	
    /** inner class to do sorting of the map **/

    private static class ValueComparer implements Comparator<String> {
        private final Map<String, BlameFunction> map;

        public ValueComparer(Map<String, BlameFunction> map) {
            this.map = map;
        }

        /** Compare to values of a Map */
        public int compare(String key1, String key2) {
        	BlameFunction value1 = (BlameFunction) this.map.get(key1);
        	BlameFunction value2 = (BlameFunction) this.map.get(key2);
        	
        
            int c = value2.compareTo(value1);
            if (c != 0) {
                return c;
            }
            Integer hashCode1 = key1.hashCode();
            Integer hashCode2 = key2.hashCode();
            return hashCode1.compareTo(hashCode2);
        }

    }

    /** Sorts a given Map according to its values and returns a sortedmap */
    public static SortedMap<String, BlameFunction> getValueSortedMap(Map<String, BlameFunction> map) {
        Comparator<String> vc = new MapValueSort.ValueComparer(map);
        SortedMap<String, BlameFunction> sm = new TreeMap<String, BlameFunction>(vc);
        // add all Elements of unsorted Map, otherwise it is empty                                                 
        sm.putAll(map);
        return sm;
    }

    

}
