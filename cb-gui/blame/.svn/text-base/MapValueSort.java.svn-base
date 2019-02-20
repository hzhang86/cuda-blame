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
