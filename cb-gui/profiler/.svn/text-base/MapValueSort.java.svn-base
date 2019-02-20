package profiler;

import java.util.*;

public class MapValueSort {
	
    /** inner class to do sorting of the map **/

    private static class ValueComparer implements Comparator<String> {
        private final Map<String, ProfilerFunction> map;

        public ValueComparer(Map<String, ProfilerFunction> map) {
            this.map = map;
        }

        /** Compare to values of a Map */
        public int compare(String key1, String key2) {
        	ProfilerFunction value1 = (ProfilerFunction) this.map.get(key1);
        	ProfilerFunction value2 = (ProfilerFunction) this.map.get(key2);
        	
        
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
    public static SortedMap<String, ProfilerFunction> getValueSortedMap(Map<String, ProfilerFunction> map) {
        Comparator<String> vc = new MapValueSort.ValueComparer(map);
        SortedMap<String, ProfilerFunction> sm = new TreeMap<String, ProfilerFunction>(vc);
        // add all Elements of unsorted Map, otherwise it is empty                                                 
        sm.putAll(map);
        return sm;
    }

    

}
