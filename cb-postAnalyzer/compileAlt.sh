#!/bin/bash
rm *.o -f
g++ -c -g -I$BOOST_ROOT/include -I$DYNINST_INSTALL/include  *.cpp -std=gnu++11  # there was -pg in three g++ cmds
echo "Finish compiling"
#g++ -c -g  altMain.cpp -std=gnu++11
g++ -g -o AltParser *.o -L$DYNINST_INSTALL/lib -ldyninstAPI -std=gnu++11
echo "Finish linking"
