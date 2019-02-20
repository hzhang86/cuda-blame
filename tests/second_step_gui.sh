#!/bin/bash

echo "Run GUI, show the result !"
java -Xms512m -Xmx1024m -cp ${CUDA_BLAME_ROOT}/cb-gui blame/MainGUI gui_config.txt #>guiOut.txt 2>&1

