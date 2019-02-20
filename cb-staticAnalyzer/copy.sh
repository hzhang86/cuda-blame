#!/bin/bash
echo "Copy everything from build to llvm and backup BFC"
"cp" ./* ../../../../../BFC/ -r
"cp" ./* ../../../../../llvm/lib/Transforms/BFC -r
