Difference between cuptiActivity.so and inst_sampling.so

cuptiActivity.so is built with __attribute((constructor)), so it's
used with: LD_PRELOAD=.../cuptiActivity.so

libinst_sampling.so is simply a shared library, so we need to insert
function calls from it inside the source code.

Functionality is exactly the same
