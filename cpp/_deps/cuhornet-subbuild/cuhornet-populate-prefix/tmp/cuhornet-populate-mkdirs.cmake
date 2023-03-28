# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/cuhornet-src"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/cuhornet-build"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/cuhornet-subbuild/cuhornet-populate-prefix"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/cuhornet-subbuild/cuhornet-populate-prefix/tmp"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/cuhornet-subbuild/cuhornet-populate-prefix/src/cuhornet-populate-stamp"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/cuhornet-subbuild/cuhornet-populate-prefix/src"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/cuhornet-subbuild/cuhornet-populate-prefix/src/cuhornet-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/cuhornet-subbuild/cuhornet-populate-prefix/src/cuhornet-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/cuhornet-subbuild/cuhornet-populate-prefix/src/cuhornet-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
