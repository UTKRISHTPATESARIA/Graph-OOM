# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-src"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-build"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/tmp"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src"
  "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-subbuild/rapids-cmake-populate-prefix/src/rapids-cmake-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
