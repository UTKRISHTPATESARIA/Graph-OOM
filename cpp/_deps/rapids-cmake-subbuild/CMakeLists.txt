# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.25.0)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(rapids-cmake-populate NONE)



include(ExternalProject)
ExternalProject_Add(rapids-cmake-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "FALSE" "URL" "https://github.com/rapidsai/rapids-cmake/archive/refs/heads/branch-23.02.zip"
                    SOURCE_DIR          "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-src"
                    BINARY_DIR          "/home/ankit/rapids/cugraph/cugraph/cpp/_deps/rapids-cmake-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


