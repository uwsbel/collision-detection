# Minimum version of CMake required. Don't touch.
cmake_minimum_required(VERSION 2.8)

# Set the name of your project
project(collision_detection)

# Include macros from the SBEL utils library
include(cmake/ParallelUtils.cmake)

##########################################################
# If using Makefiles, be sure to set CMAKE_BUILD_TYPE to # 
# select different build configurations.                 #
# Possible values:                                       #
# - RELEASE - fully optimized for speed                  #
# - DEBUG - unoptimized, with debugging symbols          #
# - RELWITHDEBUGINFO - optimized, with debugging symbols #
# - MINSIZEREL - optimized for size                      #
# - [not set] - unoptimized                              #
##########################################################

enable_cuda_support()
enable_openmp_support()
IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	
ENDIF()

INCLUDE_DIRECTORIES(
${CMAKE_SOURCE_DIR}
${CMAKE_SOURCE_DIR}/src
${CMAKE_SOURCE_DIR}/src/aabb
${CMAKE_SOURCE_DIR}/src/broadphase
${CMAKE_SOURCE_DIR}/src/narrowphase)

cuda_add_library(collision_detection SHARED 
	src/broadphase/Broadphase.cu 
	src/broadphase/Broadphase.cuh 
	src/narrowphase/Narrowphase.cu 
	src/narrowphase/Narrowphase.cuh 
	src/aabb/AABBGenerator.cu 
	src/aabb/AABBGenerator.cuh 
	src/sscd.h 
	includes/gpu_math.h 
	src/SupportFunctions.cuh)



IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	FIND_LIBRARY(CUDA_FRAMEWORK CUDA)
	SET_TARGET_PROPERTIES( collision_detection PROPERTIES XCODE_ATTRIBUTE_ENABLE_OPENMP_SUPPORT YES )
	target_link_libraries(collision_detection ${CUDA_FRAMEWORK})
ENDIF()

##########################################################
#                                                        # 
# Tests                                                  #
#                                                        #
##########################################################
cuda_add_executable(test_1 test/test1.cu)
target_link_libraries(test_1 collision_detection)
