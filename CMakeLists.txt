# Minimum version of CMake required. Don't touch.
cmake_minimum_required(VERSION 2.8)

# Set the name of your project
project(SSCD)

# Include macros from the SBEL utils library
include(SBELUtils.cmake)

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

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	enable_openmp_support()
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
ENDIF()

cuda_add_library(sscd SHARED Broadphase.cu Broadphase.h Narrowphase.cu Narrowphase.h AABBGenerator.cu AABBGenerator.h sscd.h sscdMath.h SupportFunctions.cuh)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	FIND_LIBRARY(CUDA_FRAMEWORK CUDA)
	SET_TARGET_PROPERTIES( sscd PROPERTIES XCODE_ATTRIBUTE_ENABLE_OPENMP_SUPPORT YES )
	target_link_libraries(sscd ${CUDA_FRAMEWORK})
ENDIF()

# Example CUDA program
# target_link_libraries(sscd 
# 	${GLUT_LIBRARIES}
# 	${FREEGLUT_LIB}
# 	${OPENGL_LIBRARIES}
# 	${ARMADILLO_LIBRARIES}
# 	)
# cuda_add_executable(SSCD main.cu)
# target_link_libraries(SSCD sscd)
