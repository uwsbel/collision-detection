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

# include CUSP
SET(CUSP_INC_DIR "" CACHE PATH "Where is the CUSP include directory?")
include_directories(${CUSP_INC_DIR})

## Example MPI program
#enable_mpi_support()
#add_executable(integrate_mpi integrate_mpi.cpp)
#target_link_libraries(integrate_mpi ${MPI_CXX_LIBRARIES})

# Example OpenMP program
enable_openmp_support()
#add_executable(integrate_omp integrate_omp.cpp)
include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)
include_directories(${CUDA_SDK_ROOT_DIR}/../shared/inc)

# Using Armadillo
find_package( Armadillo REQUIRED )
include_directories(${ARMADILLO_INCLUDE_DIRS})

# include OpenGL
find_package("GLUT")
find_package("OpenGL")
	#freeglut is not found properly on windows
	IF ("${CH_OS}" STREQUAL "WINDOWS")
		SET(FREEGLUT_LIB_PATH       ""    	CACHE PATH "Where is freeglut located?")		 	
		FIND_LIBRARY( FREEGLUT_LIB NAMES glut glut32 freeglut PATHS ${FREEGLUT_LIB_PATH})
	ENDIF()
	MARK_AS_ADVANCED(FORCE FREEGLUT_LIB)

# Example CUDA program
enable_cuda_support()
cuda_add_library(sscd SHARED CollisionDetector.cu sscd.h)
target_link_libraries(sscd 
	${GLUT_LIBRARIES}
	${FREEGLUT_LIB}
	${OPENGL_LIBRARIES}
	${ARMADILLO_LIBRARIES}
	)
cuda_add_executable(SSCD main.cu)
target_link_libraries(SSCD sscd)
