# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/sqg/下载/clion-2018.3.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/sqg/下载/clion-2018.3.4/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/BALProblem.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/BALProblem.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BALProblem.dir/flags.make

CMakeFiles/BALProblem.dir/common/BALProblem.cpp.o: CMakeFiles/BALProblem.dir/flags.make
CMakeFiles/BALProblem.dir/common/BALProblem.cpp.o: ../common/BALProblem.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/BALProblem.dir/common/BALProblem.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/BALProblem.dir/common/BALProblem.cpp.o -c /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/common/BALProblem.cpp

CMakeFiles/BALProblem.dir/common/BALProblem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/BALProblem.dir/common/BALProblem.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/common/BALProblem.cpp > CMakeFiles/BALProblem.dir/common/BALProblem.cpp.i

CMakeFiles/BALProblem.dir/common/BALProblem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/BALProblem.dir/common/BALProblem.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/common/BALProblem.cpp -o CMakeFiles/BALProblem.dir/common/BALProblem.cpp.s

# Object files for target BALProblem
BALProblem_OBJECTS = \
"CMakeFiles/BALProblem.dir/common/BALProblem.cpp.o"

# External object files for target BALProblem
BALProblem_EXTERNAL_OBJECTS =

libBALProblem.so: CMakeFiles/BALProblem.dir/common/BALProblem.cpp.o
libBALProblem.so: CMakeFiles/BALProblem.dir/build.make
libBALProblem.so: CMakeFiles/BALProblem.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libBALProblem.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BALProblem.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BALProblem.dir/build: libBALProblem.so

.PHONY : CMakeFiles/BALProblem.dir/build

CMakeFiles/BALProblem.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BALProblem.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BALProblem.dir/clean

CMakeFiles/BALProblem.dir/depend:
	cd /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/cmake-build-debug /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/cmake-build-debug /home/sqg/Vision_Lab/SLAM/slambook_gaoxiang/ch10/ceres_custombundle/cmake-build-debug/CMakeFiles/BALProblem.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BALProblem.dir/depend

