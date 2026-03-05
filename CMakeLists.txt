cmake_minimum_required(VERSION 3.8)
project(xplorer_mini_sysid)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ========================== Find dependencies =========================
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(Eigen3 3.4 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(xplorer_mini_cpp_utils REQUIRED)

# ========================== Library =========================
add_library(signal_generate_lib SHARED src/signal_generate_lib.cpp)
target_include_directories(signal_generate_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(signal_generate_lib PUBLIC Eigen3::Eigen)

<<<<<<< HEAD
# ========================== Python Module =========================
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS 
  xplorer_mini_sysid/km_monitor.py
  xplorer_mini_sysid/kmpc.py
  DESTINATION lib/${PROJECT_NAME}
)
=======
add_library(ocean_current_lib SHARED src/ocean_current_lib.cpp)
target_include_directories(ocean_current_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(ocean_current_lib PUBLIC xplorer_mini_cpp_utils::kinematics_utils)
>>>>>>> dbd2a0b (feat: add ocean current)

# ========================== Executable =========================
add_executable(control_mux src/control_mux.cpp)
target_include_directories(control_mux PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(control_mux 
  signal_generate_lib 
  ocean_current_lib
)

ament_target_dependencies(control_mux
  rclcpp
  std_msgs
  geometry_msgs
  std_srvs
)

# ========================== Install =========================
install(TARGETS signal_generate_lib ocean_current_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS control_mux
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# ========================== Testing =========================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
