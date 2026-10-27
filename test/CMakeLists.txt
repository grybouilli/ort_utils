FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.16.1)
FetchContent_MakeAvailable(catch)

set(unit_tests
  tests-session-handler.cpp
  tests-tensor-buffer.cpp
)

add_executable(${CMAKE_PROJECT_NAME}_testlib ${unit_tests})
target_compile_features(${CMAKE_PROJECT_NAME}_testlib PRIVATE cxx_std_14)
set_target_properties(${CMAKE_PROJECT_NAME}_testlib PROPERTIES CXX_EXTENSIONS OFF)

target_link_libraries(${CMAKE_PROJECT_NAME}_testlib PRIVATE ${CMAKE_PROJECT_NAME} Catch2::Catch2WithMain)

add_test(NAME test_hello COMMAND ${CMAKE_PROJECT_NAME}_testlib)
