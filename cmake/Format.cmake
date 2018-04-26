

# clang-format-5.0 -style=file -i $(FORMAT_SRC) $(FORMAT_HDRS)

find_program ( CLANG_FORMAT clang-format-5.0 )
file (GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.hpp)
list (REMOVE_ITEM ALL_SOURCE_FILES "include/common/catch.hpp")

add_custom_target (
    format
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ${CLANG_FORMAT} -style=file -i ${ALL_SOURCE_FILES}
    )

