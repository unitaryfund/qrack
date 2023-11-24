option (ENABLE_QBDT "Include QBinaryDecisionTree API layer (on by default)" ON)

if (ENABLE_QBDT)
    target_sources (qrack PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/src/qbdt/node_interface.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/src/qbdt/node.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/src/qbdt/tree.cpp
        )
endif (ENABLE_QBDT)
