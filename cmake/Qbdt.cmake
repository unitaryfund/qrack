option (ENABLE_QBDT "Include QBinaryDecisionTree API layer (on by default)" ON)

if (ENABLE_QBDT)
    target_sources (qrack PRIVATE
        src/qbdt/node_interface.cpp
        src/qbdt/node.cpp
        src/qbdt/tree.cpp
        src/qbdthybrid.cpp
        )
endif (ENABLE_QBDT)
