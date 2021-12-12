option (ENABLE_QBDT "Include QBinaryDecisionTree API layer (on by default)" ON)

if (ENABLE_QBDT)
    target_sources (qrack PRIVATE
        src/qbinary_decision_tree/node.cpp
        src/qbinary_decision_tree/tree.cpp
        )
endif (ENABLE_QBDT)
