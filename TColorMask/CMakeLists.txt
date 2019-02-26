project(TColorMask CXX)

################################################################################
# Source groups
################################################################################
set(Header_Files
    VapourSynth.h
    VSHelper.h
)
source_group("Header Files" FILES ${Header_Files})

set(Source_Files
    TColorMask.cpp
)
source_group("Source Files" FILES ${Source_Files})

set(ALL_FILES
    ${Header_Files}
    ${Source_Files}
)

################################################################################
# Target
################################################################################
add_library(${PROJECT_NAME} SHARED ${ALL_FILES})

if(NOT ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32"
     OR "${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64"))
    message(WARNING "${CMAKE_VS_PLATFORM_NAME} arch is not supported!")
endif()

################################################################################
# Target name
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    string(CONCAT TARGET_NAME
          $<$<CONFIG:Debug>:${PROJECT_NAME}>
        $<$<CONFIG:Release>:${PROJECT_NAME}>
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${PROJECT_NAME}>
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    string(CONCAT TARGET_NAME
          $<$<CONFIG:Debug>:${PROJECT_NAME}>
        $<$<CONFIG:Release>:${PROJECT_NAME}>
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${PROJECT_NAME}>
    )
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${TARGET_NAME})

################################################################################
# Output directory
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    string(CONCAT OUT_DIR
          $<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}//Build/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>/>
        $<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}//Build/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>/>
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${CMAKE_SOURCE_DIR}/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>>
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    string(CONCAT OUT_DIR
          $<$<CONFIG:Debug>:${CMAKE_SOURCE_DIR}//Build/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>/>
        $<$<CONFIG:Release>:${CMAKE_SOURCE_DIR}//Build/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>/>
        $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>>>:${CMAKE_SOURCE_DIR}/${CMAKE_VS_PLATFORM_NAME}/$<CONFIG>>
    )
endif()
set(ARCHIVE_OUT_DIR ${OUT_DIR})
set(ARCHIVE_OUT_NAME ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_NAME ${ARCHIVE_OUT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${ARCHIVE_OUT_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${OUT_DIR})

################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
          "$<$<CONFIG:Debug>:_WINDLL;"
                            "_MBCS"
                            ">"
        "$<$<CONFIG:Release>:_WINDLL;"
                            "_MBCS"
                            ">"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
          "$<$<CONFIG:Debug>:_WINDLL;"
                            "_MBCS"
                            ">"
        "$<$<CONFIG:Release>:_WINDLL;"
                            "_MBCS"
                            ">"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        target_compile_options(${PROJECT_NAME} PRIVATE
              $<$<CONFIG:Debug>:/Od;/MDd;/W3;/EHsc>
            $<$<CONFIG:Release>:/GL;/O2;/Oi;/MD;/W3;/EHsc>
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
              $<$<CONFIG:Debug>:/Od;/MDd;/W3;/EHsc>
            $<$<CONFIG:Release>:/GL;/O2;/Oi;/MD;/W3;/EHsc>
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "Win32")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_DEBUG "/DEBUG")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_RELEASE "/NODEFAULTLIB:libmmt.lib /LTCG /DEBUG")
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_DEBUG "/DEBUG")
        set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_RELEASE "/NODEFAULTLIB:libmmt.lib /LTCG /DEBUG")
    endif()
endif()

