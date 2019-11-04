add_executable(cheng12_dissertation_vehicle_counting2 
               BlobDetector.cpp
               BlobDetector.h
               ImageProcessFunctions.cpp
               ImageProcessFunctions.h
               VehicleCountingMain.cpp
)

target_link_libraries( cheng12_dissertation_vehicle_counting2 ${OpenCV_LIBRARIES} )
     
install (TARGETS cheng12_dissertation_vehicle_counting2 DESTINATION bin)
