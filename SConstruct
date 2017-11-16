## Compile and Build Library ##

env = Environment(CPPPATH=['include', 'src',
                           '/opt/facebook/',
                           '/opt/intel/mkl/include/',
                           '/usr/local/include/'],
                  CXXFLAGS="-std=c++11 -O3 -fopenmp")

source_files = [
                'src/Image.cc',
                'src/ImageData.cc', 'src/TDBObject.cc',
                'src/TDBImage.cc',
                'src/Exception.cc',
                'src/utils.cc',
                'src/Descriptors.cc',
                'src/DescriptorsData.cc',
                'src/DescriptorsFaiss.cc',
                'src/DescriptorsTileDB.cc',
                'src/DescriptorsTileDBDense.cc',
                'src/DescriptorsTileDBSparse.cc',
                ]

env.SharedLibrary('libvcl.so', source_files,
    LIBS = [ 'tiledb',
             'opencv_core',
             'opencv_imgproc',
             'opencv_imgcodecs',
             'gomp',
             'faiss',
             'mkl_rt', 'dl', 'pthread', 'm',
             ],

    LIBPATH = ['/usr/local/lib', '/usr/lib',
               '/opt/facebook/faiss/',
               '/opt/intel/mkl/lib/intel64/',
               ],

    LINKFLAGS="-Wl,--no-as-needed",
    )

## Compile and Run Tests ##

gtest_source = ['test/unit_tests/main_test.cc'
         , 'test/unit_tests/TDBImage_test.cc'
         , 'test/unit_tests/ImageData_test.cc'
         ,'test/unit_tests/Image_test.cc'
]

env.Program('test/unit_test', gtest_source,
        LIBS = ['vcl', 'gtest', 'pthread'
                ,'opencv_core'
                , 'opencv_imgcodecs'
                , 'opencv_highgui'
                , 'opencv_imgproc'
                , 'mkl_rt', 'dl', 'pthread', 'm',
        ],
        LIBPATH = ['.', '/usr/local/lib', '/usr/lib',
               '/opt/facebook/faiss/',
               '/opt/intel/mkl/lib/intel64/',
               ])


# # Compile and Run Tests ##

# gtest_source = ['test/unit_tests/main_test.cc'
#         , 'test/unit_tests/TDBImage_test.cc'
#         , 'test/unit_tests/ImageData_test.cc'
#        ,'test/unit_tests/Image_test.cc'
# ]

# env.Program('test/unit_test', gtest_source,
#         LIBS = ['vcl', 'gtest', 'pthread'
#                 ,'opencv_core'
#                 , 'opencv_imgcodecs'
#                 , 'opencv_highgui'
#                 , 'opencv_imgproc'
#         ],
#         LIBPATH = ['.', '/usr/lib', '/usr/local/lib'])


# test_env = Environment(CPPPATH=['include', 'src',
#         '/opt/intel/vdms/utils/include/chrono',
#         '/opt/intel/vdms/utils/include'],
#         CXXFLAGS="-std=c++11 -pg -fopenmp -O3")

# test_source = ['test/benchmarks/timing_test.cc',
#                '/opt/intel/vdms/utils/src/chrono/Chrono.cc'
#    ]

# test_env.Program('test/timing_test', test_source,
#     LIBS = ['tiledb', 'vcl'
#                ,'opencv_core'
#                , 'opencv_imgcodecs'
#                , 'opencv_highgui'
#                , 'opencv_imgproc'],
#     LIBPATH=['.', '/usr/lib', '/usr/local/lib']
#     )

# resizing = ['test/benchmarks/resizing_images.cc']

# test_env.Program('test/resize', resizing,
#     LIBS = ['tiledb', 'vcl'
#                 ,'opencv_core'
#                 , 'opencv_imgcodecs'
#                 , 'opencv_highgui'
#                 , 'opencv_imgproc'], LIBPATH=['.', '/usr/lib', '/usr/local/lib'])
