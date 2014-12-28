################################################################################################
# OpenGL interop support for OpenCV
#
# http://docs.opencv.org/ref/master/df/d60/namespacecv_1_1ogl.html
# "opengl.hpp"
################################################################################################

# namespace cv::ogl

# Default constructor
oglArrays() = @cxx cv::ogl::Arrays()

# Binds all vertex arrays
oglBind(oglArr) = @cxx oglArr->bind()

# Release all inner buffers
oglRelease(oglArr) = @cxx oglArr->release()

# Resets vertex colors
oglResetColor(oglArr) = @cxx oglArr->resetColorArray()

# Resets vertex normals
oglResetNormal(oglArr) = @cxx oglArr->resetNormalArray()

# Resets vertex texture coordinates
oglResetTexCoord(oglArr) = @cxx oglArr->resetTexCoordArray()

# Resets vertex coordinates
oglResetVertex(oglArr) = @cxx oglArr->resetVertexArray()

# Sets auto release mode all inner buffers
oglSetAutoRelease(oglArr, flag::Bool) = @cxx oglArr->setAutoRelease(flag)

# Sets an array of vertex colors
oglSetColor(oglArr, color) = @cxx oglArr->setColorArray(color)
# color::InputArray, can be both host and device memory

# Sets an array of vertex normals
oglSetNormal(oglArr, normal) = @cxx oglArr->setNormalArray(normal)
# normal array with vertex normals, can be both host and device memory

# Sets an array of vertex texture coordinates
oglSetTexCoord(oglArr, texCoord) = @cxx oglArr->setTexCoordArray(texCoord)
# texCoord	array with vertex texture coordinates, can be both host and device memory

# Sets an array of vertex coordinates
oglSetVertexArray(oglArr, vertex) = @cxx oglArr->setVertexArray(vertex)
# vertex array with vertex coordinates, can be both host and device memory.

# Returns the vertex count
oglSize(oglArr) = @cxx oglArr->size()

# OpenGL Buffers
# Buffer Objects are OpenGL objects that store an array of unformatted memory allocated by the OpenGL context.
# These can be used to store vertex data, pixel data retrieved from images or the framebuffer, and a variety of other things.
# ogl::Buffer has interface similar with Mat interface and represents 2D array memory.
# ogl::Buffer supports memory transfers between host and device and also can be mapped to CUDA memory.


# basic constructor
# Creates empty ogl::Buffer object, creates ogl::Buffer object from existed buffer ( abufId parameter),
# allocates memory for ogl::Buffer object or copies from host/device memory
oglBuffer() = @cxx cv::ogl::Buffer()

# OpenGL buffer constructors
# Parameters
  # arows	Number of rows in a 2D array
  # acols	Number of columns in a 2D array
  # atype	Array type ( CV_8UC1, ..., CV_64FC4 )
  # abufI:: Buffer object name
     #  READ_ONLY, WRITE_ONLY, READ_WRITE
  # autoRelease	Auto release mode (if true, release will be called in object's destructor)
oglBuffer(arows::Int, acols::Int, atype::Int, abufId=READ_WRITE, autoRelease = false) = @cxx cv::ogl::Buffer::Buffer(arows, acols, atype, abufId, autoRelease)

# using size::cvSize (2D array size)
oglBuffer(size, atype::Int, abufId=READ_WRITE, autoRelease = false) = @cxx cv::ogl::Buffer::Buffer(size, atype, abufId, autoRelease)

# using defined buffer usage, TARGET= ARRAY_BUFFER
# ARRAY_BUFFER
# The buffer will be used as a source for vertex data.
# ELEMENT_ARRAY_BUFFER
# The buffer will be used for indices (in glDrawElements, for example)
# PIXEL_PACK_BUFFER
# The buffer will be used for reading from OpenGL textures.
# PIXEL_UNPACK_BUFFER
# The buffer will be used for writing to OpenGL textures.

oglBuffer(arows::Int, acols::Int, atype::Int, target = ARRAY_BUFFER, autoRelease = false) = @cxx cv::ogl::Buffer::Buffer(arows, acols, atype, target, autoRelease)

# takes input array
oglBuffer(arr, atype::Int, target = ARRAY_BUFFER, autoRelease = false) = @cxx cv::ogl::Buffer::Buffer(arr, target, autoRelease)


# TO DO: Texture2D
oglTexture2D() = @cxx cv::ogl::Texture2D()
