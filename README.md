CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Name: Meghana Seshadri
* Tested on: Windows 10, i7-4870HQ @ 2.50GHz 16GB, GeForce GT 750M 2048MB (personal computer)


## Project Overview

The goal of this project was to get an introduction to writing a GPU Rasterizer in CUDA.

[Click here for instructions of this project](./INSTRUCTION.md)

The following features were implemented (most of which can be toggled with flags in `rasterize.cu`):

**Features**:

* Vertex shading. (_vertexTransformAndAssembly in rasterize.cu)
* Primitive assembly with support for triangles read from buffers of index and vertex data (_primitiveAssembly in rasterize.cu)
* Rasterization (_rasterize in rasterize.cu)
* Fragment shading (render in rasterize.cu)
* A depth buffer for storing and depth testing fragments (int * dev_depth in rasterize.cu)
* Fragment-to-depth-buffer writing (with atomics for race avoidance)
* Lambertian lighting scheme in the Fragment shader (render in rasterize.cu)


The following are extra features implemented on top of the basic rasterization pipeline:

* UV texture mapping with bilinear texture filtering and perspective correct texture coordinates
* Support for rasterizing the following primitives:
	- Points
	- Lines
	- Triangles


### Rasterization Pipeline

## Renders


## Resources 

### CUDA Mutexes

Adapted from
[this StackOverflow question](http://stackoverflow.com/questions/21341495/cuda-mutex-and-atomiccas).

```cpp
__global__ void kernelFunction(...) {
    // Get a pointer to the mutex, which should be 0 right now.
    unsigned int *mutex = ...;

    // Loop-wait until this thread is able to execute its critical section.
    bool isSet;
    do {
        isSet = (atomicCAS(mutex, 0, 1) == 0);
        if (isSet) {
            // Critical section goes here.
            // The critical section MUST be inside the wait loop;
            // if it is afterward, a deadlock will occur.
        }
        if (isSet) {
            mutex = 0;
        }
    } while (!isSet);
}
```

### Credits and other links

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)

***Texture Mapping***
* [Getting color from UV coordinates](https://stackoverflow.com/questions/35005603/get-color-of-the-texture-at-uv-coordinate)
* [Bilinear filtering](https://en.wikipedia.org/wiki/Bilinear_filtering)
* [Bilinear filtering](https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/bilinear-filtering)
* [Perspective Correct Interpolation](https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/perspective-correct-interpolation-vertex-attributes)

***Bresenham Line Rendering***
* [Lecture slides 1](http://groups.csail.mit.edu/graphics/classes/6.837/F02/lectures/6.837-7_Line.pdf)
* [Lecture slides 2](https://www.cs.helsinki.fi/group/goa/mallinnus/lines/bresenh.html)