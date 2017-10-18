/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cstring>
#include <chrono>


#define DEPTH_TEST			false
#define NORMAL_TEST			false
#define LAMBERT_SHADING		true
#define LIGHT_POS			glm::vec3(0.0f, 0.0f, 0.0f)
#define FRAG_COL			glm::vec3(0.0f, 0.5f, 1.0f)

#define BILINEAR			true
#define USING_MUTEX			true
#define DRAW_POINTS			false
#define DRAW_LINES			false

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		 glm::vec3 col; 
		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		 int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	//EYESPACE = CAMERA/VIEW SPACE

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;
		 VertexAttributeTexcoord texcoord0;
		 TextureData* dev_diffuseTex;
		 int diffuseTexWidth;
		 int diffuseTexHeight;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;

		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

static int * dev_mutex = NULL;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}



// =============================================================================
//								TIMER FUNCTIONS
// =============================================================================

using time_point_t = std::chrono::high_resolution_clock::time_point;
time_point_t time_start_cpu;
time_point_t time_end_cpu;
bool cpu_timer_started = false;
float prev_elapsed_time_cpu_milliseconds = 0.f;

void startCpuTimer()
{
	if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
	cpu_timer_started = true;
	time_start_cpu = std::chrono::high_resolution_clock::now();
}

void endCpuTimer()
{
	time_end_cpu = std::chrono::high_resolution_clock::now();

	if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

	std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
	prev_elapsed_time_cpu_milliseconds =
		static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

	cpu_timer_started = false;
}

void printCPUTimer()
{
	std::cout << "Time (in ms): " << prev_elapsed_time_cpu_milliseconds << std::endl;
}



/** 
* Writes fragment colors to the framebuffer
*/

// ===========================================================================================
//						FRAGMENT SHADING / FRAGMENT TO FRAME BUFFER WRITING
// ===========================================================================================
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) 
	{
		Fragment currFrag = fragmentBuffer[index];

		// TODO: add your fragment shader code here

		if (currFrag.dev_diffuseTex == NULL)
		{
			glm::vec3 outputColor(1.0f);
			
			if (DRAW_POINTS || DRAW_LINES)
			{
				outputColor = glm::vec3(currFrag.color);
			}

			else if (DEPTH_TEST)
			{
				// The background is black instead of white because rendering based on only frag color
				// If depth buffer was passed, then this would render the background as white 
				// since the buffer was set to high int's
				outputColor = glm::vec3(currFrag.color);
			}

			else if (LAMBERT_SHADING)
			{
				float lambert = glm::abs(glm::dot(currFrag.eyeNor, glm::normalize(LIGHT_POS - currFrag.eyePos)));
				outputColor = glm::vec3(glm::clamp(lambert * currFrag.color, 0.0f, 1.0f));
			}

			else if (NORMAL_TEST)
			{
				outputColor = glm::vec3(glm::clamp(currFrag.eyeNor, 0.0f, 1.0f));
			}

			framebuffer[index] = outputColor;
		}//end if null texture

		 // Texture Mapping with and w/o Bilinear filtering
		else
		{
			//https://stackoverflow.com/questions/35005603/get-color-of-the-texture-at-uv-coordinate
			//https://en.wikipedia.org/wiki/Bilinear_filtering
			//https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/bilinear-filtering

			glm::vec3 texColor(1.0f);
			TextureData* tex = currFrag.dev_diffuseTex;
			int texWidth = currFrag.diffuseTexWidth;
			int texHeight = currFrag.diffuseTexHeight;
			// Scale the UV coords to width and height of texture
			float uCoord = currFrag.texcoord0.x * texWidth;
			float vCoord = currFrag.texcoord0.y * texHeight;
			float r = 1.0f;
			float g = 1.0f;
			float b = 1.0f;

			if (BILINEAR)
			{
				int u_floor = glm::floor(uCoord);
				int v_floor = glm::floor(vCoord);
				float u_fract = uCoord - u_floor;
				float v_fract = vCoord - v_floor;
				float u_opposite = 1.0f - u_fract;
				float v_opposite = 1.0f - v_fract;

				int c00_idx = 3 * (u_floor + (v_floor * texWidth));
				int c10_idx = 3 * ((u_floor + 1) + (v_floor * texWidth));
				int c01_idx = 3 * (u_floor + ((v_floor + 1) * texWidth));
				int c11_idx = 3 * ((u_floor + 1) + ((v_floor + 1) * texWidth));
				
				r = ((tex[c00_idx] * u_opposite + tex[c10_idx] * u_fract) * v_opposite) + 
					((tex[c01_idx] * u_opposite + tex[c11_idx] * u_fract) * v_fract);
				
				g = ((tex[c00_idx + 1] * u_opposite + tex[c10_idx + 1] * u_fract) * v_opposite) +
					((tex[c01_idx + 1] * u_opposite + tex[c11_idx + 1] * u_fract) * v_fract);

				b = ((tex[c00_idx + 2] * u_opposite + tex[c10_idx + 2] * u_fract) * v_opposite) +
					((tex[c01_idx + 2] * u_opposite + tex[c11_idx + 2] * u_fract) * v_fract);
			}//end bilinear

			else
			{
				int i_uCoord = int(uCoord);
				int i_vCoord = int(vCoord);
				int pixelUVIdx = 3 * (i_uCoord + (i_vCoord * texWidth));
				r = tex[pixelUVIdx + 0];
				g = tex[pixelUVIdx + 1];
				b = tex[pixelUVIdx + 2];
			}//end with no bilinear

			texColor = glm::vec3(r, g, b) / 255.0f;
			framebuffer[index] = texColor;

		}//end else texture not null

    }//end index
}//end kernel

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(int));
	cudaMemset(dev_mutex, 0, width * height * sizeof(int));

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}

// ===========================================================================================
//										Vertex Assembly 
// ===========================================================================================

__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here

		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		glm::vec3 currVertPos = primitive.dev_position[vid];
		glm::vec3 currNor = primitive.dev_normal[vid];

		glm::vec4 _currVertPos = glm::vec4(currVertPos, 1.0f);
		glm::vec4 unHomScreenSpace = MVP * _currVertPos;

		// Then divide the pos by its w element to transform into NDC space
		// Perspective divide and Perspective Correct Interpolation
		unHomScreenSpace /= unHomScreenSpace.w;
		
		// Finally transform x and y to viewport space
		glm::vec4 pixelPos = unHomScreenSpace;
		float pixelX = (float)width * ((unHomScreenSpace.x + 1.0f) / 2.0f);
		float pixelY = (float)height * ((1.0f - unHomScreenSpace.y) / 2.0f);
		pixelPos.x = pixelX;
		pixelPos.y = pixelY;

		// Convert z from [-1, 1] to [0, 1] to be between clipping planes
		pixelPos.z = -(1.0f + pixelPos.z) / 2.0f;

		// Position and normal from camera 
		glm::vec3 cameraPos = glm::vec3(MV * _currVertPos);
		glm::vec3 cameraNor = glm::normalize(MV_normal * currNor);

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		primitive.dev_verticesOut[vid].pos = pixelPos;
		primitive.dev_verticesOut[vid].eyePos = cameraPos;
		primitive.dev_verticesOut[vid].eyeNor = cameraNor;

		// Give a preliminary color 
		primitive.dev_verticesOut[vid].col = FRAG_COL;

		// Texture info
		if (primitive.dev_diffuseTex == NULL)
		{
			primitive.dev_verticesOut[vid].dev_diffuseTex = NULL;
		}
		else
		{
			primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
			primitive.dev_verticesOut[vid].texWidth = primitive.diffuseTexWidth;
			primitive.dev_verticesOut[vid].texHeight = primitive.diffuseTexHeight;
		}
	}
}


// ===========================================================================================
//										Primitive Assembly 
// ===========================================================================================

static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}


		// TODO: other primitive types (point, line)

		//if (primitive.primitiveMode == TINYGLTF_MODE_POINTS)
		//{

		//}

		//if (primitive.primitiveMode == TINYGLTF_MODE_LINE)
		//{

		//}
	}
	
}

__device__
void _rasterizeLines(Fragment* fragmentBuffer, int width, int height, glm::vec3 pt1, glm::vec3 pt2, glm::vec3 color)
{
	int dx = pt2.x - pt1.x;
	int dy = pt2.y - pt1.y;
	int m = dy / dx;
	int eps = 0;
	int y = pt1.y;


	int dxe = 0;

	for (int x = pt1.x; x <= pt2.x; x++)
	{

		int fragIdx = x + (y * width);
		fragmentBuffer[fragIdx].color = color;

		// Method 1
		// https://www.cs.helsinki.fi/group/goa/mallinnus/lines/bresenh.html
		//eps += dy;
		//if (m > 0)
		//{
		//	if ((eps << 1) >= dx)
		//	{
		//		y++;
		//		eps -= dx;
		//	}
		//}

		//if (m < 0)
		//{
		//	if (eps + m > -0.5)
		//	{
		//		eps = eps + m;
		//	}
		//	else
		//	{
		//		y--;
		//		eps = eps + m + 1;
		//	}
		//}

		// http://groups.csail.mit.edu/graphics/classes/6.837/F02/lectures/6.837-7_Line.pdf
		// Method 2
		// This is only for case x1 < x2, m <= 1
		//y = pt1.y + m * (x - pt1.x);
		//eps += m;
		//if (eps > 0.5)
		//{
		//	y++;
		//	eps -= 1;
		//}


		// Method 3
		y = pt1.y + m * (x - pt1.x);
		dxe += m * dx + dy;
		if (dxe > (dx + 1) / 2)
		{
			y++;
			eps -= 1;
		}
	}
}

// ===========================================================================================
//										Rasterize Kernel
// ===========================================================================================

/*
	For every triangle
	Calculate AABB
	Iterate through min and max AABB bounds
	Calculate barycentric coord 
	Check if barycentric coord is in triangle 

	NOTES for depth test 
	- if currdepth is < curr depth in depth buffer, replace and write to depth buffer with new value
	- need to create a float buffer for this
	- or use the int depth_buffer as an index container
	- fragmentbuffer[depth_buffer[idx]].color = glm::vec3(depthval)
	- TO USE INT BUFFER, SCALE THE DEPTH VALUE YOU GET SO THAT IT BECOMES AN INT
	- AND THEN JUST COMPARE WITH THAT
	- race condition : if multiple threads trying to write to same place in depth buffer
*/

__global__
void _rasterize(int numTriIndices, Primitive* dev_primitives, Fragment* fragmentBuffer, int width, int height, int* dev_depth, int* dev_mutex)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < numTriIndices)
	{
		Primitive currPrim = dev_primitives[idx];

		glm::vec3 triEyePos[3];
		triEyePos[0] = glm::vec3(currPrim.v[0].eyePos);
		triEyePos[1] = glm::vec3(currPrim.v[1].eyePos);
		triEyePos[2] = glm::vec3(currPrim.v[2].eyePos);

		glm::vec3 triEyeNor[3];
		triEyeNor[0] = glm::vec3(currPrim.v[0].eyeNor);
		triEyeNor[1] = glm::vec3(currPrim.v[1].eyeNor);
		triEyeNor[2] = glm::vec3(currPrim.v[2].eyeNor);

		glm::vec3 triFragCol[3];
		triFragCol[0] = glm::vec3(currPrim.v[0].col);
		triFragCol[1] = glm::vec3(currPrim.v[1].col);
		triFragCol[2] = glm::vec3(currPrim.v[2].col);

		glm::vec2 triTexCoord[3];
		triTexCoord[0] = glm::vec2(currPrim.v[0].texcoord0);
		triTexCoord[1] = glm::vec2(currPrim.v[1].texcoord0);
		triTexCoord[2] = glm::vec2(currPrim.v[2].texcoord0);

		glm::vec3 triPos[3];
		triPos[0] = glm::vec3(currPrim.v[0].pos);
		triPos[1] = glm::vec3(currPrim.v[1].pos);
		triPos[2] = glm::vec3(currPrim.v[2].pos);
		AABB triAABB = getAABBForTriangle(triPos);

		if (DRAW_POINTS)
		{
			for (int i = 0; i < 3; i++)
			{
				int x = triPos[i].x;
				int y = triPos[i].y;
				int fragIdx = x + (y * width);
				fragmentBuffer[fragIdx].color = currPrim.v[0].col;	//all 3 verts are set to arbitrary color from vert shader
			}
		}//end render points

		else if (DRAW_LINES)
		{
			//Draw lines between the 3 vertices of triangle 
			_rasterizeLines(fragmentBuffer, width, height, triPos[0], triPos[1], currPrim.v[0].col);
			_rasterizeLines(fragmentBuffer, width, height, triPos[1], triPos[2], currPrim.v[0].col);
			_rasterizeLines(fragmentBuffer, width, height, triPos[2], triPos[0], currPrim.v[0].col);

		}//end render lines

		else
		{
			//int clampedWidthMin = glm::clamp((int)triAABB.min.x, 0, (int)triAABB.min.x);
			//int clampedWidthMax = glm::clamp((int)triAABB.max.x, (int)triAABB.max.x, width);
			//int clampedHeightMin = glm::clamp((int)triAABB.min.y, 0, (int)triAABB.min.y);
			//int clampedHeightMax = glm::clamp((int)triAABB.max.y, (int)triAABB.max.y, height);

			for (int x = triAABB.min.x; x <= triAABB.max.x; x++)	//for (int x = clampedWidthMin; x <= clampedWidthMax; x++)
			{
				for (int y = triAABB.min.y; y <= triAABB.max.y; y++)	//for (int y = clampedHeightMin; y <= clampedHeightMax; y++)
				{
					glm::vec3 baryCoord = calculateBarycentricCoordinate(triPos, glm::vec2(x, y));
					bool isBaryCoordInTri = isBarycentricCoordInBounds(baryCoord);
					if (isBaryCoordInTri)
					{
						int fragIdx = x + (y * width);

						if (USING_MUTEX)
						{
							bool isSet = false;
							do
							{
								isSet = (atomicCAS(&dev_mutex[fragIdx], 0, 1) == 0);

								if (isSet)
								{
									// ========================= PUT CODE IN MUTEX CHECK ===============================

									// Calculating color according to depth buffer
									float depthVal = getZAtCoordinate(baryCoord, triPos);
									int scale = 10000;	// Is this a good enough number?
									int scaledDepth = scale * depthVal;

									//atomicMin(&dev_depth[fragIdx], scaledDepth);

									if (scaledDepth < dev_depth[fragIdx])
									{
										dev_depth[fragIdx] = scaledDepth;

										if (DEPTH_TEST)
										{
											glm::vec3 newColor(dev_depth[fragIdx] / (float)scale);
											fragmentBuffer[fragIdx].color = newColor;
										}

										else if (LAMBERT_SHADING)
										{
											glm::vec3 interpolatedEyePos(baryCoord.x * triEyePos[0] + baryCoord.y * triEyePos[1] + baryCoord.z * triEyePos[2]);
											glm::vec3 interpolatedEyeNor(baryCoord.x * triEyeNor[0] + baryCoord.y * triEyeNor[1] + baryCoord.z * triEyeNor[2]);
											glm::vec3 interpolatedFragColor(baryCoord.x * triFragCol[0] + baryCoord.y * triFragCol[1] + baryCoord.z * triFragCol[2]);

											fragmentBuffer[fragIdx].eyePos = interpolatedEyePos;
											fragmentBuffer[fragIdx].eyeNor = interpolatedEyeNor;
											fragmentBuffer[fragIdx].color = interpolatedFragColor;
										}

										else if (NORMAL_TEST)
										{
											glm::vec3 interpolatedEyePos(baryCoord.x * triEyePos[0] + baryCoord.y * triEyePos[1] + baryCoord.z * triEyePos[2]);
											glm::vec3 interpolatedEyeNor(baryCoord.x * triEyeNor[0] + baryCoord.y * triEyeNor[1] + baryCoord.z * triEyeNor[2]);

											fragmentBuffer[fragIdx].eyePos = interpolatedEyePos;
											fragmentBuffer[fragIdx].eyeNor = interpolatedEyeNor;
										}

										else
										{
											glm::vec3 interpolatedFragColor(baryCoord.x * triFragCol[0] + baryCoord.y * triFragCol[1] + baryCoord.z * triFragCol[2]);
											fragmentBuffer[fragIdx].color = interpolatedFragColor;
										}

										// Texture Mapping with perspective correct coordinates
										//https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/perspective-correct-interpolation-vertex-attributes
										glm::vec3 perspCorrectBaryCoord(baryCoord.x / triEyePos[0].z,
											baryCoord.y / triEyePos[1].z,
											baryCoord.z / triEyePos[2].z);

										float uFactor = perspCorrectBaryCoord.x * triTexCoord[0].x +
											perspCorrectBaryCoord.y * triTexCoord[1].x +
											perspCorrectBaryCoord.z * triTexCoord[2].x;

										float vFactor = perspCorrectBaryCoord.x * triTexCoord[0].y +
											perspCorrectBaryCoord.y * triTexCoord[1].y +
											perspCorrectBaryCoord.z * triTexCoord[2].y;

										float z = 1.0f / (perspCorrectBaryCoord.x + perspCorrectBaryCoord.y + perspCorrectBaryCoord.z);

										fragmentBuffer[fragIdx].texcoord0 = glm::vec2(uFactor * z, vFactor * z);

										// These should be the same regardless of which prim's vertex
										fragmentBuffer[fragIdx].dev_diffuseTex = currPrim.v[0].dev_diffuseTex;
										fragmentBuffer[fragIdx].diffuseTexWidth = currPrim.v[0].texWidth;
										fragmentBuffer[fragIdx].diffuseTexHeight = currPrim.v[0].texHeight;

									}//if depths are equal
									 // ========================= PUT CODE IN MUTEX CHECK ===============================
								}//end if isSet

								if (isSet)	dev_mutex[fragIdx] = 0;

							} while (!isSet);
						} //end if using mutex

						else
						{
							// Calculating color according to depth buffer
							float depthVal = getZAtCoordinate(baryCoord, triPos);
							int scale = 10000;	// Is this a good enough number?
							int scaledDepth = scale * depthVal;

							atomicMin(&dev_depth[fragIdx], scaledDepth);
							if (scaledDepth == dev_depth[fragIdx])
							{
								if (DEPTH_TEST)
								{
									glm::vec3 newColor(dev_depth[fragIdx] / (float)scale);
									fragmentBuffer[fragIdx].color = newColor;
								}

								else if (LAMBERT_SHADING)
								{
									glm::vec3 interpolatedEyePos(baryCoord.x * triEyePos[0] + baryCoord.y * triEyePos[1] + baryCoord.z * triEyePos[2]);
									glm::vec3 interpolatedEyeNor(baryCoord.x * triEyeNor[0] + baryCoord.y * triEyeNor[1] + baryCoord.z * triEyeNor[2]);
									glm::vec3 interpolatedFragColor(baryCoord.x * triFragCol[0] + baryCoord.y * triFragCol[1] + baryCoord.z * triFragCol[2]);

									fragmentBuffer[fragIdx].eyePos = interpolatedEyePos;
									fragmentBuffer[fragIdx].eyeNor = interpolatedEyeNor;
									fragmentBuffer[fragIdx].color = interpolatedFragColor;
								}

								else if (NORMAL_TEST)
								{
									glm::vec3 interpolatedEyePos(baryCoord.x * triEyePos[0] + baryCoord.y * triEyePos[1] + baryCoord.z * triEyePos[2]);
									glm::vec3 interpolatedEyeNor(baryCoord.x * triEyeNor[0] + baryCoord.y * triEyeNor[1] + baryCoord.z * triEyeNor[2]);

									fragmentBuffer[fragIdx].eyePos = interpolatedEyePos;
									fragmentBuffer[fragIdx].eyeNor = interpolatedEyeNor;
								}

								else
								{
									glm::vec3 interpolatedFragColor(baryCoord.x * triFragCol[0] + baryCoord.y * triFragCol[1] + baryCoord.z * triFragCol[2]);
									fragmentBuffer[fragIdx].color = interpolatedFragColor;
								}

								// Texture Mapping with perspective correct coordinates
								glm::vec3 perspCorrectBaryCoord(baryCoord.x / triEyePos[0].z,
									baryCoord.y / triEyePos[1].z,
									baryCoord.z / triEyePos[2].z);

								float uFactor = perspCorrectBaryCoord.x * triTexCoord[0].x +
									perspCorrectBaryCoord.y * triTexCoord[1].x +
									perspCorrectBaryCoord.z * triTexCoord[2].x;

								float vFactor = perspCorrectBaryCoord.x * triTexCoord[0].y +
									perspCorrectBaryCoord.y * triTexCoord[1].y +
									perspCorrectBaryCoord.z * triTexCoord[2].y;

								float z = 1.0f / (perspCorrectBaryCoord.x + perspCorrectBaryCoord.y + perspCorrectBaryCoord.z);

								fragmentBuffer[fragIdx].texcoord0 = glm::vec2(uFactor * z, vFactor * z);

								// These should be the same regardless of which prim's vertex
								fragmentBuffer[fragIdx].dev_diffuseTex = currPrim.v[0].dev_diffuseTex;
								fragmentBuffer[fragIdx].diffuseTexWidth = currPrim.v[0].texWidth;
								fragmentBuffer[fragIdx].diffuseTexHeight = currPrim.v[0].texHeight;
							}//if depths are equal
						}//end else not using mutex
					}//end if baryInBounds
				}//end for y
			}//end for x
		}//end render triangles
		
	}//end if idx
}//end _rasterize


 // ===========================================================================================
 //										Rasterize CPU Function 
 // ===========================================================================================


/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// ================================== Vertex Process & primitive assembly ==================================
	//startCpuTimer();
 
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();



		for (; it != itEnd; ++it) 
		{
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) 
			{
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly <<< numBlocksForVertices, numThreadsPerBlock >>>(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				
				cudaDeviceSynchronize();

				_primitiveAssembly <<< numBlocksForIndices, numThreadsPerBlock >>>
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	} // end vertex process and prim assembly
	
	//endCpuTimer();
	//printCPUTimer();

	// ================================== initialize depth and fragment buffer ==================================
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	
	// ================================== rasterize ==================================
	//startCpuTimer();

	dim3 primitive_numThreadsPerBlock(128);
	dim3 numBlocksForPrimitives((totalNumPrimitives + primitive_numThreadsPerBlock.x - 1) / primitive_numThreadsPerBlock.x);
	_rasterize <<<numBlocksForPrimitives, primitive_numThreadsPerBlock>>> (totalNumPrimitives, 
																			dev_primitives, 
																			dev_fragmentBuffer, 
																			width, 
																			height,
																			dev_depth,
																			dev_mutex);
	checkCUDAError("_rasterize");

	//endCpuTimer();
	//printCPUTimer();

	// ================================== Copy depthbuffer colors into framebuffer ==================================
	startCpuTimer();
	render <<<blockCount2d, blockSize2d >>>(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");
	endCpuTimer();
	printCPUTimer();

	// ================================== Copy framebuffer into OpenGL buffer for OpenGL previewing ==================================
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");

}//end rasterize function

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(dev_mutex);
	dev_mutex = NULL;

    checkCUDAError("rasterize Free");
}
