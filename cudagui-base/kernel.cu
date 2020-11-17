
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
// About Desktop OpenGL function loaders:
//  Modern desktop OpenGL doesn't have a standard portable header file to load OpenGL function pointers.
//  Helper libraries are often used for this purpose! Here we are supporting gl3w.
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>            // Initialize with gl3wInit()
#endif
#include "imgui_sdl_helpers.h" 

cudaError_t addWithCuda(const int* addends_1, const int* addends_2, int* sums, unsigned int size);

// The entirety of our kernel. .
__global__ void sumIntArrays_k(int* c, const int* a, const int* b)
{
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

SDL_Window* window_p{ nullptr };
SDL_GLContext gl_context;
const char* glsl_version_p = "";

bool Setup_SDL2();
cudaError_t Cuda_Cleanup();

int main(int, char**)
{
  bool err = Setup_SDL2();
  // EXIT if SDL2 setup failed. Setup_SDL2 will have displayed an error already.
  if (err) return 1;

  //Setup_ImGui();
  cao::Setup_ImGui(window_p, gl_context, glsl_version_p);
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  // Our state
  bool show_demo_window = true;
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  bool done = false;

  const int array_size = 5;
  const int addends_1[array_size] = { 1, 2, 3, 4, 5 };
  const int addends_2[array_size] = { 10, 20, 30, 40, 50 };
  int sums[array_size] = { 0 };

  // Add arrays in parallel.
  cudaError_t cuda_status = addWithCuda(addends_1, addends_2, sums, array_size);
  if (cuda_status != cudaSuccess) {
    fprintf(stderr, "addWithCuda failed!");
    return 1;
  }

  //////// begin imgui sdl main loop
  while (!done)
  {
    // Check if any window close or sdl exit events have been fired coming into this frame.
    done = cao::Check_If_User_Closed(window_p); // Check_If_User_Closed();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(window_p);
    ImGui::NewFrame();

    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {
      static int counter = 0;

      ImGui::Begin("Vec add results:");                          // Create a window called "Hello, world!" and append into it.
      ImGui::Text("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        sums[0], sums[1], sums[2], sums[3], sums[4]);
      ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state

      //ImGui::SameLine();
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
      ImGui::End();
    }

    cao::Render_GUI(clear_color, io, window_p);
  }
  
  // Shut down and clean up the GL/SDL/ImGui bits
  cao::Cleanup_GUI(window_p, gl_context);
  
  // exit with error if cuda cleanup fails.
  if (Cuda_Cleanup() != cudaSuccess) return 1;

  return 0;
}

// CudaCleanup makes sure cuda profilers can display info up to the end of execution.
cudaError_t Cuda_Cleanup()
{
  //cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaError_t status = cudaDeviceReset();
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
  }
  return status;
}

// Helper function for using CUDA to add vectors in parallel.
// I'm guessing it adds numbers from vectors a and b and places them in the
// same index in vector c. Meaningful names would be nice.
cudaError_t addWithCuda(const int* addends_1, const int* addends_2, int* sums, unsigned int size)
{
  int* dev_addends_1 = 0;
  int* dev_addends_2 = 0;
  int* dev_sums = 0;
  cudaError_t cudaStatus;

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    goto Error;
  }
  
  // Allocate GPU buffers for three vectors (two input, one output)    .
  // GPU buffers means some space in the local DRAM on the gpu, what 
  // the gpu will consider "global" ram.
  // The amount = The number of integers the vector can hold
  // multiplied by this system's int data type size (say, 32 or 16 bits).
  // On windows it defaults to 32.
  cudaStatus = cudaMalloc((void**)&dev_sums, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_addends_1, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_addends_2, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(dev_addends_1, addends_1, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }

  cudaStatus = cudaMemcpy(dev_addends_2, addends_2, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }

  // This is where it all happens. We're issuing the go command to the 
  // gpu: run this kernel, and for each number one of these vectors can hold
  // create one thread. If there are 10 ints per vector, there will be 
  // 10 total threads. The first will read the first two numbers from a and b,
  // add them and store the result in c.
  sumIntArrays_k <<<1, size >>> (dev_sums, dev_addends_1, dev_addends_2);

  // Execution doesn't stop for the threads: it's the same as when we create
  // C++ standard threads, things keep going unless we choose to communicate with
  // or wait for data from a thread.

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "sumIntArrays_k launch failed: %s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  // Like waiting for a whole pool/group of threads to join.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sumIntArrays_k!\n", cudaStatus);
    goto Error;
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(sums, dev_sums, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }

  // Tell c++/cuda we're done with the memory we requested on the device with malloc.
  // Please release it for reuse.
Error:
  cudaFree(dev_sums);
  cudaFree(dev_addends_1);
  cudaFree(dev_addends_2);

  return cudaStatus;
}

bool Setup_SDL2()
{
  // Setup SDL
// (Some versions of SDL before <2.0.10 appears to have performance/stalling issues on a minority of Windows systems,
// depending on whether SDL_INIT_GAMECONTROLLER is enabled or disabled.. updating to latest version of SDL is recommended!)
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
  {
    printf("Error: %s\n", SDL_GetError());
    return true;
  }

  // Decide GL+GLSL versions
#ifdef __APPLE__
    // GL 3.2 Core + GLSL 150
  glsl_version_p = "#version 150";
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
    // GL 3.0 + GLSL 130
  glsl_version_p = "#version 130";
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif

  // Create window with graphics context
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  window_p = SDL_CreateWindow("Dear ImGui SDL2+OpenGL3 example", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
  gl_context = SDL_GL_CreateContext(window_p);
  SDL_GL_MakeCurrent(window_p, gl_context);
  SDL_GL_SetSwapInterval(1); // Enable vsync

  // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
  bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
  bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
  bool err = gladLoadGL() == 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD2)
  bool err = gladLoadGL((GLADloadfunc)SDL_GL_GetProcAddress) == 0; // glad2 recommend using the windowing library loader instead of the (optionally) bundled one.
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
  bool err = false;
  glbinding::Binding::initialize();
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
  bool err = false;
  glbinding::initialize([](const char* name) { return (glbinding::ProcAddress)SDL_GL_GetProcAddress(name); });
#else
  bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
  if (err)
  {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
  }
  return err;
}
