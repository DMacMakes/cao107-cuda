
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
//  Modern desktop OpenGL doesn't have a standard portable header file to load OpenGL function pointers.
//  Helper libraries are often used for this purpose! Here we are supporting gl3w.
#include <GL/gl3w.h>            // Initialize with gl3wInit()
// Find this one in the cao filter/folder in solution explorer.
#include "imgui_sdl_helpers.h"
#include <stdlib.h>
#include <time.h>
#include "curand_kernel.h"

struct Creeper
{
  float health = 1.0f;
  float distance = 10.0f;
};

bool Setup_SDL2();
// Set up arrays of addends (numbers to add), array for sums (answers), push to gpu to execute.
cudaError_t addWithCuda(const int* addends_1, const int* addends_2, int* sums, unsigned int size);
cudaError_t castMeteorStrikeCuda(Creeper* creepers);
cudaError_t Cuda_Cleanup();

SDL_Window* window_p{ nullptr };
SDL_GLContext gl_context;
const char* glsl_version_p = "";
const unsigned int CREEPER_COUNT = 32;
const int CREEPER_MAX_DIST = 20;

// The kernel! __global__ tells nvcc to define this over in GPU memory
__global__ void sumIntArrays_k(const int* addends_1, const int* addends_2, int* sums)
{
  int i = threadIdx.x;
  sums[i] = addends_1[i] + addends_2[i];
}

void rollCreepers(Creeper* creepers)
{
  //int i = threadIdx.x;
  for (int id = 0; id < CREEPER_COUNT; id++)
  {
    creepers[id].health = static_cast<float>(rand()) / RAND_MAX;
    creepers[id].distance = static_cast<float>(rand() % (CREEPER_MAX_DIST+1));
  }
    // Set random health and distance for each creeper.
  // Maybe we can set up 300 in parallel? Will sizeof Creeper work
}

__global__ void castMeteorStrike_kernel(Creeper* creepers)
{
  // Area of effect spell, with three bands of damage.
  // Damage all n bad guys based on their distance from hero.
  int i = threadIdx.x;
  if (creepers[i].distance < 1.5f)
  {
    creepers[i].health -= 1.0f;
  }
  else if (creepers[i].distance < 2.5f)
  {
    creepers[i].health -= 0.9f;
  }
  else if (creepers[i].distance < 6.0f)
  {
    creepers[i].health -= 60.0f;
  }
  else if (creepers[i].distance < 12.0f)
  {
    creepers[i].health -= 35.0f;
  }
  // Change to actually testing the distance from a hero using
  // distance between 2 points. (pythagoras)
}

// Don't remove these arguments from the main function declaration
// or the nvidia nvcc compiler driver will spit the dummy.
int main(int argc, char** argv)
{
  bool err = Setup_SDL2();
  // EXIT if SDL2 setup failed. Setup_SDL2 will have displayed an error already.
  if (err) return 1;
  srand(time(NULL));
  //Setup_ImGui();
  cao::Setup_ImGui(window_p, gl_context, glsl_version_p);
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  // Our state
  bool show_demo_window = true;
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  bool done = false;
  
  /* Create N creepers with full health at 10m distance. ------------------------*/
  Creeper creepers[CREEPER_COUNT]{ 1.0f,10.0f };
  
  // Randomise creeper distance and health
  rollCreepers(creepers);

  /*   Add two arrays -----------------------------------------------------------*/
  const int array_size = 5;
  const int addends_1[array_size] = { 1, 2, 3, 4, 5 };
  const int addends_2[array_size] = { 10, 20, 30, 40, 50 };
  int sums[array_size] = { 0 };
  bool added_arrays = false;
  // Add arrays in parallel.


  /*  begin imgui sdl main loop ------------------------------------------------*/
  while (!done)
  {
    // Check if any window close or sdl exit events have been fired coming into this frame.
    done = cao::Check_If_User_Closed(window_p); // Check_If_User_Closed();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(window_p);
    ImGui::NewFrame();

    // We have no problems until something fails:.
    cudaError_t cuda_status = cudaSuccess;

    // Imgui's example implementation of most features, great reference
    if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    ImGui::Begin("Coding our Cuda GPU");                          // Create a window called "Hello, world!" and append into it.

    if (ImGui::Button("Add arrays"))
    {
      cuda_status = addWithCuda(addends_1, addends_2, sums, array_size);
      if (cuda_status != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
      }
      added_arrays = true;
    }
    if (added_arrays)
    {
      ImGui::Text("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        sums[0], sums[1], sums[2], sums[3], sums[4]);
    }

    ImGui::Spacing();
    ImGui::Spacing();
    // Smash creepers with meteorstrike spell.
    ImGui::Text("Creeper 1 is %.1fm away and has %.0f%% health.", creepers[0].distance, roundf(creepers[0].health * 100));
    if (ImGui::Button("Cast meteor strike"))
    {
      castMeteorStrikeCuda(creepers);
      if (cuda_status != cudaSuccess) {
        fprintf(stderr, "castMeteorStrikeCuda failed!");
      }
    }
    ImGui::Spacing();

    if (ImGui::Button("Re-roll creeper stats"))
    {
      rollCreepers(creepers);
    }
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state

      //ImGui::SameLine();
//    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

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

cudaError_t castMeteorStrikeCuda(Creeper* creepers)
{
  cudaError_t cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }

  Creeper* dev_creepers;
  // Allocate a block of memory on the gpu to hold the creepers
/*  cudaStatus = cudaMalloc((void**)&dev_______, CREEPER_COUNT * ______(______));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
  }

  // Copy creepers from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(_______, ______, ______ * ______(______), ______);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
  }

  // Using 1 thread per creeper, blow them up.
  _________kernel <<<1, CREEPER_COUNT >>> (______);

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "castMeteorStrike_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  // Like waiting for a whole pool/group of threads to join.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching castMeteorStrike_kernel!\n", cudaStatus);
  }

  // Copy output vector from GPU buffer to host memory.
  ______ = ______(______, ______, ______ * ______(______), ______);
  if (______ != ______) {
    fprintf(stderr, "______ failed!");
  }

  cudaFree(dev_creepers);
  */
  return cudaStatus;

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
  sumIntArrays_k <<<1, size >>> (dev_addends_1, dev_addends_2, dev_sums);

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

  bool err = gl3wInit() != 0;
  if (err)
  {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
  }
  return err;
}
