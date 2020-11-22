/***********************************************************************************
* 
* CREEPERS 
*   Defend yourself from swarming creatures that do terrible things (like take
*   your phone and message people who's contacts you should have deleted ages ago).
*
*   Use the power of your GPU to cast a Meteor Strike spell, damaging all enemies
*   within a given radius, and to various levels based on their distance.
*
* HOMEWORK
*   Search for the word TODO. You'll find it in several comments along with 
*   tasks for you to carry out.
*     - In summary, get castMeteorStrikeCuda working. It needs to set up
*       the GPU memory, call the spell kernel, copy back back the info.
* 
* NOTES
*   
* Creeper Stats
*   Health and distance are rolled (randomised, as in rolling dice in a 
*   table top game) on start up and can be rolled again with a button. 
*     - The CPU is used because random number libraries for CUDA are 
*       too complicated for this example


***********************************************************************************/
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

// Swarming creatures that do terrible things like ask your friends where they 
// bought their shirt/pants/shoes and then look you up and down.
struct Creeper
{
  // Health in range 0.0 to 1.0, equivalent to a percentage.
  float health = 1.0f;
  
  // Distance in meters. 10.0f = 10m.
  float distance = 10.0f;
};

bool Setup_SDL2();
// Set up arrays of addends (numbers to add), array for sums (answers), push to gpu to execute.
cudaError_t addWithCuda(const int* addends_1, const int* addends_2, int* sums, unsigned int size);
cudaError_t castMeteorStrikeCuda(Creeper* creepers);
cudaError_t Cuda_Cleanup();
void rollCreepers(Creeper* creepers);

SDL_Window* window_p{ nullptr };
SDL_GLContext gl_context;
const char* glsl_version_p = "";

const unsigned int CREEPER_COUNT = 32;
const int CREEPER_MAX_DIST = 20;

/*--------------------------------------------------------------- KERNELS for gpu */

// __global__ tells nvcc to define kernels in GPU memory

// For each index n, add the value at n in each array of addends (terms)
// and place the sum (result) in the sums array.
__global__ void sumIntArrays_k(const int* addends_1, const int* addends_2, int* sums)
{
  int i = threadIdx.x;
  sums[i] = addends_1[i] + addends_2[i];
}

// Kernel of Meteor Strike: Area of effect spell, with three bands of damage.
// Damage creepers in supplied array based on their distance from hero.
__global__ void castMeteorStrike_kernel(Creeper* creepers)
{
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
    creepers[i].health -= 0.6f;
  }
  else if (creepers[i].distance < 12.0f)
  {
    creepers[i].health -= 0.35f;
  }
}

/*------------------------------------------------------------------------- MAIN */

// Don't remove these arguments from the main function declaration
// or the nvidia nvcc compiler driver will spit the dummy.
int main(int argc, char** argv)
{
  bool err = Setup_SDL2();
  // EXIT if SDL2 setup failed. Setup_SDL2 will have displayed an error already.
  if (err) return 1;
  srand(static_cast<int>(time(NULL)));
  
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
    
    // TODO: Display the rest of the creepers here too, not just the first.
    //  Use something more visual and immediate than text output below: Look through
    //  the demo window, maybe there's a status bar or some other indicator.
    ImGui::Text("Creeper 1 is %.1fm away and has %.0f%% health.", 
      creepers[0].distance, roundf(creepers[0].health * 100));
    
    // Smash creepers with meteorstrike spell.
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
    ImGui::Checkbox("Demo Window", &show_demo_window);  

    ImGui::End();

    cao::Render_GUI(clear_color, io, window_p);
  }
  
  // Shut down and clean up the GL/SDL/ImGui bits
  cao::Cleanup_GUI(window_p, gl_context);
  // exit with error if cuda cleanup fails.
  if (Cuda_Cleanup() != cudaSuccess) return 1;
  return 0;
}

// Set random health (minimum 50% aka 0.5) and distance (meters) for each creeper.
// "roll" as in rolling dice to generate random stats in a table top game.
void rollCreepers(Creeper* creepers)
{
  // Set random health (minimum 50% aka 0.5) and distance (meters) for each creeper.
  for (int id = 0; id < CREEPER_COUNT; id++)
  {
    float randRatio = static_cast<float>(rand()) / RAND_MAX; // float from 0-1.
    creepers[id].health = 1.0f - (randRatio * 0.5f);          // between 0.5 and 1
    creepers[id].distance = static_cast<float>(rand() % (CREEPER_MAX_DIST + 1));
  }
}

// Summon a mighty meteor blast to, using the gpu, deal area-of-effect 
// damage to the supplied Creeper collection (pointer to an array)
cudaError_t castMeteorStrikeCuda(Creeper* creepers)
{
  cudaError_t cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  }
  
  cudaStatus = cudaSuccess; // Reset
  Creeper* dev_creepers = nullptr;    // Initialize your pointers with nullptr.

  // TODO: Allocate a block of memory on the gpu to hold the creeper collection
  //cudaStatus = cudaMalloc((void**)&dev_______, CREEPER_COUNT * ______(______));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
  }

  // TODO: Copy creepers from host memory to GPU buffers.
  //cudaStatus = cudaMemcpy(_______, ______, ______ * ______(______), ______);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
  }

  // TODO: Using 1 thread per creeper, blow them up.
  //_________kernel <<<1, CREEPER_COUNT >>> (______);

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

  // TODO: Copy output vector from GPU memory to host memory.
  // Look at the addWithCuda function to see how it works, compare with
  // the copy TO gpu above (around line 220)
  //______ = ______(______, ______, ______ * ______(______), ______);
  //if (______ != ______) {
  //  fprintf(stderr, "______ failed!");
 // }

  cudaFree(dev_creepers);
  
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
  }

  cudaStatus = cudaMalloc((void**)&dev_addends_1, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
  }

  cudaStatus = cudaMalloc((void**)&dev_addends_2, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(dev_addends_1, addends_1, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
  }

  cudaStatus = cudaMemcpy(dev_addends_2, addends_2, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
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
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  // Like waiting for a whole pool/group of threads to join.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sumIntArrays_k!\n", cudaStatus);
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(sums, dev_sums, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
  }

  // Tell c++/cuda we're done with the memory we requested on the device with malloc.
  // Please release it for reuse.
  cudaFree(dev_sums);
  cudaFree(dev_addends_1);
  cudaFree(dev_addends_2);

  return cudaStatus;
}

/*----------------------------------------------------------- SETUP and CLEANUP */

// CudaCleanup makes sure cuda profilers can display info 
// up to the end of execution.
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
