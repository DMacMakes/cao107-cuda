#pragma once
#ifndef __IMGUI_SDL_HELPERS__
#define __IMGUI_SDL_HELPERS__

#include <SDL.h>
#include <imgui.h>

namespace cao {
  
  //bool Setup_SDL2(SDL_Window* window_p, SDL_GLContext gl_context, const char* glsl_version_p);
  bool Check_If_User_Closed(SDL_Window* window_p);
  void Setup_ImGui(SDL_Window* window_p, SDL_GLContext gl_context, const char* glsl_version_p);
  void Render_GUI(ImVec4& clear_color, ImGuiIO& io, SDL_Window* window_p);
  void Cleanup_GUI(SDL_Window* window_p, SDL_GLContext gl_context);
}

#endif
