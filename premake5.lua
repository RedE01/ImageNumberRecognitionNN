workspace "ImageRecognitionNN"
	configurations  { "Release" }

project "ImageRecognitionNN"
	architecture "x64"
	kind "ConsoleApp"
	language "C++"
	targetdir "bin/%{cfg.buildcfg}"

	files { "src/**.h", "src/**.cpp"}

	includedirs { "./vendor"}

	filter "configurations:Release"
		optimize "On"