# SPDX-License-Identifier: Apache-2.0
from conan import ConanFile
from conan.tools.cmake import cmake_layout


required_conan_version = ">=2.0"


class EvoCppDependencies(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("libnpy/1.0.1")

    def layout(self):
        cmake_layout(self)
