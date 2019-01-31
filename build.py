from pybuilder.core import use_plugin, init

use_plugin("python.core")
use_plugin("python.unittest")
use_plugin("python.install_dependencies")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")


name = "vision"
version = "0.0.1"
default_task = "publish"


@init
def set_properties(project):
    project.build_depends_on("mock")
    project.build_depends_on("numpy")
    project.build_depends_on("opencv-python")
    project.build_depends_on("mrcnn")
    # required by mrcnn
    project.build_depends_on("tensorflow", version=">= 1.3.0")
    project.build_depends_on("scipy")
    project.build_depends_on("scikit-image")
    project.build_depends_on("keras", version=">= 2.0.8")
