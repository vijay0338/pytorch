# Only used for PyTorch open source BUCK build

load(":buck_helpers.bzl", "filter_attributes", "to_oss_deps")

def fb_xplat_cxx_library(
        name,
        deps = [],
        exported_deps = [],
        **kwgs):
    if repository_name() != "@":
        fail("This file is only for open source PyTorch build. Use the one in fbsource/tools instead!")

    cxx_library(
        name = name,
        deps = to_oss_deps(deps),
        exported_deps = to_oss_deps(exported_deps),
        **filter_attributes(kwgs)
    )
