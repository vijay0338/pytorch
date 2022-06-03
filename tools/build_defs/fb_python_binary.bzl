# Only used for PyTorch open source BUCK build
# @lint-ignore-every FBCODEBZLADDLOADS

def fb_python_binary(**kwgs):
    if repository_name() != "@":
        fail("This file is only for open source PyTorch build. Use the one in fbsource/tools instead!")

    python_binary(**kwgs)
