# Only used for PyTorch open source BUCK build

def is_arvr_mode():
    if repository_name() != "@":
        fail("This file is only for open source PyTorch build. Use the one in fbsource/tools instead!")

    return False
