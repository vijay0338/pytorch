# Only used for PyTorch open source BUCK build

def expect(condition, message = None):
    if repository_name() != "@":
        fail("This file is only for open source PyTorch build. Use the one in fbsource/tools instead!")

    if not condition:
        fail(message)
