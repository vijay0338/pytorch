# Only used for PyTorch open source BUCK build

def compose_platform_setting_list(settings):
    """Settings object:
    os/cpu pair: should be valid key, or at most one part can be wildcard.
    flags: the values added to the compiler flags
    """
    if repository_name() != "@":
        fail("This file is only for open source PyTorch build. Use the one in fbsource/tools instead!")

    result = []
    for setting in settings:
        result.append([
            "^{}-{}$".format(setting["os"], setting["cpu"]),
            setting["flags"],
        ])
    return result
