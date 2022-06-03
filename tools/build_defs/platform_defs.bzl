# Only used for PyTorch open source BUCK build

if repository_name() != "@":
    fail("This file is only for open source PyTorch build. Use the one in fbsource/tools instead!")

CXX = "Default"

ANDROID = "Android"

APPLE = "Apple"

FBCODE = "Fbcode"

WINDOWS = "Windows"

UNIFIED = "Unified"

# Apple SDK Definitions
IOS = "ios"

WATCHOS = "watchos"

MACOSX = "macosx"

APPLETVOS = "appletvos"

xplat_platforms = struct(
    ANDROID = ANDROID,
    APPLE = APPLE,
    CXX = CXX,
    FBCODE = FBCODE,
    WINDOWS = WINDOWS,
    UNIFIED = UNIFIED,
)

apple_sdks = struct(
    IOS = IOS,
    WATCHOS = WATCHOS,
    MACOSX = MACOSX,
    APPLETVOS = APPLETVOS,
)
