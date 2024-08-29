import Quartz as QZ
import Quartz.CoreGraphics as CG
import AppKit as AK

import numpy as np
import numpy.typing as npt

import typing as t

Matcher = t.Callable[[dict[str, t.Any]], bool]

def get_windows(window: dict[str, t.Any]):
    
    return window.get("kCGWindowName")

def _find_window(matcher: Matcher) -> list[dict]:
    """Find the window id of the window with the given name."""

    window_list = QZ.CGWindowListCopyWindowInfo(
        QZ.kCGWindowListOptionAll, QZ.kCGNullWindowID
    )

    result = []

    for window in window_list:
        if matcher(window):
            result.append(window)

    return result


def new_name_matcher(name: str) -> Matcher:
    def matcher(window: dict[str, t.Any]) -> bool:
        # print(window.get("kCGWindowName"))
        return window.get("kCGWindowName") == name

    return matcher


def capture(name: str):
    windows = _find_window(new_name_matcher(name))

    if len(windows) == 0:
        raise ValueError(f"Could not find window with name: {name}")
    elif len(windows) > 1:
        raise ValueError(f"Found multiple windows with name: {name}")

    window = windows[0]

    return _cg_capture_region_as_image(
        window_id=window["kCGWindowNumber"],
    )

def capture_full_screen():
    return _cg_capture_region_as_image(region=(0, 0, AK.NSScreen.mainScreen().frame().size.width, AK.NSScreen.mainScreen().frame().size.height))

def _cg_capture_region_as_image(
    region: tuple[int, int, int, int] | None = None,
    window_id: int | None = None,
) -> npt.NDArray[np.uint8]:
    """Capture a region of the screen using CoreGraphics."""

    if window_id is not None and region is not None:
        raise ValueError("Only one of region or window_id must be specified")

    image: CG.CGImage | None = None

    if region is not None:
        cg_region = None

        if region is None:
            cg_region = CG.CGRectInfinite
        else:
            cg_region = CG.CGRectMake(*region)

        image = CG.CGWindowListCreateImage(
            cg_region,
            CG.kCGWindowListOptionOnScreenOnly,
            CG.kCGNullWindowID,
            CG.kCGWindowImageDefault,
        )
    elif window_id is not None:
        image = CG.CGWindowListCreateImage(
            CG.CGRectNull,
            CG.kCGWindowListOptionIncludingWindow,
            window_id,
            CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageNominalResolution,
        )
    else:
        raise ValueError("Either region or window_id must be specified")

    if image is None:
        raise ValueError("Could not capture image")

    bpr = CG.CGImageGetBytesPerRow(image)
    width = CG.CGImageGetWidth(image)
    height = CG.CGImageGetHeight(image)

    cg_dataprovider = CG.CGImageGetDataProvider(image)
    cg_data = CG.CGDataProviderCopyData(cg_dataprovider)

    np_raw_data = np.frombuffer(cg_data, dtype=np.uint8)

    return np.lib.stride_tricks.as_strided(
        np_raw_data,
        shape=(height, width, 3),
        strides=(bpr, 4, 1),
        writeable=True,
    )