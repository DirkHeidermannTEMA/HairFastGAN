from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class HairSwapRequest(_message.Message):
    __slots__ = ("face", "shape", "color", "blending", "poisson_iters", "poisson_erosion", "use_cache")
    FACE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    BLENDING_FIELD_NUMBER: _ClassVar[int]
    POISSON_ITERS_FIELD_NUMBER: _ClassVar[int]
    POISSON_EROSION_FIELD_NUMBER: _ClassVar[int]
    USE_CACHE_FIELD_NUMBER: _ClassVar[int]
    face: bytes
    shape: bytes
    color: bytes
    blending: str
    poisson_iters: int
    poisson_erosion: int
    use_cache: bool
    def __init__(self, face: _Optional[bytes] = ..., shape: _Optional[bytes] = ..., color: _Optional[bytes] = ..., blending: _Optional[str] = ..., poisson_iters: _Optional[int] = ..., poisson_erosion: _Optional[int] = ..., use_cache: bool = ...) -> None: ...

class HairSwapResponse(_message.Message):
    __slots__ = ("image",)
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    def __init__(self, image: _Optional[bytes] = ...) -> None: ...
