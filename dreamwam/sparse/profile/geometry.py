"""Native frame/row/column positions; camera labels denote image footprints."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenGrid:
    frames: int
    height: int
    width: int
    cameras: int = 2

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in
               (self.frames, self.height, self.width, self.cameras)):
            raise ValueError("grid dimensions must be positive integers")
        if self.width % self.cameras:
            raise ValueError("camera seam must align to a token column")

    @property
    def frame_size(self):
        return self.height * self.width

    @property
    def length(self):
        return self.frames * self.frame_size

    def coordinates(self):
        """Columns: token, frame, row, joint column, camera, camera-local column."""
        index = np.arange(self.length, dtype=np.int64)
        frame, within = np.divmod(index, self.frame_size)
        row, col = np.divmod(within, self.width)
        camera, local_col = np.divmod(col, self.width // self.cameras)
        return np.stack((index, frame, row, col, camera, local_col), axis=1)

    def regions(self, height=2, width=2):
        """Ragged edge regions are explicit; no region crosses frame or camera."""
        if any(type(x) is not int or x < 1 for x in (height, width)):
            raise ValueError("region dimensions must be positive integers")
        columns = self.width // self.cameras
        result = []
        for frame in range(self.frames):
            for camera in range(self.cameras):
                for row in range(0, self.height, height):
                    for col in range(0, columns, width):
                        result.append(np.array([
                            frame * self.frame_size + r * self.width + camera * columns + c
                            for r in range(row, min(row + height, self.height))
                            for c in range(col, min(col + width, columns))], dtype=np.int64))
        return result
