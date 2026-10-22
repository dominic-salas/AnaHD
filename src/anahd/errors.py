"""Error types. Each one is something the user can act on; defects surface as normal exceptions."""


class AnaHDError(Exception):
    """Base class for errors reported to the user without a traceback."""


class MediaError(AnaHDError):
    """FFmpeg/ffprobe is missing, or could not read or write a file."""


class OutputExistsError(AnaHDError):
    """The output file exists and --force was not given."""
