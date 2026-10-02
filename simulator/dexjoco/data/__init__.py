"""Dexjoco data storage and video-writing utilities."""

__all__ = ["ZarrEpisodeStore", "Mp4VideoWriter"]


def __getattr__(name):
    if name == "ZarrEpisodeStore":
        from .episode_store import ZarrEpisodeStore

        return ZarrEpisodeStore
    if name == "Mp4VideoWriter":
        from .video_writer import Mp4VideoWriter

        return Mp4VideoWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
