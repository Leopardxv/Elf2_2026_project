#!/usr/bin/env python3
"""Voice assistant package."""

__all__ = ["VoiceAgent"]


def __getattr__(name):
    if name == "VoiceAgent":
        from .voice_agent import VoiceAgent
        return VoiceAgent
    raise AttributeError(name)
