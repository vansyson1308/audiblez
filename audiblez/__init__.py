import sys


def ensure_utf8_stdio():
    """Windows consoles default to a legacy code page (cp1252...) that cannot print Cyrillic: use UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and (stream.encoding or '').lower().replace('-', '') != 'utf8':
                stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
