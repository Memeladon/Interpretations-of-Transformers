"""Обратная совместимость: см. ui/interactive_tone_style.py."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap()

if __name__ == "__main__":
    from ui.interactive_tone_style import main

    main()
    sys.exit(0)
