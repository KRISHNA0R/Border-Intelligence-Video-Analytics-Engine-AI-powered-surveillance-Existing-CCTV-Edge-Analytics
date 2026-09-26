"""BorderEye optional feature modules.

Every module in this package must import cleanly on its own and never
crash the core app: web_demo.py imports each inside try/except and
degrades to a sane message if a dependency is missing.
"""