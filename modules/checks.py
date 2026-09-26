"""Feature availability checks used by the optional modules."""
from importlib.util import find_spec

HAS_FOLIUM = find_spec("folium") is not None and find_spec("streamlit_folium") is not None