"""Sphinx ReadTheDocs theme.
From https://github.com/ryan-roemer/sphinx-bootstrap-theme.
"""
import os


def get_html_theme_path():
    """Return list of HTML theme paths."""
    cur_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    return cur_dir