project = 'FMEngine'
copyright = '2023, Xiaozhe Yao'
author = 'Xiaozhe Yao'

extensions = []

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'furo'
html_static_path = ['_static']
extensions = ['myst_parser']
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}