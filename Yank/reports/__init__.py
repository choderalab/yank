try:
    from . import notebook
except ImportError:
    print("Rendering this notebook requires the following packages:\n"
          " - matplotlib\n"
          "These are not required to generate the notebook through the `yank analyze report` command\n"
          "The modules are not required to use the rest of YANK and no ImportError is raised,\n"
          "but the reports module cannot be used.")
