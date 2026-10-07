"""Connects `tqls`, the Topiq language server, through the LSP package.

The server's command, selector and initialisation options are in
`Topiq.sublime-settings`, beside this file; override them from
Preferences > Package Settings > Topiq > Settings. Without the LSP package
installed this does nothing, and the syntaxes still work.
"""

try:
    from LSP.plugin import LspPlugin
except ImportError:
    LspPlugin = None

try:
    from LSP.plugin import AbstractPlugin, register_plugin, unregister_plugin
except ImportError:
    AbstractPlugin = None

import sublime

# the directory this package is installed as, which names its settings file
PACKAGE = __package__ or __name__.split(".")[0]
SETTINGS = "Topiq.sublime-settings"


if LspPlugin is not None:
    # LSP 2.14 and later derive the session name and the settings file,
    # `Packages/<package>/<package>.sublime-settings`, from the package
    class Tqls(LspPlugin):
        pass

    def plugin_loaded():
        Tqls.register()

    def plugin_unloaded():
        Tqls.unregister()

elif AbstractPlugin is not None:
    class Tqls(AbstractPlugin):
        @classmethod
        def name(cls):
            return PACKAGE

        @classmethod
        def configuration(cls):
            path = "Packages/{}/{}".format(PACKAGE, SETTINGS)
            return sublime.load_settings(SETTINGS), path

    def plugin_loaded():
        register_plugin(Tqls)

    def plugin_unloaded():
        unregister_plugin(Tqls)
