# Topiq for Sublime Text

Syntax highlighting for Topiq source (`.tq`) and QUON documents (`.quon`),
and the Topiq language server, `tqls`, through the [LSP] package.

## Installing

Copy or link this directory into Sublime Text's `Packages` directory as
`Topiq` (Preferences > Browse Packages…). On Linux, for example:

```sh
ln -s "$PWD/editors/sublime/Topiq" ~/.config/sublime-text/Packages/Topiq
```

The syntaxes work on their own. For diagnostics, hover, go to definition,
the outline and completion, also:

1. Install `tqls`, so it is on `PATH`:

   ```sh
   cargo install --path tqls
   ```

2. Install the [LSP] package with Package Control.

The server starts when a `.tq` file is opened. `.quon` files are highlighted
but not checked, since a QUON document is not a program.

## Settings

Preferences > Package Settings > Topiq > Settings overrides the defaults in
`Topiq.sublime-settings`:

- `command`: how to start the server, `["tqls"]` by default. Give the full
  path if `tqls` is not on `PATH`.
- `initialization_options.unitPath`: directories searched for imported
  units, besides the importing file's own, relative to the window's first
  folder.
- `enabled`: `false` keeps the syntaxes but not the server.

Per project, the same keys go under `"settings" > "LSP" > "Topiq"` in the
`.sublime-project` file:

```json
{
  "settings": {
    "LSP": {
      "Topiq": { "initialization_options": { "unitPath": ["units"] } }
    }
  }
}
```

[LSP]: https://packagecontrol.io/packages/LSP
