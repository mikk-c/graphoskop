class_name AppSettings
## Memory of the application between sessions (user://settings.cfg): the Python interpreter and
## the recent ones, the last folders of the file dialogs, the last choices of the load dialog and
## the size of the window.

## Settings file.
const PATH := "user://settings.cfg"


## Value of [param key] in [param section], or [param default] if never stored.
static func value(section: String, key: String, default: Variant = null) -> Variant:
   var config := ConfigFile.new()
   config.load(PATH)
   return config.get_value(section, key, default)


## Stores [param v] as [param key] of [param section].
static func store(section: String, key: String, v: Variant) -> void:
   var config := ConfigFile.new()
   config.load(PATH)
   config.set_value(section, key, v)
   config.save(PATH)


## Remembers the folder of [param file] as the last folder of the dialog [param dialog].
static func store_folder(dialog: String, file: String) -> void:
   store("folders", dialog, file.get_base_dir())


## Last folder of the dialog [param dialog], or else [param fallback], or else the home folder.
static func folder(dialog: String, fallback := "") -> String:
   var f: String = value("folders", dialog, fallback)
   if f.is_empty() or not DirAccess.dir_exists_absolute(f):
      f = fallback if not fallback.is_empty() and DirAccess.dir_exists_absolute(fallback) else OS.get_environment("USERPROFILE" if OS.get_name() == "Windows" else "HOME")
   return f
