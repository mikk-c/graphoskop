class_name Environments
## Python environments the worker can run in: the recently used interpreters, the conda
## environments (listed by conda in ~/.conda/environments.txt), the active virtual environment
## and the interpreters on the PATH.


## Interpreter of an environment folder (bin/python, or python.exe on Windows).
static func interpreter(folder: String) -> String:
   return folder.path_join("python.exe") if OS.get_name() == "Windows" else folder.path_join("bin/python")


## Name of an environment from the path of its interpreter: its folder for conda and virtual
## environments, the path itself otherwise.
static func name_of(python: String) -> String:
   var dir := python.get_base_dir()
   if dir.get_file() in ["bin", "Scripts"]:
      dir = dir.get_base_dir()
   if DirAccess.dir_exists_absolute(dir.path_join("conda-meta")):
      return "base" if DirAccess.dir_exists_absolute(dir.path_join("envs")) else dir.get_file()
   if FileAccess.file_exists(dir.path_join("pyvenv.cfg")):
      return dir.get_file()
   return python


## Existing interpreters, without duplicates: recent ones first, then conda environments, the
## active virtual environment and the PATH.
static func find() -> PackedStringArray:
   var found := PackedStringArray()
   var add := func(p: String):
      if not p.is_empty() and FileAccess.file_exists(p) and not found.has(p):
         found.append(p)
   for p in AppSettings.value("python", "recent", PackedStringArray()):
      add.call(p)
   var home := OS.get_environment("USERPROFILE" if OS.get_name() == "Windows" else "HOME")
   var conda := FileAccess.open(home.path_join(".conda/environments.txt"), FileAccess.READ)
   if conda:
      for line in conda.get_as_text().split("\n", false):
         add.call(interpreter(line.strip_edges()))
   if OS.has_environment("VIRTUAL_ENV"):
      add.call(interpreter(OS.get_environment("VIRTUAL_ENV")))
   var windows := OS.get_name() == "Windows"
   for dir in OS.get_environment("PATH").split(";" if windows else ":", false):
      for exe in (["python.exe"] if windows else ["python3", "python"]):
         add.call(dir.path_join(exe))
   return found


## Remembers [param python] as the interpreter of the worker, first among the recent ones.
static func choose(python: String) -> void:
   AppSettings.store("python", "path", python)
   var recent: PackedStringArray = AppSettings.value("python", "recent", PackedStringArray())
   var i := recent.find(python)
   if i >= 0:
      recent.remove_at(i)
   recent.insert(0, python)
   AppSettings.store("python", "recent", recent.slice(0, 10))
