@tool
extends EditorPlugin
## Editor plugin (enabled in project.godot) that adds the export of the Python worker: Godot exports
## only the resources it knows, so the scripts of res://python/ would be left out of the package.
## [GskExport] puts them in the package (where [code]worker.gd[/code] finds them and copies them out
## to user://python/ when no copy is next to the executable) and next to the exported executable
## (python/gsk_worker.py, the first place the interface looks): in a folder python/ beside it, or,
## when the project is exported as a .zip archive, in a folder python/ inside the archive.

var _export: EditorExportPlugin


func _enter_tree() -> void:
   _export = GskExport.new()
   add_export_plugin(_export)


func _exit_tree() -> void:
   remove_export_plugin(_export)
   _export = null


## The export of the files of res://python/.
class GskExport extends EditorExportPlugin:
   var _target := ""

   func _get_name() -> String:
      return "GraphoskopPython"

   func _export_begin(_features: PackedStringArray, _is_debug: bool, path: String, _flags: int) -> void:
      _target = path
      for file in DirAccess.get_files_at("res://python"):
         if file.get_extension() == "py":
            add_file("res://python/" + file, FileAccess.get_file_as_bytes("res://python/" + file), false)

   func _export_end() -> void:
      if _target.is_empty():
         return
      var files := Array(DirAccess.get_files_at("res://python")).filter(func(f): return f.get_extension() == "py")
      if _target.get_extension().to_lower() != "zip":
         var folder := _target.get_base_dir().path_join("python")
         DirAccess.make_dir_recursive_absolute(folder)
         for file in files:
            DirAccess.copy_absolute(ProjectSettings.globalize_path("res://python/" + file), folder.path_join(file))
         return
      var reader := ZIPReader.new()   # a resource pack (Export PCK/ZIP) has them already
      if reader.open(_target) != OK or reader.file_exists("python/gsk_worker.py"):
         return
      reader.close()
      var zip := ZIPPacker.new()
      if zip.open(_target, ZIPPacker.APPEND_ADDINZIP) != OK:
         return
      for file in files:
         zip.start_file("python/" + file)
         zip.write_file(FileAccess.get_file_as_bytes("res://python/" + file))
         zip.close_file()
      zip.close()
