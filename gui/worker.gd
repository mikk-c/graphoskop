extends Node
## Connection to the Python worker (python/gsk_worker.py), which runs the library.
##
## The worker is a separate process, started with the user's Python interpreter. The interface
## listens on a free local TCP port, passes it to the worker, and exchanges frames with it: 8 bytes
## (header and payload lengths, little-endian), a JSON header and a binary payload. Requests are
## answered in order; the reply of each goes to the callback given to [method request].

## Emitted when the worker has imported the library; [param info] is its "ready" header (Python
## and library versions, whether CUDA is available).
signal started(info: Dictionary)
## Emitted when the worker cannot start or the library cannot be imported.
signal failed(message: String)
## Emitted while a function with a progress bar runs: [param header] has name, fraction, elapsed
## and remaining (seconds), or done = true when the bar closes.
signal progress(header: Dictionary)

## Header of the "ready" frame, empty until the worker is ready.
var info := {}
var _hello := {}
var _server := TCPServer.new()
var _peer: StreamPeerTCP
var _pid := -1
var _buffer := PackedByteArray()
var _next_id := 0
var _callbacks := {}
var _queue := []


## Path of the worker script: next to the project when run from the editor, next to the
## executable when exported.
static func script_path() -> String:
   if OS.has_feature("editor"):
      return ProjectSettings.globalize_path("res://python/gsk_worker.py")
   return OS.get_executable_path().get_base_dir().path_join("python/gsk_worker.py")


## Starts the worker with the interpreter [param python] (a path, or a command on the PATH).
func start(python: String) -> void:
   stop()
   var err := _server.listen(0, "127.0.0.1")
   if err != OK:
      failed.emit("Cannot open a local port for the Python worker (error %d)." % err)
      return
   _pid = OS.create_process(python, [script_path(), str(_server.get_local_port())])
   if _pid <= 0:
      failed.emit("Cannot run the Python interpreter \"%s\"." % python)


## Stops the worker, if running; pending requests are dropped (their callbacks are never called).
func stop() -> void:
   if _pid > 0 and OS.is_process_running(_pid):
      OS.kill(_pid)
   _pid = -1
   _peer = null
   _server.stop()
   _buffer = PackedByteArray()
   _queue.clear()
   _callbacks.clear()
   info = {}


## Sends a request (a header with "cmd"), and calls [param callback] with the reply header and
## payload when it comes. Requests sent before the worker is ready wait in a queue.
func request(header: Dictionary, callback: Callable) -> void:
   _next_id += 1
   header["id"] = _next_id
   _callbacks[_next_id] = callback
   if info.is_empty():
      _queue.append(header)
   else:
      _send(header)


## Asks the worker to interrupt the request it is running (sent at once, not after the queued
## requests); the request then fails with the error "Aborted".
func abort() -> void:
   if _peer != null and not info.is_empty():
      _send({"cmd": "abort"})


## Whether requests are waiting for their replies.
func busy() -> bool:
   return not _callbacks.is_empty()


func _send(header: Dictionary) -> void:
   var h := JSON.stringify(header, "", false).to_utf8_buffer()   # keys in their order (arguments of calls)
   var frame := PackedByteArray()
   frame.resize(8)
   frame.encode_u32(0, h.size())
   frame.encode_u32(4, 0)
   frame.append_array(h)
   _peer.put_data(frame)


func _process(_delta: float) -> void:
   if _pid <= 0:
      return
   if _peer == null:
      if _server.is_connection_available():
         _peer = _server.take_connection()
      elif not OS.is_process_running(_pid):
         _pid = -1
         failed.emit("The Python worker stopped before connecting: check the interpreter.")
      return
   _peer.poll()
   if _peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
      stop()
      failed.emit("The Python worker stopped.")
      return
   var available := _peer.get_available_bytes()
   while available > 0:
      var got: Array = _peer.get_partial_data(available)
      _buffer.append_array(got[1])
      available = _peer.get_available_bytes()
   _frames()


## Dispatches the complete frames in the buffer.
func _frames() -> void:
   var start := 0
   while _buffer.size() - start >= 8:
      var h := _buffer.decode_u32(start)
      var p := _buffer.decode_u32(start + 4)
      if _buffer.size() - start < 8 + h + p:
         break
      var header: Dictionary = JSON.parse_string(_buffer.slice(start + 8, start + 8 + h).get_string_from_utf8())
      var payload := _buffer.slice(start + 8 + h, start + 8 + h + p)
      start += 8 + h + p
      _dispatch(header, payload)
   if start > 0:
      _buffer = _buffer.slice(start)


func _dispatch(header: Dictionary, payload: PackedByteArray) -> void:
   match header.get("cmd"):
      "hello":
         _hello = header
         return
      "progress":
         progress.emit(header)
         return
      "ready":
         if header.ok:
            info = _hello.merged(header, true)
            started.emit(info)
            for h in _queue:
               _send(h)
            _queue.clear()
         else:
            failed.emit("The Python worker cannot import graphoskop:\n" + str(header.error))
         return
   var id := int(header.get("id", -1))
   if _callbacks.has(id):
      var callback: Callable = _callbacks[id]
      _callbacks.erase(id)
      callback.call(header, payload)


func _exit_tree() -> void:
   stop()
