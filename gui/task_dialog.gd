extends AcceptDialog
## Dialog of a task of the library that takes time (loading, generating, laying out, analyzing): it
## opens when the task has run for [member Delay] (0.8 s), with what runs, its progress bar (for
## functions with one), the time elapsed and left, and an Abort button. Abort asks the worker to
## interrupt the task, which happens as soon as Python code runs again; if it has not stopped
## after 5 seconds (a long call into compiled code), the button becomes Force stop, which restarts
## the worker (the open networks are rebuilt from their code). It opens above the dialog that
## started the task (as its child window), and moves to the main window if that dialog closes
## while the task runs. Closing the dialog leaves the task
## running (the status bar still shows it). The dialog updates once per second at most, so it does
## not take time from the task.

## Emitted when the user asks to abort the task.
signal abort_requested
## Emitted when the user asks to stop the worker, since the task did not abort.
signal force_stop_requested
## Function returning the window the dialog opens in (set by the main scene: the dialog that
## started the task, so that this one is drawn above it, or the main window).
var host := Callable()

var _start := 0.0
var _remaining = null
var _aborting := false
var _watched: Window = null


func _ready() -> void:
   confirmed.connect(_on_button)
   %Delay.timeout.connect(_open)
   %Clock.timeout.connect(_tick)
   %Grace.timeout.connect(func(): ok_button_text = "Force stop"; get_ok_button().disabled = false)


## A task starts, described by [param text].
func begin(text: String) -> void:
   %Task.text = text
   %Bar.visible = false
   _remaining = null
   _aborting = false
   ok_button_text = "Abort"
   get_ok_button().disabled = false
   _start = Time.get_ticks_msec() / 1000.0
   %Delay.start()


## The task ended: the dialog closes (or does not open).
func end() -> void:
   %Delay.stop()
   %Clock.stop()
   %Grace.stop()
   hide()


## Progress frame of the worker (see [code]ProgressBar[/code] in the worker).
func progress(header: Dictionary) -> void:
   if header.get("done", false):
      %Bar.visible = false
      _remaining = null
      return
   %Bar.visible = true
   %Bar.value = header.fraction
   %Bar.tooltip_text = header.name
   _remaining = header.get("remaining")
   _tick()


func _open() -> void:
   var window: Window = host.call() if host.is_valid() else get_tree().root
   _move_to(window)
   _tick()
   reset_size()
   popup_centered()
   %Clock.start()


## Makes the dialog a child of [param window] (drawn above it); if that window closes while the
## task runs, the dialog moves to the main window.
func _move_to(window: Window) -> void:
   if get_parent() != window:
      reparent(window, false)
   if _watched != null and _watched.visibility_changed.is_connected(_host_closed):
      _watched.visibility_changed.disconnect(_host_closed)
   _watched = window if window != get_tree().root else null
   if _watched != null:
      _watched.visibility_changed.connect(_host_closed)


func _host_closed() -> void:
   if _watched != null and not _watched.visible and not %Clock.is_stopped():
      _reopen.call_deferred()   # not while the host emits its signal


func _reopen() -> void:
   _move_to(get_tree().root)
   popup_centered()


func _tick() -> void:
   var elapsed := Time.get_ticks_msec() / 1000.0 - _start
   var text := "%s elapsed" % _clock(elapsed)
   if _remaining != null:
      text += ", about %s left" % _clock(float(_remaining))
   if _aborting:
      text += " · aborting…"
   %Time.text = text


static func _clock(s: float) -> String:
   return "%d:%02d" % [int(s) / 60, int(s) % 60]


func _on_button() -> void:
   if _aborting:
      force_stop_requested.emit()
      return
   _aborting = true
   ok_button_text = "Aborting…"
   get_ok_button().disabled = true
   %Grace.start()
   _tick()
   abort_requested.emit()
