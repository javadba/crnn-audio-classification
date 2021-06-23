import json
import logging
from datetime import datetime

Debug =True
logging.basicConfig(level=logging.INFO, format='')

class Logger:
    """
    Training process logger

    Note:
        Used by BaseTrainer to save training history.
    """
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)

def p(msg): print(f'[{str(datetime.now())[6:-3]}] {msg}', flush=True)

def debug(msg):
  if Debug: p(msg)

def error(msg,e):
  emsg = tracerr(msg,True)
  omsg = f"{emsg} { str(e) if e else ''}"
  p(omsg)
  
def tracerr(msg, printException=True):
  import pickle
  import sys
  import traceback
  if printException:
    exc_type, exc_value, tb = sys.exc_info()
    # Save the traceback and attach it to the exception object
    exc_lines = traceback.format_exception(exc_type, exc_value, tb)
    exc_value = ''.join(exc_lines)
    # errmsg = "  %s" % (pickle.dumps(exc_value))
    errmsg = "  %s" %exc_value
  else:
    errmsg = ""
  return "[%s] ERROR: %s%s\n" % (datetime.now(), msg, errmsg)

