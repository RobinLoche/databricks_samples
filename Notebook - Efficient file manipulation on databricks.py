# Databricks notebook source
from typing import List
import os
import queue
import threading
import shutil
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import multiprocessing as mp
import subprocess
import sys
import time
from pyspark.sql.functions import col
from os import walk

# COMMAND ----------

def pathWithDBFS(path: str):
  """Function that return the correct path starting with /dbfs"""
  if path.startswith("/dbfs"):
    return path
  return "/dbfs" + path


def pathWithoutDBFS(path: str):
  """Function that return the correct path starting without /dbfs"""
  if path.startswith("/dbfs"):
    return path[5:]
  return path

# COMMAND ----------

def pathExists(path: str):
  """Function that test if a file (or a folder) exists"""
  return os.path.exists(pathWithDBFS(path))

# COMMAND ----------

def listFolders(path):
  """
  Function that list the folders inside a given path
    Parameters:
      path (string): the path of the folder to get the descendants of
    Return:
      list of string: the list of names of the folder inside the given path, or None if the folder doesn't exists
  """
  if not pathExists(path):
    return None

  folder_list = []
  for (dirpath, dirnames, filenames) in walk(pathWithDBFS(path)):
      folder_list.extend(dirnames)
      break
  return folder_list


def listFiles(path):
  """
  Function that list the files inside a given path
    Parameters:
      path (string): the path of the folder to get the descendants of
    Return:
      list of string: the list of names of the files inside the given path, or None if the folder doesn't exists
  """
  if not pathExists(path):
    return None
  file_list = []
  for (dirpath, dirnames, filenames) in walk(pathWithDBFS(path)):
      file_list.extend(filenames)
      break
  # Sometimes there is technicals files that ends with _$folder$ and that are not listed by dbutils.fs.ls, so I remove them
  file_list = [filename for filename in file_list if not filename.endswith("_$folder$")]
  return file_list

# COMMAND ----------

" | ".join(listFolders("/databricks-datasets/"))

# COMMAND ----------

" | ".join(listFiles("/databricks-datasets/"))

# COMMAND ----------

def getFolderTreeList(root_path):
  """
  Function that return the list of all the parents (and the current folder name) folders until the root (in order from / to folder)
  """
  folders = []
  path = os.path.normpath(root_path)
  while 1:
    path, folder = os.path.split(path)
    if folder != "":
      folders.append(folder)
    elif path != "":
      folders.append(path)
      break
  folders.reverse()
  return folders

# COMMAND ----------

getFolderTreeList("/databricks-datasets/airlines")

# COMMAND ----------

def sizeByteToHuman(nb_bytes, base = 1024, byte_display = "B"):
  """
  Function to return a file size given in bytes to a human readable version (using the given base, aka 1000 for Kb or 1024 for Kib)
  """
  if base == 1024:
    units = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]
    default = "Yi"
  elif base == 1000:
    units = ["", "K", "M", "G", "T", "P", "E", "Z"]
    default = "Y"
  else:
    raise ValueError("Unsupported base " + str(base) + ", please use 1000 or 1024.")
  for unit in units:
    if abs(nb_bytes) < float(base):
      return f"{nb_bytes:3.1f} {unit}{byte_display}"
    nb_bytes /= float(base)
  return f"{nb_bytes:.1f} {default}{byte_display}"

# COMMAND ----------

sizeByteToHuman(123456, 1000) + " | " + sizeByteToHuman(123456789, 1024)

# COMMAND ----------

def folderSize(path):
  """Function that returns the size (in bytes) of the folder"""
  process = subprocess.Popen(["du", "-sb", pathWithDBFS(path)], stdout = subprocess.PIPE)
  output, error = process.communicate()
  if(error is not None):
    return "Error for " + path + ": " + error
  return int(str(output)[2:-3].split('\\t')[0])

# COMMAND ----------

sizeByteToHuman(folderSize("/databricks-datasets/airlines"))

# COMMAND ----------

class FolderScanner:
  """
  Class that compute the size of a folder by scanning each subfolder in a different sub-process. 
  Once initialized (my_scan = FolderScanner(path)), you can get the folder size by accessing the attribut "total" (my_scan.total).
  """
  element_queue = queue.Queue()
  nb_cpu = mp.cpu_count()
  nb_folders_finished = 0
  total = 0
  lock = threading.Lock()
  progress_bar = None
  validate = False

  def __init__(self, root_path: str, nb_min_parallel: int = mp.cpu_count() * 3):
    # Clean and check source dir
    self.root_path = os.path.abspath(pathWithDBFS(root_path))
    if not os.path.exists(self.root_path):
      raise ValueError(f'Error: source directory {self.root_path} does not exist.')
    self.dispatch_workers()    

  def getElementSize(self):
    while True:
      path = self.element_queue.get()
      folds = listFolders(path)
      files = listFiles(path)
      # Recurs on folders
      for fold in folds:
        self.element_queue.put(os.path.join(path, fold))
      total = 0
      # Add files size
      for file in files:
        total += os.stat(os.path.join(path, file)).st_size # in bytes
      with self.lock:
        self.total += total
        self.nb_folders_finished += 1
      self.element_queue.task_done()

  def dispatch_workers(self):
    for i in range(self.nb_cpu):
      t = threading.Thread(target = self.getElementSize)
      t.daemon = True
      t.start()
    print(f'{self.nb_cpu} analysing deamons started.')
    self.element_queue.put(self.root_path)
    self.element_queue.join()
    print("Analyze finished, folder is of size: " + sizeByteToHuman(self.total) + " (" + str(self.total) + " bytes)")

# COMMAND ----------

res = FolderScanner("/dbfs/databricks-datasets/Rdatasets")
# Use res.total to get the exact number of bytes

# COMMAND ----------

def moveFolder(src_path, dest_path, overwrite = False, verbose = True):
  """Function to move a folder or a file from a source to a destination"""
  verboseprint = print if verbose else lambda *a, **k: None
  if not pathExists(pathWithDBFS(src_path)):
    raise FileNotFoundError("Path " + src_path + " does not exists.")
  if pathExists(pathWithDBFS(dest_path)):
    if overwrite:
      verboseprint("Destination folder already exists, deleting it...")
      dbutils.fs.rm(pathWithoutDBFS(dest_path), recurse = True)
    else:
      raise FileExistsError("Path " + dest_path + " already exists.")
  verboseprint("Copy the folder...")
  shutil.move(pathWithDBFS(src_path), pathWithDBFS(dest_path))

# COMMAND ----------

# code adapted from https://github.com/ikonikon/fast-copy/blob/master/fast-copy.py
class FastCopy:
  file_queue = queue.Queue()
  total_files = 0
  copy_count = 0
  lock_bar = threading.Lock()
  lock_folder = threading.Lock()
  progress_bar = None
  validate = False
  running = False

  def __init__(self, src_dir: str, dest_dir: str, validate: bool = False):
    self.validate = validate
    # Clean and check source dir
    self.src_dir = os.path.abspath(pathWithDBFS(src_dir))
    if not os.path.exists(self.src_dir):
      raise ValueError('Error: source directory {} does not exist.'.format(self.src_dir))
    # Clean and check output dir
    self.dest_dir = os.path.abspath(pathWithDBFS(dest_dir))
    # create output dir
    if not os.path.exists(self.dest_dir):
      print('Destination folder {} does not exist - creating now.'.format(self.dest_dir))
      os.makedirs(self.dest_dir)

    print("Listing the files to copy...")
    file_list = []
    for root, _, files in os.walk(os.path.abspath(self.src_dir)):
      for file in files:
        file_list.append(os.path.join(root, file))
    self.total_files = len(file_list)
    print("{} files to copy from {} to {}".format(self.total_files, self.src_dir, self.dest_dir))
    self.dispatch_workers(file_list)

  def single_copy(self):
    while self.running:
      try:
        # Get with a timeout to check self.running every seconds
        file = self.file_queue.get(timeout = 1)
      except:
        continue
      # If no except we got a file
      dest = self.dest_dir + file[len(self.src_dir):]
      dest_folder = os.path.split(dest)[0]
      if not pathExists(dest_folder):
        with self.lock_folder:
          res = Path(dest_folder + "/").mkdir(parents = True, exist_ok = True)
      try:
        res = shutil.copyfile(file, dest)
      except:
        print("Error while copying " + file + " to " + dest)
        raise
      self.file_queue.task_done()
      with self.lock_bar:
        self.progress_bar.update(1)

  def dispatch_workers(self, file_list: List[str]):
    n_threads = 16
    self.running = True
    for i in range(n_threads):
      t = threading.Thread(target=self.single_copy)
      t.daemon = True
      t.start()
    print('{} copy deamons started.'.format(n_threads))
    self.progress_bar = tqdm(total=self.total_files, position=0, leave=True)
    for file_name in file_list:
      self.file_queue.put(file_name)
    self.file_queue.join()
    self.running = False
    time.sleep(1) # Wait a little just to be sure to update the progress bar and stop the threads
    self.progress_bar.close()
    # Validate
    if self.validate:
      print("Listing results...")
      file_list = []
      for root, _, files in os.walk(os.path.abspath(self.dest_dir)):
        for file in files:
          file_list.append(os.path.join(root, file))
      print('{}/{} files copied successfully.'.format(len(file_list), self.total_files))
      if len(file_list) != self.total_files:
        raise Exception("Error while copying the files, one or several files missing")
    print("Copy finished.")

# COMMAND ----------

FastCopy("/dbfs/databricks-datasets/Rdatasets", "/FileStore/test/")

# COMMAND ----------

# MAGIC %fs ls /FileStore/test

# COMMAND ----------

dbutils.fs.cp("/databricks-datasets/Rdatasets", "/FileStore/test/", recurse = True)
