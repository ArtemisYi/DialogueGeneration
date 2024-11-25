import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置文件路径
QUERIES_FILE = os.path.join(BASE_DIR, "queries.txt")
COMMAND_FILE = os.path.join(BASE_DIR, "command.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "output.json")