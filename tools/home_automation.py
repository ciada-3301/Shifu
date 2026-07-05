"""
1. A tool that allows shifu to control home appliances and gadgets using an API hitting an outlaid server.
2. Unified JSON layout
3. Auto discovered by Shifu

"""

import json
from langchain_core.tools import tool
import requests

SERVER_ADDRESS = "127.0.0.1:5000"