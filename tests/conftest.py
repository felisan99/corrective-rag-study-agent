from dotenv import load_dotenv

# Runs once, before any test module imports the graph -- `main.py` calls this
# itself, but nothing does for a bare `pytest` invocation.
load_dotenv()
