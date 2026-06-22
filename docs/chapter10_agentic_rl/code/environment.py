# environment.py
import subprocess
import tempfile
import os

class SandboxEnv:
    """：subprocess + """

    def __init__(self, timeout=10, max_memory=256 * 1024 * 1024):
        self.timeout = timeout
        self.max_memory = max_memory

    def step(self, action_type: str, action_args: dict) -> dict:
        """，。"""
        if action_type == "execute_code":
            return self._exec_code(action_args["code"])
        elif action_type == "finish":
            return {"observation": "", "done": True}
        else:
            return {"observation": f"Unknown action: {action_type}", "done": False}

    def _exec_code(self, code: str) -> dict:
        """， CPU 。"""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                result = subprocess.run(
                    ["python", f.name],
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                )
                os.unlink(f.name)
                return {
                    "observation": (result.stdout + result.stderr)[-500:],  # 
                    "done": False,
                }
        except subprocess.TimeoutExpired:
            return {"observation": "TIMEOUT", "done": True}
        except Exception as e:
            return {"observation": f"ERROR: {e}", "done": False}

    def reset(self):
        """（ episode ）。"""
        pass
