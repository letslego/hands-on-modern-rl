# rollout_worker.py

class RolloutWorker:
    """
     Agent Loop，。
    : [(prompt, response_1, obs_1, response_2, obs_2, ..., final_response), reward]
    """

    def __init__(self, policy, env, max_turns=5):
        self.policy = policy
        self.env = env
        self.max_turns = max_turns

    def rollout(self, prompt: str, reward_fn) -> dict:
        """
         Agent Loop， reward。
        reward_fn: callable(trajectory) -> float
        """
        messages = [{"role": "user", "content": prompt}]
        trajectory = {"prompt": prompt, "interactions": []}

        for turn in range(self.max_turns):
            # 
            context = self._format_context(messages)
            model_output = self.policy.generate(context)

            # ：
            action = self._parse_action(model_output)

            if action["type"] == "finish":
                trajectory["interactions"].append({
                    "turn": turn,
                    "response": model_output,
                    "action": action,
                    "observation": None,
                })
                trajectory["final_response"] = action.get("answer", model_output)
                break

            # 
            obs = self.env.step(action["type"], action["args"])

            trajectory["interactions"].append({
                "turn": turn,
                "response": model_output,
                "action": action,
                "observation": obs["observation"],
            })

            messages.append({"role": "assistant", "content": model_output})
            messages.append({"role": "user", "content": f":\n{obs['observation']}"})

            if obs.get("done"):
                break

        # 
        trajectory["reward"] = reward_fn(trajectory)
        return trajectory

    def _format_context(self, messages):
        """ prompt。"""
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            else:
                parts.append(f"Assistant: {msg['content']}")
        return "\n".join(parts)

    def _parse_action(self, model_output: str) -> dict:
        """
        。
        ：。
        """
        if "```python" in model_output:
            # 
            code = model_output.split("```python")[1].split("```")[0]
            return {"type": "execute_code", "args": {"code": code}}
        elif "FINAL ANSWER:" in model_output:
            answer = model_output.split("FINAL ANSWER:")[1].strip()
            return {"type": "finish", "answer": answer}
        else:
            return {"type": "execute_code", "args": {"code": model_output}}
