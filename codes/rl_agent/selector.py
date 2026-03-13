import numpy as np

from rl_agent.agent import DQNAgent
# from rl_agent.agent import DQNAgent

class RLTestSelector:
    def __init__(self, config, label_data=None):
        self.agent = DQNAgent(state_dim=13, action_dim=2, config=config)
        self.pending_feedback = []
        self.delay_window = config["delay_window"]
        self.global_step = 0
        
        if label_data is not None:
            self.agent.warm_start(label_data)

    def build_state(self, change, score):
        #change是np list,将score转化为np list并和change拼接
        score = np.array(score)
        # print("score", score)
        change = np.squeeze(change)
        # print("change", change)
        return np.append(score,change)

    def select_action(self, state):
        return self.agent.select_action(state)

    def evaluate_test_result(self, result):
        # if result == 1:
        #     return 1.0
        # elif result == 0:
        #     return -0.2
        # else:
        #     return 0.0
        return 1.0 if result == 1 else -0.2
        
    def evaluate_test_result_res(self, result):
        # if result == 1:
        #     return 0.2
        # elif result == 0:
        #     return -1.0
        # else:
        #     return 0.0
        return 1.0 if result == 1 else -0.5
    
    def defer_feedback(self, change, score, y, count_in_t_window):
        self.pending_feedback.append({
            "change": change,
            "step": self.global_step,
            "state": self.build_state(change, score),
            "future_exposed": y,
            "count_in_t_window": count_in_t_window,
            "action": 0
        })
        return 0.0

    def store(self, state, action, reward, done=True):
        dummy_next = state
        self.agent.store_transition(state, action, reward, dummy_next, done)

    def learn(self):
        self.agent.learn()
        self.global_step += 1

    def update_delayed_feedback(self):
        to_remove = []
        for item in self.pending_feedback:
            if self.global_step - item["step"] >= item["count_in_t_window"]:
                # exposed = item["change"].get("future_exposed", False)
                reward = -1.0 if item["future_exposed"] == 1 else 0.0
                self.store(item["state"], item["action"], reward)
                to_remove.append(item)
        # self.pending_feedback = [x for x in self.pending_feedback if x not in to_remove]
        self.pending_feedback = [x for x in self.pending_feedback if not any(np.array_equal(x, item) for item in to_remove)]

    def update_delayed_feedback_1(self):
        to_remove = []
        for item in self.pending_feedback:
            if self.global_step - item["step"] >= item["count_in_t_window"]:
                # exposed = item["change"].get("future_exposed", False)
                reward = -1.0 if item["future_exposed"] == 1 else 0.5
                self.store(item["state"], item["action"], reward)
                to_remove.append(item)
        # self.pending_feedback = [x for x in self.pending_feedback if x not in to_remove]
        self.pending_feedback = [x for x in self.pending_feedback if not any(np.array_equal(x, item) for item in to_remove)]