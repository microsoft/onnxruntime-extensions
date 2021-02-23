
class ONNXTraceSession:
    activated_sessions = []

    def __init__(self, inputs):
        self.inputs = inputs
        self.torch_ops = []

    def __enter__(self):
        self.activated_sessions.append(self)
        return self

    def __exit__(self, exec_type, exec_value, exec_tb):
        last = self.activated_sessions[-1]
        del self.activated_sessions[-1:]
        return last

    @classmethod
    def get_active_session(cls):
        return cls.activated_sessions[0] if cls.activated_sessions else None

    def set_outputs(self, output_list):
        pass
