

class TaskRegistry:
    def __init__(self):
        self.tasks = {}

    def register(self, task):
        self.tasks[task.__name__] = task

    def get_task(self, task_name):
        return self.tasks[task_name]

task_reg = TaskRegistry()
