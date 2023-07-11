class MyQueue:
    def __init__(self):
        self._readyStack = []
        self._pushStack = []

    def push(self, x: int) -> None:
        self._pushStack.append(x)

    def pop(self) -> int:
        if len(self._readyStack) == 0:
            while len(self._pushStack) > 0:
                self._readyStack.append(self._pushStack.pop(-1))
        return self._readyStack.pop()

    def peek(self) -> int:
        if len(self._readyStack) == 0:
            while len(self._pushStack) > 0:
                self._readyStack.append(self._pushStack.pop(-1))
        return self._readyStack[-1]

    def empty(self) -> bool:
        return len(self._readyStack) + len(self._pushStack) == 0


class MyStack:
    def __init__(self):
        self.queue = []
        self.stack = []

    def push(self, x: int) -> None:
        self.queue.append(x)

    def pop(self) -> int:
        self.move()
        return self.stack.pop(0)

    def top(self) -> int:
        self.move()
        return self.stack[0]

    def move(self) -> None:
        tempst = []
        while len(self.queue) > 0:
            l = len(self.queue)
            for _ in range(0, l - 1):
                # move n-1 front to end
                self.queue.append(self.queue.pop(0))
            tempst.append(self.queue.pop(0))
        while len(self.stack) > 0:
            tempst.append(self.stack.pop(0))
        self.stack = tempst

    def empty(self) -> bool:
        return 0 == len(self.queue)
