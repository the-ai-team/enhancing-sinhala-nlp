import threading, time, sys
from translate import deep_translate


class MultiThreadHandler:
    def __init__(self):
        self.MAX_REQUESTS_PER_SECOND = 5
        self.request_times = []
        self.request_lock = threading.Lock()
        self.print_lock = threading.Lock()

    def safe_print(self, *args, **kwargs):
        with self.print_lock:
            print(*args, **kwargs)
            sys.stdout.flush()

    def rate_limited_translate(self, text: str) -> str:
        with self.request_lock:
            current_time = time.time()

            # Remove old request times
            self.request_times = [t for t in self.request_times if current_time - t < 1]

            # If we've made too many requests recently, wait
            if len(self.request_times) >= self.MAX_REQUESTS_PER_SECOND:
                time.sleep(1 - (current_time - self.request_times[0]))

            self.request_times.append(time.time())

        return deep_translate(text)


mth = MultiThreadHandler()
