import datetime

def log_event(message):
    with open("events.log", "a") as f:
        f.write(f"{datetime.datetime.now()} - {message}\n")