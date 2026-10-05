import os

from tiredai.startup import wait_for_file


class Clock:
    """Fake time: sleeping advances it, and `on_sleep` runs after each sleep."""

    def __init__(self, now: float, on_sleep=lambda clock: None):
        self.now = now
        self.on_sleep = on_sleep
        self.sleeps = 0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.sleeps += 1
        self.on_sleep(self)


def test_waits_for_the_file_to_appear(tmp_path):
    csv = tmp_path / "tires_sample_10k_sku.csv"

    def copy_in(clock):
        if clock.sleeps == 5:
            csv.write_text("sku,name\n")
            os.utime(csv, (clock.now, clock.now))

    clock = Clock(1_000.0, copy_in)
    log = []
    wait_for_file(csv, settle_seconds=2, poll_seconds=1, clock=clock.time, sleep=clock.sleep, log=log.append)

    assert clock.sleeps == 7  # appeared after 5 polls, then 2 seconds unchanged
    assert log == ["Waiting for tires_sample_10k_sku.csv: copy the tire catalog CSV into the data folder to continue."]


def test_a_file_still_being_copied_is_not_read_yet(tmp_path):
    csv = tmp_path / "tires.csv"
    csv.write_text("sku")

    def keep_writing(clock):
        if clock.sleeps <= 3:
            os.utime(csv, (clock.now, clock.now))

    clock = Clock(1_000.0, keep_writing)
    os.utime(csv, (clock.now, clock.now))
    log = []
    wait_for_file(csv, settle_seconds=2, poll_seconds=1, clock=clock.time, sleep=clock.sleep, log=log.append)

    assert clock.sleeps == 5  # last change after sleep 3, settled 2 seconds later
    assert log == []  # the file was there, so there is nothing to ask for


def test_an_existing_file_is_used_at_once(tmp_path):
    csv = tmp_path / "tires.csv"
    csv.write_text("sku")
    os.utime(csv, (500.0, 500.0))
    clock = Clock(1_000.0)

    wait_for_file(csv, clock=clock.time, sleep=clock.sleep, log=print)

    assert clock.sleeps == 0
