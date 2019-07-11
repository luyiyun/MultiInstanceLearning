import sys
import time


import progressbar as pb



def up():
    # My terminal breaks if we don't flush after the escape-code
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()

def down():
    # I could use '\x1b[1B' here, but newline is faster and easier
    sys.stdout.write('\n')
    sys.stdout.flush()


class ProgressBarNest(pb.ProgressBar):
    def __init__(level, *args, **kwargs):
        super(ProgressBarNest, self).__init__(*args, **kwargs)
        if level > 1:
            raise NotImplementedError
        else:
            self.level = level

    def start(*args, **kwargs):
        if self.level == 0:
            down()
        else:
            up()
        super(ProgressBarNest, self).start(*args, **kwargs)


def test1():
    # Total bar is at the bottom. Move down to draw it
    down()
    total = pb.ProgressBar(maxval=50)
    total.start()

    for i in range(1,51):
        # Move back up to prepare for sub-bar
        up()

        # I make a new sub-bar for every iteration, thinking it could be things
        # like "File progress", with total being total file progress.
        sub = pb.ProgressBar(maxval=50)
        sub.start()
        for y in range(51):
            sub.update(y)
            time.sleep(0.005)
        sub.finish()

        # Update total - The sub-bar printed a newline on finish, so we already
        # have focus on it
        total.update(i)
    total.finish()


def test2():
    total = ProgressBarNest(0, maxval=50)
    for i in total(range(1,51)):
        sub = ProgressBarNest(1, maxval=50)
        for y in sub(range(51)):
            time.sleep(0.005)


if __name__ == '__main__':
    # test1()
    test2()
