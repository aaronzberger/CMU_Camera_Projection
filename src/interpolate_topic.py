import rospy
from bisect import bisect


class Interpolation_Subscriber:
    def __init__(self, topic_name, topic_type, get_fn, queue_size=20):
        '''
        Parameters:
            topic_name (str): ROS topic name to subscribe to
            topic_type (any): type of message in ROS topic
            get_fn (fn): function to retrieve the desired values to track
                (return a list if multiple)
            queue_size (int): how many messages to store in the queue at once
        '''
        self.subscriber = rospy.Subscriber(
            topic_name, topic_type, self.callback)
        self.msg_queue = []
        self.time_queue = []
        self.queue_size = queue_size
        self.get_fn = get_fn
        self.topic_name = topic_name

        self.bounds_queue_size = 10
        self.bounds_queue_thresh = 0.5
        self.bounds_queue = [False] * self.bounds_queue_size
        self.warning_on = False
        self.counter = 0

    def callback(self, msg):
        data = self.get_fn(msg)
        self.msg_queue.append(data)
        self.time_queue.append(msg.header.stamp.to_sec())

        if len(self.msg_queue) >= self.queue_size:
            self.msg_queue.pop(0)
            self.time_queue.pop(0)

    def get(self, stamp):
        '''
        Get the interpolated values (same number & type as returned by get_fn)

        Parameters:
            stamp (rospy.Time): desired time stamp for estimation

        Returns:
            any: interpolated value(s)
        '''
        # Check for a high ratio of beginning selections
        pos = len(list(filter(lambda x: x is True, self.bounds_queue)))
        self.warning_on = pos / self.bounds_queue_size >= self.bounds_queue_thresh

        self.counter += 1
        if self.warning_on and self.counter % 5 == 0:
            print('\033[01;31m' + 'Warning: For topic {}, Interpolator\'s queue size is likely too small: {} of the last {} stamps were out of bounds'.format(
                self.topic_name, pos, self.bounds_queue_size) + '\033[02;0m')

        stamp = stamp.to_sec()

        # Find the two messages this is in between ->
        # where this time stamp should be inserted to keep the array sorted
        # print(self.topic_type, self.time_queue, stamp)
        idx = bisect(self.time_queue, stamp)

        self.bounds_queue.pop(0)

        if idx == 0:
            self.bounds_queue.append(True)
            return self.msg_queue[0]
        elif idx == len(self.msg_queue):
            return self.msg_queue[-1]

        self.bounds_queue.append(False)

        # Get the positions and times for the message directly before and after
        before, after = self.msg_queue[idx - 1:idx + 1]
        t_before, t_after = self.time_queue[idx - 1:idx + 1]

        # Calculate the fraction of time we are between the two messages
        time_thru = (stamp - t_before) / (t_after - t_before)

        # Interpolate the positions
        if type(before) != list:
            before = [before]
            after = [after]
        between = [before[i] + (time_thru * (after[i] - before[i]))
                   for i in range(len(before))]

        return between if len(between) > 1 else between[0]
