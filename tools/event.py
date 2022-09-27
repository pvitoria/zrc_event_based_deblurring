import numpy as np
from PIL import Image

X_COLUMN = 0
Y_COLUMN = 1
TIMESTAMP_COLUMN = 2
POLARITY_COLUMN = 3


class EventSequence(object):
    """Stores events in oldest-first order."""

    def __init__(
        self, features, image_height, image_width, start_time=None, end_time=None
    ):
        """Returns object of EventSequence class.

        Args:
            features: numpy array with events softed in oldest-first order. Inside,
                      rows correspond to individual events and columns to event
                      features (x, y, timestamp, polarity)

            image_height, image_width: widht and height of the event sensor. 
                                       Note, that it can not be inferred
                                       directly from the events, because
                                       events are spares.
            start_time, end_time: start and end times of the event sequence.
                                  If they are not provided, this function infers
                                  them from the events. Note, that it can not be
                                  inferred from the events when there is no motion.
        """
        self._features = features
        self._image_width = image_width
        self._image_height = image_height
        self._start_time = (
            start_time if start_time is not None else features[0, TIMESTAMP_COLUMN]
        )
        self._end_time = (
            end_time if end_time is not None else features[-1, TIMESTAMP_COLUMN]
        )

    def __len__(self):
        return self._features.shape[0]

    def duration(self):
        return self.end_time() - self.start_time()

    def start_time(self):
        return self._start_time

    def end_time(self):
        return self._end_time


    def filter_by_mask(self, mask, make_deep_copy=True):
        if make_deep_copy:
            return EventSequence(
                features=np.copy(self._features[mask]),
                image_height=self._image_height,
                image_width=self._image_width,
                start_time=self._start_time,
                end_time=self._end_time,
            )
        else:
            return EventSequence(
                features=self._features[mask],
                image_height=self._image_height,
                image_width=self._image_width,
                start_time=self._start_time,
                end_time=self._end_time,
            )

    def filter_by_timestamp(self, start_time, duration, make_deep_copy=True):
        """Returns event sequence filtered by the timestamp.
        
        The new sequence includes event in [start_time, start_time+duration).
        """
        end_time = start_time + duration
        mask = (start_time <= self._features[:, TIMESTAMP_COLUMN]) & (
            end_time > self._features[:, TIMESTAMP_COLUMN]
        )
        event_sequence = self.filter_by_mask(mask, make_deep_copy)
        event_sequence._start_time = start_time
        event_sequence._end_time = start_time + duration
        return event_sequence
