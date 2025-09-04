import dataclasses
from datetime import timedelta
from enum import Enum
from os import PathLike
from typing import Optional, Union, IO

import numpy as np
import numpy.typing as npt
from slider import Beatmap, Circle, Slider, Spinner, HoldNote, TimingPoint
from slider.curve import Linear, Catmull, Perfect, MultiBezier
from transformers import FeatureExtractionMixin, AutoFeatureExtractor

from cm3p import CM3PConfig


class EventType(Enum):
    CIRCLE = "circle"
    SPINNER = "spinner"
    SPINNER_END = "spinner_end"
    SLIDER_HEAD = "slider_head"
    BEZIER_ANCHOR = "bezier_anchor"
    PERFECT_ANCHOR = "perfect_anchor"
    CATMULL_ANCHOR = "catmull_anchor"
    RED_ANCHOR = "red_anchor"
    LAST_ANCHOR = "last_anchor"
    SLIDER_END = "slider_end"
    REPEAT_END = "repeat_end"
    BEAT = "beat"
    MEASURE = "measure"
    TIMING_POINT = "timing_point"
    KIAI_ON = "kiai_on"
    KIAI_OFF = "kiai_off"
    HOLD_NOTE = "hold_note"
    HOLD_NOTE_END = "hold_note_end"
    SCROLL_SPEED_CHANGE = "scroll_speed_change"
    DRUMROLL = "drumroll"
    DRUMROLL_END = "drumroll_end"
    DENDEN = "denden"
    DENDEN_END = "denden_end"


@dataclasses.dataclass
class Group:
    event_type: EventType = None
    time: int = 0
    has_time: bool = False
    snapping: int = None
    distance: int = None
    x: int = None
    y: int = None
    mania_column: int = None
    new_combo: bool = False
    hitsounds: list[int] = dataclasses.field(default_factory=list)
    samplesets: list[int] = dataclasses.field(default_factory=list)
    additions: list[int] = dataclasses.field(default_factory=list)
    volumes: list[int] = dataclasses.field(default_factory=list)
    scroll_speed: float = None


def merge_groups(groups1: list[Group], groups2: list[Group]) -> list[Group]:
    """Merge two lists of groups in a time sorted manner. Assumes both lists are sorted by time.

    Args:
        groups1: List of groups.
        groups2: List of groups.

    Returns:
        merged_groups: Merged list of groups.
    """
    merged_groups = []
    i = 0
    j = 0
    t1 = -np.inf
    t2 = -np.inf

    while i < len(groups1) and j < len(groups2):
        t1 = groups1[i].time or t1
        t2 = groups2[j].time or t2

        if t1 <= t2:
            merged_groups.append(groups1[i])
            i += 1
        else:
            merged_groups.append(groups2[j])
            j += 1

    # Add remaining groups from both lists
    merged_groups.extend(groups1[i:])
    merged_groups.extend(groups2[j:])
    return merged_groups


def speed_groups(groups: list[Group], speed: float) -> list[Group]:
    """Change the speed of a list of groups.

    Args:
        groups: List of groups.
        speed: Speed multiplier.

    Returns:
        sped_groups: Sped up list of groups.
    """
    sped_groups = []
    for group in groups:
        group.time = int(group.time / speed)
        sped_groups.append(group)

    return sped_groups


def get_median_mpb_beatmap(beatmap: Beatmap) -> float:
    # Not include last slider's end time
    last_time = max(ho.end_time if isinstance(ho, HoldNote) else ho.time for ho in beatmap.hit_objects(stacking=False))
    last_time = int(last_time.seconds * 1000)
    return get_median_mpb(beatmap.timing_points, last_time)


def get_median_mpb(timing_points: list[TimingPoint], last_time: float) -> float:
    # This is identical to osu! stable implementation
    this_beat_length = 0

    bpm_durations = {}

    for i in range(len(timing_points) - 1, -1, -1):
        tp = timing_points[i]
        offset = int(tp.offset.seconds * 1000)

        if tp.parent is None:
            this_beat_length = tp.ms_per_beat

        if this_beat_length == 0 or offset > last_time or (tp.parent is not None and i > 0):
            continue

        if this_beat_length in bpm_durations:
            bpm_durations[this_beat_length] += int(last_time - (0 if i == 0 else offset))
        else:
            bpm_durations[this_beat_length] = int(last_time - (0 if i == 0 else offset))

        last_time = offset

    longest_time = 0
    median = 0

    for bpm, duration in bpm_durations.items():
        if duration > longest_time:
            longest_time = duration
            median = bpm

    return median


def load_beatmap(beatmap: Union[str, PathLike, IO[str], Beatmap]) -> Beatmap:
    """Load a beatmap from a file path, file object, or Beatmap object.

    Args:
        beatmap: Beatmap file path, file object, or Beatmap object.

    Returns:
        beatmap: Loaded Beatmap object.
    """
    if isinstance(beatmap, (str, PathLike)):
        beatmap = Beatmap.from_path(beatmap)
    elif isinstance(beatmap, IO):
        beatmap = Beatmap.from_file(beatmap.name)
    return beatmap


def get_song_length(
        samples: np.ndarray = None,
        sample_rate: int = None,
        beatmap: Union[Beatmap | list[TimingPoint]] = None,
) -> float:
    if samples is not None and sample_rate is not None:
        return len(samples) / sample_rate

    if beatmap is None:
        return 0

    if isinstance(beatmap, Beatmap) and len(beatmap.hit_objects(stacking=False)) > 0:
        last_ho = beatmap.hit_objects(stacking=False)[-1]
        last_time = last_ho.end_time if hasattr(last_ho, "end_time") else last_ho.time
        return last_time.total_seconds() + 0.000999  # Add a small buffer to the last time

    timing = beatmap.timing_points if isinstance(beatmap, Beatmap) else beatmap
    if len(timing) == 0:
        return 0

    return timing[-1].offset.total_seconds() + 0.01


class CM3PBeatmapParser(FeatureExtractionMixin):
    """
    A class to parse CM3P beatmap files.
    """
    def __init__(
            self,
            add_timing: bool = True,
            add_snapping: bool = True,
            add_timing_points: bool = True,
            add_hitsounds: bool = True,
            add_distances: bool = True,
            add_positions: bool = True,
            add_kiai: bool = True,
            add_sv: bool = True,
            add_mania_sv: bool = True,
            mania_bpm_normalized_scroll_speed: bool = True,
            position_split_axes: bool = True,
            slider_version: int = 2,
            **kwargs,
    ):
        self.add_timing = add_timing
        self.add_snapping = add_snapping
        self.add_timing_points = add_timing_points
        self.add_hitsounds = add_hitsounds
        self.add_distances = add_distances
        self.add_positions = add_positions
        self.add_kiai = add_kiai
        self.add_sv = add_sv
        self.add_mania_sv = add_mania_sv
        self.mania_bpm_normalized_scroll_speed = mania_bpm_normalized_scroll_speed
        self.position_split_axes = position_split_axes
        self.slider_version = slider_version
        super().__init__(**kwargs)

    def parse_beatmap(
            self,
            beatmap: Union[str, PathLike, IO[str], Beatmap],
            speed: float = 1.0,
            song_length: Optional[float] = None
    ) -> list[Group]:
        """Parse an .osu beatmap.

        Each hit object is parsed into a list of Event objects, in order of its
        appearance in the beatmap. In other words, in ascending order of time.

        Args:
            beatmap: Beatmap object parsed from an .osu file.
            speed: Speed multiplier for the beatmap.
            song_length: Length of the song in seconds. If not provided, it will be calculated from the beatmap.

        Returns:
            events: List of Event object lists.
            event_times: List of event times.
        """
        beatmap = load_beatmap(beatmap)
        hit_objects = beatmap.hit_objects(stacking=False)
        last_pos = np.array((256, 192))
        groups = []

        for hit_object in hit_objects:
            if isinstance(hit_object, Circle):
                last_pos = self._parse_circle(hit_object, groups, last_pos, beatmap)
            elif isinstance(hit_object, Slider):
                if beatmap.mode == 1:
                    self._parse_drumroll(hit_object, groups, beatmap)
                else:
                    last_pos = self._parse_slider(hit_object, groups, last_pos, beatmap)
            elif isinstance(hit_object, Spinner):
                if beatmap.mode == 1:
                    self._parse_denden(hit_object, groups, beatmap)
                else:
                    last_pos = self._parse_spinner(hit_object, groups, beatmap)
            elif isinstance(hit_object, HoldNote):
                last_pos = self._parse_hold_note(hit_object, groups, beatmap)

        # Sort groups by time
        if len(groups) > 0:
            groups = sorted(groups, key=lambda x: x.time)
        result = list(groups)

        if self.add_mania_sv and beatmap.mode == 3:
            scroll_speed_events = self.parse_scroll_speeds(beatmap)
            result = merge_groups(scroll_speed_events, result)

        if self.add_kiai:
            kiai_events = self.parse_kiai(beatmap)
            result = merge_groups(kiai_events, result)

        if self.add_timing:
            timing_events = self.parse_timing(beatmap, song_length=song_length)
            result = merge_groups(timing_events, result)

        if speed != 1.0:
            result = speed_groups(result, speed)

        return result

    def parse_scroll_speeds(self, beatmap: Beatmap, speed: float = 1.0) -> list[Group]:
        """Extract all BPM-normalized scroll speed changes from a beatmap."""
        normalized = self.mania_bpm_normalized_scroll_speed
        groups = []
        median_mpb = get_median_mpb_beatmap(beatmap)
        mpb = median_mpb
        last_normalized_scroll_speed = -1

        for i, tp in enumerate(beatmap.timing_points):
            if tp.parent is None:
                mpb = tp.ms_per_beat
                scroll_speed = 1
            else:
                scroll_speed = -100 / tp.ms_per_beat

            if i == len(beatmap.timing_points) - 1 or beatmap.timing_points[i + 1].offset > tp.offset:
                normalized_scroll_speed = scroll_speed * median_mpb / mpb if normalized else scroll_speed

                if normalized_scroll_speed != last_normalized_scroll_speed or last_normalized_scroll_speed == -1:
                    self._add_group(
                        EventType.SCROLL_SPEED_CHANGE,
                        groups,
                        time=tp.offset,
                        beatmap=beatmap,
                        scroll_speed=normalized_scroll_speed,
                    )
                last_normalized_scroll_speed = normalized_scroll_speed

        if speed != 1.0:
            groups = speed_groups(groups, speed)

        return groups

    def parse_kiai(self, beatmap: Beatmap, speed: float = 1.0) -> list[Group]:
        """Extract all kiai information from a beatmap."""
        groups = []
        kiai = False

        for tp in beatmap.timing_points:
            if tp.kiai_mode == kiai:
                continue

            self._add_group(
                EventType.KIAI_ON if tp.kiai_mode else EventType.KIAI_OFF,
                groups,
                time=tp.offset,
                beatmap=beatmap,
            )
            kiai = tp.kiai_mode

        if speed != 1.0:
            groups = speed_groups(groups, speed)

        return groups

    def parse_timing(self, beatmap: Beatmap | list[TimingPoint], speed: float = 1.0, song_length: Optional[float] = None) -> list[Group]:
        """Extract all timing information from a beatmap."""
        timing = beatmap.timing_points if isinstance(beatmap, Beatmap) else beatmap
        assert len(timing) > 0, "No timing points found in beatmap."

        groups = []
        last_time = song_length or get_song_length(beatmap=beatmap)
        last_time = int(last_time * 1000)

        # Get all timing points with BPM changes
        timing_points = [tp for tp in timing if tp.bpm]

        for i, tp in enumerate(timing_points):
            # Generate beat and measure events until the next timing point
            next_tp = timing_points[i + 1] if i + 1 < len(timing_points) else None
            next_time = next_tp.offset.total_seconds() * 1000 - 10 if next_tp else last_time
            start_time = tp.offset.total_seconds() * 1000
            time = start_time
            measure_counter = 0
            beat_delta = tp.ms_per_beat
            while time <= next_time:
                if self.add_timing_points and measure_counter == 0:
                    event_type = EventType.TIMING_POINT
                elif measure_counter % tp.meter == 0:
                    event_type = EventType.MEASURE
                else:
                    event_type = EventType.BEAT

                self._add_group(
                    event_type,
                    groups,
                    time=timedelta(milliseconds=time),
                    add_snap=False,
                )

                # Exit early if the beat_delta is too small to avoid infinite loops
                if beat_delta <= 10:
                    break

                measure_counter += 1
                time = start_time + measure_counter * beat_delta

        if speed != 1.0:
            groups = speed_groups(groups, speed)

        return groups

    @staticmethod
    def uninherited_point_at(time: timedelta, beatmap: Beatmap):
        tp = beatmap.timing_point_at(time)
        return tp if tp.parent is None else tp.parent

    @staticmethod
    def hitsound_point_at(time: timedelta, beatmap: Beatmap):
        hs_query = time + timedelta(milliseconds=5)
        return beatmap.timing_point_at(hs_query)

    def scroll_speed_at(self, time: timedelta, beatmap: Beatmap) -> float:
        query = time
        tp = beatmap.timing_point_at(query)
        return self.tp_to_scroll_speed(tp)

    def tp_to_scroll_speed(self, tp: TimingPoint) -> float:
        if tp.parent is None or tp.ms_per_beat >= 0 or np.isnan(tp.ms_per_beat):
            return 1
        else:
            return np.clip(-100 / tp.ms_per_beat, 0.01, 10)

    def _get_snapping(self, time: timedelta, beatmap: Beatmap, add_snap: bool = True) -> int:
        """Add a snapping event to the event list.

        Args:
            time: Time of the snapping event.
            beatmap: Beatmap object.
            add_snap: Whether to add a snapping event.
        """
        if not add_snap or not self.add_snapping:
            return None

        tp = self.uninherited_point_at(time, beatmap)
        beats = (time - tp.offset).total_seconds() * 1000 / tp.ms_per_beat
        snapping = 0
        for i in range(1, 17):
            # If the difference between the time and the snapped time is less than 2 ms, that is the correct snapping
            if abs(beats - round(beats * i) / i) * tp.ms_per_beat < 2:
                snapping = i
                break

        return snapping

    def _get_hitsounds(self, time: timedelta, hitsound: int, addition: str, beatmap: Beatmap) -> tuple[int, int, int, int]:
        tp = self.hitsound_point_at(time, beatmap)
        tp_sample_set = tp.sample_type if tp.sample_type != 0 else 2  # Inherit to soft sample set
        addition_split = addition.split(":")
        sample_set = int(addition_split[0]) if addition_split[0] != "0" else tp_sample_set
        addition_set = int(addition_split[1]) if addition_split[1] != "0" else sample_set
        volume = int(addition_split[3]) if len(addition_split) > 3 and addition_split[3] != "0" else tp.volume

        sample_set = sample_set if 0 < sample_set < 4 else 1  # Overflow default to normal sample set
        addition_set = addition_set if 0 < addition_set < 4 else 1  # Overflow default to normal sample set
        hitsound = hitsound & 14  # Only take the bits for whistle, finish, and clap
        volume = np.clip(volume, 0, 100)

        return hitsound, sample_set, addition_set, volume

    def _get_position(self, pos: npt.NDArray, last_pos: npt.NDArray) -> tuple[int, int, int, npt.NDArray]:
        x, y, dist = None, None, None

        if self.add_distances:
            dist = int(np.linalg.norm(pos - last_pos))

        if self.add_positions:
            x = int(pos[0])
            y = int(pos[1])

        return x, y, dist, pos

    def _get_mania_column(self, pos: npt.NDArray, columns: int) -> int:
        column = int(np.clip(pos[0] / 512 * columns, 0, columns - 1))
        return column

    def _add_group(
            self,
            event_type: EventType,
            groups: list[Group],
            time: timedelta,
            *,
            beatmap: Beatmap = None,
            add_snap: bool = True,
            has_time: bool = True,
            pos: npt.NDArray = None,
            last_pos: npt.NDArray = None,
            new_combo: bool = False,
            hitsound_ref_times: list[timedelta] = None,
            hitsounds: list[int] = None,
            additions: list[str] = None,
            scroll_speed: Optional[float] = None,
    ) -> npt.NDArray:
        """Add a group of events to the event list."""
        group = Group(
            event_type=event_type,
            time=int(time.total_seconds() * 1000 + 1e-5)
        )

        if has_time:
            group.has_time = True
            group.snapping = self._get_snapping(time, beatmap, add_snap)
        if pos is not None:
            if beatmap.mode in [0, 2]:
                x, y, dist, last_pos = self._get_position(pos, last_pos)
                group.x = x
                group.y = y
                group.distance = dist
            elif beatmap.mode == 3:
                group.column = self._get_mania_column(pos, int(beatmap.circle_size))
        if new_combo and beatmap.mode in [0, 2]:
            group.new_combo = True
        if scroll_speed is not None:
            group.scroll_speed = scroll_speed
        if hitsound_ref_times is not None and self.add_hitsounds:
            for i, ref_time in enumerate(hitsound_ref_times):
                hitsound, sample_set, addition_set, volume = self._get_hitsounds(ref_time, hitsounds[i], additions[i], beatmap)
                group.hitsounds.append(hitsound)
                group.samplesets.append(sample_set)
                group.additions.append(addition_set)
                group.volumes.append(volume)

        groups.append(group)

        return last_pos

    def _parse_circle(self, circle: Circle, groups: list[Group], last_pos: npt.NDArray, beatmap: Beatmap) -> npt.NDArray:
        """Parse a circle hit object.

        Args:
            circle: Circle object.
            groups: List of groups to add to.
            last_pos: Last position of the hit objects.

        Returns:
            pos: Position of the circle.
        """
        return self._add_group(
            EventType.CIRCLE,
            groups,
            time=circle.time,
            beatmap=beatmap,
            pos=np.array(circle.position),
            last_pos=last_pos,
            new_combo=circle.new_combo,
            hitsound_ref_times=[circle.time],
            hitsounds=[circle.hitsound],
            additions=[circle.addition],
            scroll_speed=self.scroll_speed_at(circle.time, beatmap) if beatmap.mode == 1 else None,
        )

    def _parse_slider(self, slider: Slider, groups: list[Group], last_pos: npt.NDArray, beatmap: Beatmap) -> npt.NDArray:
        """Parse a slider hit object.

        Args:
            slider: Slider object.
            groups: List of groups to add to.
            last_pos: Last position of the hit objects.

        Returns:
            pos: Last position of the slider.
        """
        # Ignore sliders which are too big
        if len(slider.curve.points) >= 100:
            return last_pos

        last_pos = self._add_group(
            EventType.SLIDER_HEAD,
            groups,
            time=slider.time,
            beatmap=beatmap,
            pos=np.array(slider.position),
            last_pos=last_pos,
            new_combo=slider.new_combo,
            hitsound_ref_times=[slider.time],
            hitsounds=[slider.edge_sounds[0] if len(slider.edge_sounds) > 0 else 0],
            additions=[slider.edge_additions[0] if len(slider.edge_additions) > 0 else '0:0'],
            scroll_speed=self.scroll_speed_at(slider.time, beatmap) if self.add_sv else None,
        )

        duration: timedelta = (slider.end_time - slider.time) / slider.repeat
        control_point_count = len(slider.curve.points)

        def append_control_points(event_type: EventType, last_pos: npt.NDArray = last_pos) -> npt.NDArray:
            for i in range(1, control_point_count - 1):
                last_pos = add_anchor(event_type, i, last_pos)

            return last_pos

        def add_anchor(event_type: EventType, i: int, last_pos: npt.NDArray) -> npt.NDArray:
            return self._add_group(
                event_type,
                groups,
                time=slider.time + i / (control_point_count - 1) * duration if self.slider_version == 1 else slider.time,
                beatmap=beatmap,
                has_time=False,
                pos=np.array(slider.curve.points[i]),
                last_pos=last_pos,
            )

        if isinstance(slider.curve, Linear):
            last_pos = append_control_points(EventType.RED_ANCHOR, last_pos)
        elif isinstance(slider.curve, Catmull):
            last_pos = append_control_points(EventType.CATMULL_ANCHOR, last_pos)
        elif isinstance(slider.curve, Perfect):
            last_pos = append_control_points(EventType.PERFECT_ANCHOR, last_pos)
        elif isinstance(slider.curve, MultiBezier):
            for i in range(1, control_point_count - 1):
                if slider.curve.points[i] == slider.curve.points[i + 1]:
                    last_pos = add_anchor(EventType.RED_ANCHOR, i, last_pos)
                elif slider.curve.points[i] != slider.curve.points[i - 1]:
                    last_pos = add_anchor(EventType.BEZIER_ANCHOR, i, last_pos)

        if self.slider_version == 2:
            # Add last control point without time
            last_pos = self._add_group(
                EventType.LAST_ANCHOR,
                groups,
                time=slider.time,
                beatmap=beatmap,
                has_time=False,
                pos=np.array(slider.curve.points[-1]),
                last_pos=last_pos,
            )

        # Add body hitsounds and remaining edge hitsounds
        last_pos = self._add_group(
            EventType.SLIDER_END,
            groups,
            time=slider.time + duration,
            beatmap=beatmap,
            pos=np.array(slider.curve.points[-1]) if self.slider_version == 1 else None,
            last_pos=last_pos,
            hitsound_ref_times=[slider.time + timedelta(milliseconds=1)] + [slider.time + i * duration for i in range(1, slider.repeat)],
            hitsounds=[slider.hitsound] + [slider.edge_sounds[i] if len(slider.edge_sounds) > i else 0 for i in range(1, slider.repeat)],
            additions=[slider.addition] + [slider.edge_additions[i] if len(slider.edge_additions) > i else '0:0' for i in range(1, slider.repeat)],
        )

        return self._add_group(
            EventType.REPEAT_END,
            groups,
            time=slider.end_time,
            beatmap=beatmap,
            pos=np.array(slider.curve(1)),
            last_pos=last_pos,
            hitsound_ref_times=[slider.end_time],
            hitsounds=[slider.edge_sounds[-1] if len(slider.edge_sounds) > 0 else 0],
            additions=[slider.edge_additions[-1] if len(slider.edge_additions) > 0 else '0:0'],
        )

    def _parse_spinner(self, spinner: Spinner, groups: list[Group], beatmap: Beatmap) -> npt.NDArray:
        """Parse a spinner hit object.

        Args:
            spinner: Spinner object.
            groups: List of groups to add to.

        Returns:
            pos: Last position of the spinner.
        """
        self._add_group(
            EventType.SPINNER,
            groups,
            time=spinner.time,
            beatmap=beatmap,
        )

        self._add_group(
            EventType.SPINNER_END,
            groups,
            time=spinner.end_time,
            beatmap=beatmap,
            hitsound_ref_times=[spinner.end_time],
            hitsounds=[spinner.hitsound],
            additions=[spinner.addition],
        )

        return np.array((256, 192))

    def _parse_hold_note(self, hold_note: HoldNote, groups: list[Group], beatmap: Beatmap) -> npt.NDArray:
        """Parse a hold note hit object.

        Args:
            hold note: Hold note object.
            groups: List of groups to add to.

        Returns:
            pos: Last position of the spinner.
        """
        pos = np.array(hold_note.position)

        self._add_group(
            EventType.HOLD_NOTE,
            groups,
            time=hold_note.time,
            beatmap=beatmap,
            pos=pos,
            hitsound_ref_times=[hold_note.time],
            hitsounds=[hold_note.hitsound],
            additions=[hold_note.addition],
        )

        self._add_group(
            EventType.HOLD_NOTE_END,
            groups,
            time=hold_note.end_time,
            beatmap=beatmap,
            pos=pos,
        )

        return pos

    def _parse_drumroll(self, slider: Slider, groups: list[Group], beatmap: Beatmap):
        """Parse a drumroll hit object.

        Args:
            slider: Slider object.
            groups: List of groups to add to.
        """
        self._add_group(
            EventType.DRUMROLL,
            groups,
            time=slider.time,
            beatmap=beatmap,
            hitsound_ref_times=[slider.time],
            hitsounds=[slider.hitsound],  # Edge hitsounds are not supported in drumrolls
            additions=[slider.addition],
            scroll_speed=self.scroll_speed_at(slider.time, beatmap),
        )

        self._add_group(
            EventType.DRUMROLL_END,
            groups,
            time=slider.end_time,
            beatmap=beatmap,
        )

    def _parse_denden(self, spinner: Spinner, groups: list[Group], beatmap: Beatmap):
        """Parse a denden hit object.

        Args:
            spinner: Spinner object.
            groups: List of groups to add to.
        """
        self._add_group(
            EventType.DENDEN,
            groups,
            time=spinner.time,
            beatmap=beatmap,
            hitsound_ref_times=[spinner.time],
            hitsounds=[spinner.hitsound],
            additions=[spinner.addition],
            scroll_speed=self.scroll_speed_at(spinner.time, beatmap),
        )

        self._add_group(
            EventType.DENDEN_END,
            groups,
            time=spinner.end_time,
            beatmap=beatmap,
        )


AutoFeatureExtractor.register(CM3PConfig, CM3PBeatmapParser)

__all__ = ["CM3PBeatmapParser", "EventType", "Group", "load_beatmap", "get_song_length"]
