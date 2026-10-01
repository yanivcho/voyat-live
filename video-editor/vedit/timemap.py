"""Time map: source time (original video) <-> output time (edited video).

Every effect, caption and sound is anchored to a moment in the SOURCE (usually a word). The map converts
it to output time, and the same map drives both the picture and the audio, so everything stays in sync
whatever we cut, crossfade or freeze.
"""
import bisect
from dataclasses import dataclass


@dataclass
class Segment:
    src_in: float
    src_out: float
    hold: float = 0.0   # > 0: freeze frame of src_in, shown for `hold` seconds (audio is silent)
    xfade: float = 0.0  # soft transition: overlap with the previous segment, in seconds

    @property
    def dur(self):
        return self.hold if self.hold > 0 else self.src_out - self.src_in


class TimeMap:
    def __init__(self, segments):
        self.segs = [Segment(s.src_in, s.src_out, s.hold, s.xfade) for s in segments if s.dur > 1e-6]
        if not self.segs:
            raise ValueError("time map is empty")
        self.starts, t = [], 0.0
        for i, s in enumerate(self.segs):
            if i == 0:
                s.xfade = 0.0
            else:
                s.xfade = max(0.0, min(s.xfade, 0.5 * s.dur, 0.5 * self.segs[i - 1].dur))
                t -= s.xfade
            self.starts.append(t)
            t += s.dur
        self.duration = t

    @classmethod
    def identity(cls, duration):
        return cls([Segment(0.0, duration)])

    def _src(self, i, t_out):
        s = self.segs[i]
        return s.src_in if s.hold > 0 else s.src_in + (t_out - self.starts[i])

    def sources(self, t_out):
        """[(segment index, source time, weight)] - two entries while inside a soft transition."""
        t_out = min(max(t_out, 0.0), self.duration)
        i = max(0, bisect.bisect_right(self.starts, t_out) - 1)
        s = self.segs[i]
        if i > 0 and s.xfade > 0 and t_out < self.starts[i] + s.xfade:
            a = (t_out - self.starts[i]) / s.xfade
            return [(i - 1, self._src(i - 1, t_out), 1.0 - a), (i, self._src(i, t_out), a)]
        return [(i, self._src(i, t_out), 1.0)]

    def to_out(self, t_src, side="nearest"):
        """Output time of a source moment. If that moment was cut out, snap to the closest kept moment
        ('nearest'), the next one ('next') or the previous one ('prev')."""
        for i, s in enumerate(self.segs):
            if s.hold == 0 and s.src_in - 1e-6 <= t_src <= s.src_out + 1e-6:
                return self.starts[i] + (t_src - s.src_in)
        cands = []
        for i, s in enumerate(self.segs):
            if s.hold:
                continue
            if s.src_in > t_src:
                cands.append((s.src_in - t_src, "next", self.starts[i]))
            if s.src_out < t_src:
                cands.append((t_src - s.src_out, "prev", self.starts[i] + s.dur))
        if side != "nearest":
            cands = [c for c in cands if c[1] == side] or cands
        return min(cands)[2] if cands else None

    def cut_points(self):
        """Output times where a new segment begins (for markers in contact sheets)."""
        return [self.starts[i] for i in range(1, len(self.segs))]

    def freeze(self, t_src, hold):
        """New map with a freeze frame of `hold` seconds inserted at source time t_src."""
        out = []
        for s in self.segs:
            if s.hold == 0 and s.src_in < t_src < s.src_out:
                out += [Segment(s.src_in, t_src, xfade=s.xfade), Segment(t_src, t_src, hold=hold), Segment(t_src, s.src_out)]
            else:
                out.append(s)
        return TimeMap(out)


def silence_cut(duration, speech, min_silence=0.40, keep_gap=0.20, xfade=0.12, lead=0.15, tail=0.45):
    """Keep speech, remove pauses longer than `min_silence`.

    speech: sorted [(start, end)] in source seconds. Every removed pause leaves `keep_gap` seconds of
    perceived silence, of which `xfade` is a soft transition (video dissolve + equal-power audio crossfade).
    Returns (TimeMap, report).
    """
    if not speech:
        return TimeMap.identity(duration), dict(cuts=0, removed=0.0, note="no speech detected - nothing cut")
    pad = (keep_gap + xfade) / 2
    min_silence = max(min_silence, 2 * pad + 0.05)
    keep = [[max(0.0, speech[0][0] - lead), speech[0][1]]]
    for s, e in speech[1:]:
        if s - keep[-1][1] < min_silence:
            keep[-1][1] = max(keep[-1][1], e)
        else:
            keep[-1][1] += pad
            keep.append([s - pad, e])
    keep[-1][1] = min(duration, keep[-1][1] + tail)
    segs = [Segment(a, b, xfade=xfade if i else 0.0) for i, (a, b) in enumerate(keep)]
    tm = TimeMap(segs)
    return tm, dict(cuts=len(segs) - 1, removed=round(float(duration - tm.duration), 3))
