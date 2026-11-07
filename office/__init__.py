"""
Office activity monitor: who is in the room, and what is each person doing?

    frame -> perception  people, 17 body keypoints each, a stable id per person
          -> posture     sitting / standing / walking, smoothed over time
          -> events      a timeline: P3 appeared 10:42, sat 10:42-10:58, left 10:59

office_monitor.py at the repository root runs the whole thing.
"""
